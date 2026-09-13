"""Thin eager bridge for the runtime-dynamic normalization forward kernels.

The device kernels live in ``normalization_forward_kernels`` (layer / group
norm) and ``batch_norm_kernels``; this module only validates and converts the
raw Python pointer ABI, then enqueues on the caller's supplied DeviceContext.
It performs no allocation, host read, synchronization, or vendor-library call.

Four operations over two kernel families.  Layer norm and group norm are the
SAME kernels, differing only in how the affine parameters are indexed (a
comptime parameter, so each build carries exactly one of them):

  ``LayerNormForward``  gamma[j] / beta[j]
  ``GroupNormForward``  gamma[g*cpg + j//hxw] / beta[...]

Batch norm splits by where its statistics come from, sharing its elementwise
pass between the two — and, with layer/group norm and
``aten.var.correction``, the shifted moment scan in ``op_utils``:

  ``BatchNormInfer``  running statistics; the op is purely elementwise
  ``BatchNormTrain``  per-channel statistics over N*HxW, then that same pass
"""

from std.os import abort

from batch_norm_kernels import (
    enqueue_batch_norm_elementwise,
    enqueue_batch_norm_stats,
)
from normalization_forward_kernels import (
    AFFINE_CHAN,
    AFFINE_COL,
    enqueue_norm_rows,
)
from op_utils import (
    Arg,
    Argv,
    FLOAT_DTYPES,
    _raw_ctx,
    _raw_f64,
    _raw_int,
    _raw_tuple_f64,
    _raw_tuple_int,
    _spec_dispatcher10,
    _spec_dispatcher15,
    _spec_dispatcher8,
)

from variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)


def _norm_rows_go[
    affine: Int
](
    output_obj: Arg,
    mean_obj: Arg,
    rstd_obj: Arg,
    input_obj: Arg,
    weight_obj: Arg,
    bias_obj: Arg,
    rows_obj: Arg,
    cols_obj: Arg,
    epsilon_obj: Arg,
    has_weight_obj: Arg,
    has_bias_obj: Arg,
    hxw_obj: Arg,
    cpg_obj: Arg,
    group_obj: Arg,
    context_obj: Arg,
) raises:
    var output_addr = _raw_int(output_obj)
    var mean_addr = _raw_int(mean_obj)
    var rstd_addr = _raw_int(rstd_obj)
    var input_addr = _raw_int(input_obj)
    var weight_addr = _raw_int(weight_obj)
    var bias_addr = _raw_int(bias_obj)
    var rows = _raw_int(rows_obj)
    var cols = _raw_int(cols_obj)
    var has_weight = _raw_int(has_weight_obj) != 0
    var has_bias = _raw_int(has_bias_obj) != 0
    if rows <= 0 or cols <= 0:
        raise Error("normalization rows and cols must be positive")
    if output_addr == 0 or mean_addr == 0 or rstd_addr == 0 or input_addr == 0:
        raise Error("normalization required pointers must be nonzero")
    if has_weight and weight_addr == 0:
        raise Error("normalization enabled weight pointer must be nonzero")
    if has_bias and bias_addr == 0:
        raise Error("normalization enabled bias pointer must be nonzero")
    var ctx = _raw_ctx(context_obj)
    if ctx.api() == "cpu":
        raise Error("the normalization forward kernels require an accelerator")

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            handled = True
            enqueue_norm_rows[dt, affine](
                output_addr,
                mean_addr,
                rstd_addr,
                input_addr,
                weight_addr,
                bias_addr,
                rows,
                cols,
                Float32(_raw_f64(epsilon_obj)),
                _raw_int(hxw_obj),
                _raw_int(cpg_obj),
                _raw_int(group_obj),
                has_weight,
                has_bias,
                ctx,
            )
    if not handled:
        raise Error("unsupported dtype for the normalization forward kernels")


def _layer_norm_forward_go(
    output_obj: Arg,
    mean_obj: Arg,
    rstd_obj: Arg,
    input_obj: Arg,
    weight_obj: Arg,
    bias_obj: Arg,
    rows_obj: Arg,
    cols_obj: Arg,
    epsilon_obj: Arg,
    has_weight_obj: Arg,
    has_bias_obj: Arg,
    hxw_obj: Arg,
    cpg_obj: Arg,
    group_obj: Arg,
    context_obj: Arg,
) raises:
    _norm_rows_go[AFFINE_COL](
        output_obj,
        mean_obj,
        rstd_obj,
        input_obj,
        weight_obj,
        bias_obj,
        rows_obj,
        cols_obj,
        epsilon_obj,
        has_weight_obj,
        has_bias_obj,
        hxw_obj,
        cpg_obj,
        group_obj,
        context_obj,
    )


def _group_norm_forward_go(
    output_obj: Arg,
    mean_obj: Arg,
    rstd_obj: Arg,
    input_obj: Arg,
    weight_obj: Arg,
    bias_obj: Arg,
    rows_obj: Arg,
    cols_obj: Arg,
    epsilon_obj: Arg,
    has_weight_obj: Arg,
    has_bias_obj: Arg,
    hxw_obj: Arg,
    cpg_obj: Arg,
    group_obj: Arg,
    context_obj: Arg,
) raises:
    _norm_rows_go[AFFINE_CHAN](
        output_obj,
        mean_obj,
        rstd_obj,
        input_obj,
        weight_obj,
        bias_obj,
        rows_obj,
        cols_obj,
        epsilon_obj,
        has_weight_obj,
        has_bias_obj,
        hxw_obj,
        cpg_obj,
        group_obj,
        context_obj,
    )


def _batch_norm_infer_go(
    out_obj: Arg,
    in_obj: Arg,
    mean_obj: Arg,
    var_obj: Arg,
    weight_obj: Arg,
    bias_obj: Arg,
    params: Arg,  # (eps, channels, inner, planes, has_weight,
    #  has_bias, save_mean_addr, save_invstd_addr)
    context_obj: Arg,
) raises:
    """`aten::_native_batch_norm_legit_no_training` / eval-mode batch norm."""
    var out_addr = _raw_int(out_obj)
    var in_addr = _raw_int(in_obj)
    var mean_addr = _raw_int(mean_obj)
    var var_addr = _raw_int(var_obj)
    var weight_addr = _raw_int(weight_obj)
    var bias_addr = _raw_int(bias_obj)
    var eps = Float32(_raw_tuple_f64(params, 0))
    var channels = _raw_tuple_int(params, 1)
    var inner = _raw_tuple_int(params, 2)
    var planes = _raw_tuple_int(params, 3)
    var has_weight = _raw_tuple_int(params, 4) != 0
    var has_bias = _raw_tuple_int(params, 5) != 0
    # torch fills both saved statistics on this route (Normalization.cu:454);
    # the elementwise kernel emits them from the prologue it already runs.
    var save_mean_addr = _raw_tuple_int(params, 6)
    var save_invstd_addr = _raw_tuple_int(params, 7)
    if channels <= 0 or inner <= 0 or planes <= 0:
        raise Error("batch norm geometry must be positive")
    if out_addr == 0 or in_addr == 0 or mean_addr == 0 or var_addr == 0:
        raise Error("batch norm required pointers must be nonzero")
    if save_mean_addr == 0 or save_invstd_addr == 0:
        raise Error("batch norm inference saved-stat pointers must be nonzero")
    if (has_weight and weight_addr == 0) or (has_bias and bias_addr == 0):
        raise Error("batch norm enabled affine pointers must be nonzero")
    var ctx = _raw_ctx(context_obj)
    if ctx.api() == "cpu":
        raise Error("the batch norm forward kernels require an accelerator")

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime for st in FLOAT_DTYPES:
            comptime for pt in FLOAT_DTYPES:
                comptime if (
                    _dtype_arg_on[0, dt]()
                    and _dtype_arg_on[1, st]()
                    and _dtype_arg_on[2, pt]()
                ):
                    handled = True
                    enqueue_batch_norm_elementwise[dt, pt, st, False](
                        out_addr,
                        in_addr,
                        mean_addr,
                        var_addr,
                        weight_addr,
                        bias_addr,
                        eps,
                        channels,
                        inner,
                        planes,
                        has_weight,
                        has_bias,
                        ctx,
                        save_mean_addr,
                        save_invstd_addr,
                    )
    if not handled:
        raise Error("unsupported dtype combination for batch norm inference")


def _batch_norm_train_go(
    out_obj: Arg,
    save_mean_obj: Arg,
    save_invstd_obj: Arg,
    in_obj: Arg,
    weight_obj: Arg,
    bias_obj: Arg,
    run_mean_obj: Arg,
    run_var_obj: Arg,
    # (eps, momentum, channels, runs, hxw, has_weight, has_bias, has_running)
    params: Arg,
    context_obj: Arg,
) raises:
    """`aten::native_batch_norm` with `training=True`: per-channel statistics
    over N*HxW, the ATen running-stat update, then the elementwise pass."""
    var out_addr = _raw_int(out_obj)
    var save_mean_addr = _raw_int(save_mean_obj)
    var save_invstd_addr = _raw_int(save_invstd_obj)
    var in_addr = _raw_int(in_obj)
    var weight_addr = _raw_int(weight_obj)
    var bias_addr = _raw_int(bias_obj)
    var run_mean_addr = _raw_int(run_mean_obj)
    var run_var_addr = _raw_int(run_var_obj)
    var eps = Float32(_raw_tuple_f64(params, 0))
    var momentum = Float32(_raw_tuple_f64(params, 1))
    var channels = _raw_tuple_int(params, 2)
    var runs = _raw_tuple_int(params, 3)
    var hxw = _raw_tuple_int(params, 4)
    var has_weight = _raw_tuple_int(params, 5) != 0
    var has_bias = _raw_tuple_int(params, 6) != 0
    var has_running = _raw_tuple_int(params, 7) != 0
    if channels <= 0 or runs <= 0 or hxw <= 0:
        raise Error("batch norm geometry must be positive")
    if out_addr == 0 or in_addr == 0:
        raise Error("batch norm required pointers must be nonzero")
    if save_mean_addr == 0 or save_invstd_addr == 0:
        raise Error("batch norm saved-statistics pointers must be nonzero")
    if has_running and (run_mean_addr == 0 or run_var_addr == 0):
        raise Error("batch norm running-statistics pointers must be nonzero")
    if (has_weight and weight_addr == 0) or (has_bias and bias_addr == 0):
        raise Error("batch norm enabled affine pointers must be nonzero")
    var ctx = _raw_ctx(context_obj)
    if ctx.api() == "cpu":
        raise Error("the batch norm forward kernels require an accelerator")

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime for st in FLOAT_DTYPES:
            comptime for pt in FLOAT_DTYPES:
                comptime if (
                    _dtype_arg_on[0, dt]()
                    and _dtype_arg_on[1, st]()
                    and _dtype_arg_on[2, pt]()
                ):
                    handled = True
                    enqueue_batch_norm_stats[dt, st](
                        save_mean_addr,
                        save_invstd_addr,
                        run_mean_addr,
                        run_var_addr,
                        in_addr,
                        channels,
                        runs,
                        hxw,
                        eps,
                        momentum,
                        has_running,
                        ctx,
                    )
                    # The saved statistics are float32 whatever the input is
                    # (ATen's `acc_type`), and the second one is already the
                    # inverse standard deviation.
                    enqueue_batch_norm_elementwise[dt, pt, DType.float32, True](
                        out_addr,
                        in_addr,
                        save_mean_addr,
                        save_invstd_addr,
                        weight_addr,
                        bias_addr,
                        eps,
                        channels,
                        hxw,
                        runs * channels,
                        has_weight,
                        has_bias,
                        ctx,
                    )
    if not handled:
        raise Error("unsupported dtype combination for batch norm training")


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["LayerNormForward"]():
            _spec_dispatcher15[_layer_norm_forward_go, "LayerNormForward"](
                argv, argc
            )
            return 0
        comptime if _op_on["GroupNormForward"]():
            _spec_dispatcher15[_group_norm_forward_go, "GroupNormForward"](
                argv, argc
            )
            return 0
        comptime if _op_on["BatchNormInfer"]():
            _spec_dispatcher8[_batch_norm_infer_go, "BatchNormInfer"](
                argv, argc
            )
            return 0
        comptime if _op_on["BatchNormTrain"]():
            _spec_dispatcher10[_batch_norm_train_go, "BatchNormTrain"](
                argv, argc
            )
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
