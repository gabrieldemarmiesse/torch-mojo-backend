# ===----------------------------------------------------------------------=== #
# Thin eager-mode LayerNorm-backward bridge for mojo_device (float32 GPU).
#
# Device-kernel bodies live in the two Fable-owned internal modules imported
# below.  This Python-visible module only unpacks the pointer ABI, selects the
# requested ATen outputs, and enqueues work on the caller's DeviceContext.
# It performs no host reads or synchronization.
# ===----------------------------------------------------------------------=== #

from max.gpu.host import DeviceContext
from std.os import abort

from normalization_backward_dx import enqueue_layer_norm_backward_dx_f32
from normalization_backward_params import (
    enqueue_layer_norm_backward_params_f32,
)
from op_utils import (
    Arg,
    Argv,
    _make_ptr,
    _raw_ctx,
    _raw_int,
)

from variant_gates import ErrBuf, NO_OP_COMPILED, _op_on, _tmb_entry_error


def enqueue_layer_norm_backward_f32(
    grad_input: Pointer[Scalar[DType.float32], MutAnyOrigin],
    grad_weight: Pointer[Scalar[DType.float32], MutAnyOrigin],
    grad_bias: Pointer[Scalar[DType.float32], MutAnyOrigin],
    grad_output: Pointer[Scalar[DType.float32], MutAnyOrigin],
    input: Pointer[Scalar[DType.float32], MutAnyOrigin],
    mean: Pointer[Scalar[DType.float32], MutAnyOrigin],
    rstd: Pointer[Scalar[DType.float32], MutAnyOrigin],
    weight: Pointer[Scalar[DType.float32], MutAnyOrigin],
    rows: Int,
    cols: Int,
    output_mask: Int,
    ctx: DeviceContext,
) raises:
    if rows <= 0 or cols <= 0:
        return

    var want_input = (output_mask & 1) != 0
    var want_weight = (output_mask & 2) != 0
    var want_bias = (output_mask & 4) != 0

    if want_input:
        enqueue_layer_norm_backward_dx_f32(
            grad_input,
            grad_output,
            input,
            mean,
            rstd,
            weight,
            rows,
            cols,
            Int(weight) != 0,
            ctx,
        )

    if want_weight or want_bias:
        enqueue_layer_norm_backward_params_f32(
            grad_weight,
            grad_bias,
            grad_output,
            input,
            mean,
            rstd,
            rows,
            cols,
            want_weight,
            want_bias,
            ctx,
        )


def _layer_norm_backward_go(
    grad_input_ptr_obj: Arg,
    grad_weight_ptr_obj: Arg,
    grad_bias_ptr_obj: Arg,
    grad_output_ptr_obj: Arg,
    input_ptr_obj: Arg,
    mean_ptr_obj: Arg,
    rstd_ptr_obj: Arg,
    weight_ptr_obj: Arg,
    rows_obj: Arg,
    cols_obj: Arg,
    output_mask_obj: Arg,
    device_context_ptr: Arg,
) raises:
    var grad_input = _make_ptr[DType.float32](
        _raw_int(grad_input_ptr_obj)
    ).as_unsafe_any_origin()
    var grad_weight = _make_ptr[DType.float32](
        _raw_int(grad_weight_ptr_obj)
    ).as_unsafe_any_origin()
    var grad_bias = _make_ptr[DType.float32](
        _raw_int(grad_bias_ptr_obj)
    ).as_unsafe_any_origin()
    var grad_output = _make_ptr[DType.float32](
        _raw_int(grad_output_ptr_obj)
    ).as_unsafe_any_origin()
    var input = _make_ptr[DType.float32](
        _raw_int(input_ptr_obj)
    ).as_unsafe_any_origin()
    var mean = _make_ptr[DType.float32](
        _raw_int(mean_ptr_obj)
    ).as_unsafe_any_origin()
    var rstd = _make_ptr[DType.float32](
        _raw_int(rstd_ptr_obj)
    ).as_unsafe_any_origin()
    var weight = _make_ptr[DType.float32](
        _raw_int(weight_ptr_obj)
    ).as_unsafe_any_origin()
    var ctx = _raw_ctx(device_context_ptr)
    enqueue_layer_norm_backward_f32(
        grad_input,
        grad_weight,
        grad_bias,
        grad_output,
        input,
        mean,
        rstd,
        weight,
        _raw_int(rows_obj),
        _raw_int(cols_obj),
        _raw_int(output_mask_obj),
        ctx,
    )


def _layer_norm_backward_dispatcher(argv: Argv, argc: Int) raises:
    var args = argv
    _layer_norm_backward_go(
        args[unsafe_offset=0],
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        args[unsafe_offset=5],
        args[unsafe_offset=6],
        args[unsafe_offset=7],
        args[unsafe_offset=8],
        args[unsafe_offset=9],
        args[unsafe_offset=10],
        args[unsafe_offset=11],
    )


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["LayerNormBackwardF32"]():
            _layer_norm_backward_dispatcher(argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
