"""Thin eager bridge for runtime-dynamic FP32 native LayerNorm forward.

The device kernel lives in ``normalization_forward``.  This module only
validates and converts the raw Python pointer ABI, then enqueues on the
caller's supplied DeviceContext.  It performs no allocation, host read,
synchronization, or vendor-library call.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python._cpython import PyObjectPtr, Py_ssize_t

from normalization_forward_kernels import enqueue_layer_norm_forward_f32
from op_utils import (
    _make_ptr,
    _raw_ctx,
    _raw_f64,
    _raw_int,
    _spec_dispatcher12,
)

from variant_gates import _op_on, _register_call


def _layer_norm_forward_go(
    output_obj: PyObjectPtr,
    mean_obj: PyObjectPtr,
    rstd_obj: PyObjectPtr,
    input_obj: PyObjectPtr,
    weight_obj: PyObjectPtr,
    bias_obj: PyObjectPtr,
    rows_obj: PyObjectPtr,
    cols_obj: PyObjectPtr,
    epsilon_obj: PyObjectPtr,
    has_weight_obj: PyObjectPtr,
    has_bias_obj: PyObjectPtr,
    context_obj: PyObjectPtr,
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
        raise Error("LayerNorm rows and cols must be positive")
    if output_addr == 0 or mean_addr == 0 or rstd_addr == 0 or input_addr == 0:
        raise Error("LayerNorm required pointers must be nonzero")
    if has_weight and weight_addr == 0:
        raise Error("LayerNorm enabled weight pointer must be nonzero")
    if has_bias and bias_addr == 0:
        raise Error("LayerNorm enabled bias pointer must be nonzero")
    var ctx = _raw_ctx(context_obj)
    if ctx.api() == "cpu":
        raise Error("optimized FP32 LayerNorm requires an accelerator device")

    enqueue_layer_norm_forward_f32(
        _make_ptr[DType.float32](output_addr).as_unsafe_any_origin(),
        _make_ptr[DType.float32](mean_addr).as_unsafe_any_origin(),
        _make_ptr[DType.float32](rstd_addr).as_unsafe_any_origin(),
        _make_ptr[DType.float32](input_addr).as_unsafe_any_origin(),
        _make_ptr[DType.float32](weight_addr).as_unsafe_any_origin(),
        _make_ptr[DType.float32](bias_addr).as_unsafe_any_origin(),
        rows,
        cols,
        Float32(_raw_f64(epsilon_obj)),
        has_weight,
        has_bias,
        ctx,
    )


@export
def PyInit_normalization_forward_ops() abi("C") -> PythonObject:
    try:
        var builder = PythonModuleBuilder("normalization_forward_ops")
        comptime if _op_on["LayerNormForwardF32"]():
            _register_call(
                builder,
                _spec_dispatcher12[
                    _layer_norm_forward_go, "LayerNormForwardF32"
                ],
                docstring=(
                    "(output, mean, rstd, input, weight, bias, rows, cols, "
                    "epsilon, has_weight, has_bias, context); runtime-dynamic "
                    "FP32 native LayerNorm forward"
                ),
            )
        return builder.finalize()
    except e:
        abort(t"failed to create normalization_forward_ops python module: {e}")
