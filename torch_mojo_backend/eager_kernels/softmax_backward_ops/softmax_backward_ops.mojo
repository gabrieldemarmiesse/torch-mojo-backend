# ===----------------------------------------------------------------------=== #
# Thin eager-mode log_softmax-backward bridge for mojo_device
# (float32/float16/bfloat16 GPU).
#
# The device-kernel implementation lives in softmax_backward_kernels.mojo.
# Same architecture as loss_ops.mojo / embedding_backward_ops.mojo: the
# Python-visible function gets raw integer pointers (tensor `._ptr`, offset
# pre-applied) plus shape/dtype ints and the device's DeviceContext pointer,
# and enqueues the fused kernel on the device queue (fire and forget, no
# sync, no host reads, no host allocation).
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python._cpython import PyObjectPtr, Py_ssize_t

from softmax_backward_kernels import enqueue_log_softmax_backward
from op_utils import (
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_int,
    _spec_dispatcher7,
)

from variant_gates import _dtype_arg_on, _op_on, _register_call


def _log_softmax_backward_go(
    grad_input_ptr_obj: PyObjectPtr,
    grad_output_ptr_obj: PyObjectPtr,
    output_ptr_obj: PyObjectPtr,
    rows_obj: PyObjectPtr,
    cols_obj: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype_val = _raw_dtype_int(dtype_obj)
    var grad_input_addr = _raw_int(grad_input_ptr_obj)
    var grad_output_addr = _raw_int(grad_output_ptr_obj)
    var output_addr = _raw_int(output_ptr_obj)
    var rows = _raw_int(rows_obj)
    var cols = _raw_int(cols_obj)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in [DType.float32, DType.float16, DType.bfloat16]:
        comptime if _dtype_arg_on[0, dt]():
            if dtype_val == dt:
                enqueue_log_softmax_backward[dt](
                    _make_ptr[dt](grad_input_addr).as_unsafe_any_origin(),
                    _make_ptr[dt](grad_output_addr)
                    .as_unsafe_any_origin()
                    .as_immutable(),
                    _make_ptr[dt](output_addr)
                    .as_unsafe_any_origin()
                    .as_immutable(),
                    rows,
                    cols,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error(
            "unsupported dtype for fused log_softmax backward: "
            + String(dtype_val)
        )


@export
def PyInit_softmax_backward_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("softmax_backward_ops")
        comptime if _op_on["LogSoftmaxBackwardData"]():
            _register_call(
                b,
                _spec_dispatcher7[
                    _log_softmax_backward_go, "LogSoftmaxBackwardData"
                ],
                docstring=(
                    "(grad_input_ptr, grad_output_ptr, output_ptr, rows, cols,"
                    " dtype, context_ptr); fused trailing-dim log_softmax"
                    " backward: grad_input = grad_output - exp(output) *"
                    " rowsum(grad_output); f32/f16/bf16, fp32 accumulation"
                ),
            )
        return b.finalize()
    except e:
        abort(t"failed to create softmax_backward_ops python module: {e}")
