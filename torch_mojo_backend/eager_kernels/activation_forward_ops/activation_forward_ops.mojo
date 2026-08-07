"""Thin eager bridge for runtime-dynamic BF16 GELU forward.

The optimized device kernel lives in ``activation_forward_kernels``.  This
module only validates and converts the raw Python pointer ABI, then enqueues
on the caller's supplied DeviceContext.  It performs no allocation, host
read, synchronization, or vendor-library call.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python._cpython import PyObjectPtr, Py_ssize_t

from activation_forward_kernels import enqueue_gelu_forward_bf16
from op_utils import (
    _make_ptr,
    _raw_ctx,
    _raw_int,
    _spec_dispatcher5,
)

from variant_gates import _op_on, _register_call


def _gelu_forward_bf16_go(
    output_obj: PyObjectPtr,
    input_obj: PyObjectPtr,
    elements_obj: PyObjectPtr,
    tanh_approx_obj: PyObjectPtr,
    context_obj: PyObjectPtr,
) raises:
    var output_addr = _raw_int(output_obj)
    var input_addr = _raw_int(input_obj)
    var elements = _raw_int(elements_obj)
    if elements < 0:
        raise Error("GELU elements must be nonnegative")
    if elements == 0:
        return
    if output_addr == 0 or input_addr == 0:
        raise Error("GELU pointers must be nonzero")
    var ctx = _raw_ctx(context_obj)
    if ctx.api() == "cpu":
        raise Error("optimized BF16 GELU requires an accelerator device")

    enqueue_gelu_forward_bf16(
        _make_ptr[DType.bfloat16](output_addr).as_unsafe_any_origin(),
        _make_ptr[DType.bfloat16](input_addr).as_unsafe_any_origin(),
        elements,
        _raw_int(tanh_approx_obj) != 0,
        ctx,
    )


@export
def PyInit_activation_forward_ops() abi("C") -> PythonObject:
    try:
        var builder = PythonModuleBuilder("activation_forward_ops")
        comptime if _op_on["GeluForwardBF16"]():
            _register_call(
                builder,
                _spec_dispatcher5[_gelu_forward_bf16_go, "GeluForwardBF16"],
                docstring=(
                    "(output, input, elements, tanh_approx, context); "
                    "runtime-dynamic BF16 GELU forward"
                ),
            )
        return builder.finalize()
    except e:
        abort(t"failed to create activation_forward_ops python module: {e}")
