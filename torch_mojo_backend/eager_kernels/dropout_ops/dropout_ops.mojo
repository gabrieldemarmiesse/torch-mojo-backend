# ===----------------------------------------------------------------------=== #
# Thin eager-mode native-dropout bridge for mojo_device (float32 GPU).
#
# Device-kernel bodies live in the Fable-owned internal module imported below.
# This Python-visible module only unpacks the pointer ABI, reconstructs the
# full-width RNG seed/counter, and enqueues work on the caller's DeviceContext.
# It performs no host reads or synchronization.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python._cpython import PyObjectPtr, Py_ssize_t

from native_dropout_kernels import (
    enqueue_native_dropout_backward_f32,
    enqueue_native_dropout_f32,
)
from op_utils import (
    _make_ptr,
    _raw_ctx,
    _raw_f64,
    _raw_int,
    _spec_dispatcher6,
    _spec_dispatcher10,
)

from variant_gates import _op_on, _register_call


@always_inline
def _join_u64(lo: Int, hi: Int) -> UInt64:
    return UInt64(lo) | (UInt64(hi) << 32)


def _native_dropout_go(
    output_ptr_obj: PyObjectPtr,
    mask_ptr_obj: PyObjectPtr,
    input_ptr_obj: PyObjectPtr,
    elements_obj: PyObjectPtr,
    p_obj: PyObjectPtr,
    seed_lo_obj: PyObjectPtr,
    seed_hi_obj: PyObjectPtr,
    offset_lo_obj: PyObjectPtr,
    offset_hi_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var output = _make_ptr[DType.float32](
        _raw_int(output_ptr_obj)
    ).as_unsafe_any_origin()
    var mask = _make_ptr[DType.bool](
        _raw_int(mask_ptr_obj)
    ).as_unsafe_any_origin()
    var input = _make_ptr[DType.float32](
        _raw_int(input_ptr_obj)
    ).as_unsafe_any_origin()
    var seed = _join_u64(_raw_int(seed_lo_obj), _raw_int(seed_hi_obj))
    var base_offset = _join_u64(
        _raw_int(offset_lo_obj), _raw_int(offset_hi_obj)
    )
    var ctx = _raw_ctx(device_context_ptr)
    enqueue_native_dropout_f32(
        output,
        mask,
        input,
        _raw_int(elements_obj),
        _raw_f64(p_obj),
        seed,
        base_offset,
        ctx,
    )


def _native_dropout_backward_go(
    grad_input_ptr_obj: PyObjectPtr,
    grad_output_ptr_obj: PyObjectPtr,
    mask_ptr_obj: PyObjectPtr,
    elements_obj: PyObjectPtr,
    scale_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var grad_input = _make_ptr[DType.float32](
        _raw_int(grad_input_ptr_obj)
    ).as_unsafe_any_origin()
    var grad_output = _make_ptr[DType.float32](
        _raw_int(grad_output_ptr_obj)
    ).as_unsafe_any_origin()
    var mask = _make_ptr[DType.bool](
        _raw_int(mask_ptr_obj)
    ).as_unsafe_any_origin()
    var ctx = _raw_ctx(device_context_ptr)
    enqueue_native_dropout_backward_f32(
        grad_input,
        grad_output,
        mask,
        _raw_int(elements_obj),
        _raw_f64(scale_obj),
        ctx,
    )


@export
def PyInit_dropout_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("dropout_ops")
        comptime if _op_on["NativeDropoutF32"]():
            _register_call(
                b,
                _spec_dispatcher10[_native_dropout_go, "NativeDropoutF32"],
                docstring=(
                    "(output_ptr, mask_ptr, input_ptr, elements, p, seed_lo,"
                    " seed_hi, offset_lo, offset_hi, context_ptr); float32"
                    " native dropout forward"
                ),
            )
        comptime if _op_on["NativeDropoutBackwardF32"]():
            _register_call(
                b,
                _spec_dispatcher6[
                    _native_dropout_backward_go, "NativeDropoutBackwardF32"
                ],
                docstring=(
                    "(grad_input_ptr, grad_output_ptr, mask_ptr, elements,"
                    " scale, context_ptr); float32 native dropout backward"
                ),
            )
        return b.finalize()
    except e:
        abort(t"failed to create dropout_ops python module: {e}")
