# ===----------------------------------------------------------------------=== #
# Thin eager-mode bridge for fused SDPA dropout/softmax backward.
#
# Device-kernel bodies live in the Fable-owned internal module imported below.
# This Python-visible module only validates and unpacks the pointer ABI, builds
# the optional mask pointer, and enqueues on the caller's DeviceContext.  It
# performs no allocation, host read, or synchronization.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python._cpython import PyObjectPtr, Py_ssize_t

from sdpa_dropout_softmax_backward_kernels import (
    enqueue_sdpa_dropout_softmax_backward,
)
from op_utils import (
    FLOAT_DTYPES,
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _raw_ret_none,
    _spec_unsupported,
)


def _sdpa_dropout_softmax_backward_go(
    output_ptr_obj: PyObjectPtr,
    probabilities_ptr_obj: PyObjectPtr,
    grad_after_dropout_ptr_obj: PyObjectPtr,
    mask_ptr_obj: PyObjectPtr,
    rows_obj: PyObjectPtr,
    cols_obj: PyObjectPtr,
    q_len_obj: PyObjectPtr,
    has_mask_obj: PyObjectPtr,
    causal_obj: PyObjectPtr,
    dropout_scale_obj: PyObjectPtr,
    score_scale_obj: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var output_address = _raw_int(output_ptr_obj)
    var probabilities_address = _raw_int(probabilities_ptr_obj)
    var grad_address = _raw_int(grad_after_dropout_ptr_obj)
    var mask_address = _raw_int(mask_ptr_obj)
    var has_mask = _raw_int(has_mask_obj) != 0
    if has_mask:
        if mask_address == 0:
            raise Error(
                "SDPADropoutSoftmaxBackward requires a non-null mask when"
                " has_mask is true"
            )
    elif mask_address != 0:
        raise Error(
            "SDPADropoutSoftmaxBackward requires a null mask when has_mask is"
            " false"
        )

    var rows = _raw_int(rows_obj)
    var cols = _raw_int(cols_obj)
    var q_len = _raw_int(q_len_obj)
    var causal = _raw_int(causal_obj) != 0
    var dropout_scale = _raw_f64(dropout_scale_obj)
    var score_scale = _raw_f64(score_scale_obj)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        if dtype == dt:
            var mask: Optional[
                UnsafePointer[Scalar[DType.bool], MutAnyOrigin]
            ] = None
            if has_mask:
                mask = _make_ptr[DType.bool](
                    mask_address
                ).as_unsafe_any_origin()
            enqueue_sdpa_dropout_softmax_backward[dt](
                _make_ptr[dt](output_address).as_unsafe_any_origin(),
                _make_ptr[dt](probabilities_address).as_unsafe_any_origin(),
                _make_ptr[dt](grad_address).as_unsafe_any_origin(),
                mask,
                rows,
                cols,
                q_len,
                has_mask,
                causal,
                dropout_scale,
                score_scale,
                ctx,
            )
            handled = True
    if not handled:
        raise Error(
            "unsupported dtype for fused SDPA softmax backward: "
            + String(dtype)
        )


def _sdpa_dropout_softmax_backward_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 13:
            raise Error(
                "SDPADropoutSoftmaxBackward expects exactly 13 arguments"
            )
        _sdpa_dropout_softmax_backward_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
            args[8],
            args[9],
            args[10],
            args[11],
            args[12],
        )
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


@export
def PyInit_sdpa_backward_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("sdpa_backward_ops")
        b.def_py_c_function(
            _sdpa_dropout_softmax_backward_dispatcher,
            "SDPADropoutSoftmaxBackward",
            docstring=(
                "(output_ptr, probabilities_ptr, grad_after_dropout_ptr,"
                " mask_ptr_or_zero, rows, cols, q_len, has_mask, causal,"
                " dropout_scale, score_scale, dtype, context_ptr); fused SDPA"
                " dropout and softmax backward, float32/bfloat16/float16"
            ),
        )
        return b.finalize()
    except e:
        abort(t"failed to create sdpa_backward_ops python module: {e}")
