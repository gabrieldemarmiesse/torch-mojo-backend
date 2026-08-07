# ===----------------------------------------------------------------------=== #
# Thin eager-mode embedding-backward bridge for mojo_device (float32 GPU).
#
# The device-kernel implementation is Fable-owned and imported below.  This
# Python-visible module only unpacks the pointer ABI and enqueues the complete
# operation on the caller's DeviceContext.  It performs no host reads or
# synchronization.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python._cpython import PyObjectPtr, Py_ssize_t

from embedding_backward_kernels import (
    enqueue_embedding_dense_backward_f32_i64,
)
from op_utils import (
    _make_ptr,
    _raw_ctx,
    _raw_int,
    _spec_dispatcher9,
)

from variant_gates import _op_on, _register_call


def _embedding_dense_backward_go(
    grad_weight_ptr_obj: PyObjectPtr,
    grad_output_ptr_obj: PyObjectPtr,
    indices_ptr_obj: PyObjectPtr,
    num_indices_obj: PyObjectPtr,
    embedding_dim_obj: PyObjectPtr,
    num_weights_obj: PyObjectPtr,
    padding_idx_obj: PyObjectPtr,
    scale_grad_by_freq_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var grad_weight = _make_ptr[DType.float32](
        _raw_int(grad_weight_ptr_obj)
    ).as_unsafe_any_origin()
    var grad_output = _make_ptr[DType.float32](
        _raw_int(grad_output_ptr_obj)
    ).as_unsafe_any_origin()
    var indices = _make_ptr[DType.int64](
        _raw_int(indices_ptr_obj)
    ).as_unsafe_any_origin()
    var ctx = _raw_ctx(device_context_ptr)
    enqueue_embedding_dense_backward_f32_i64(
        grad_weight,
        grad_output,
        indices,
        _raw_int(num_indices_obj),
        _raw_int(embedding_dim_obj),
        _raw_int(num_weights_obj),
        _raw_int(padding_idx_obj),
        _raw_int(scale_grad_by_freq_obj) != 0,
        ctx,
    )


@export
def PyInit_embedding_backward_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("embedding_backward_ops")
        comptime if _op_on["EmbeddingDenseBackwardF32I64"]():
            _register_call(
                b,
                _spec_dispatcher9[
                    _embedding_dense_backward_go,
                    "EmbeddingDenseBackwardF32I64",
                ],
                docstring=(
                    "(grad_weight_ptr, grad_output_ptr, indices_ptr,"
                    " num_indices, embedding_dim, num_weights, padding_idx,"
                    " scale_grad_by_freq, context_ptr); float32/int64 dense"
                    " embedding backward"
                ),
            )
        return b.finalize()
    except e:
        abort(t"failed to create embedding_backward_ops python module: {e}")
