# ===----------------------------------------------------------------------=== #
# Thin eager-mode embedding-backward bridge for mojo_device (float32 GPU).
#
# The device-kernel implementation is Fable-owned and imported below.  This
# Python-visible module only unpacks the pointer ABI and enqueues the complete
# operation on the caller's DeviceContext.  It performs no host reads or
# synchronization.
# ===----------------------------------------------------------------------=== #

from std.os import abort

from embedding_backward_kernels import (
    enqueue_embedding_dense_backward_f32_i64,
)
from op_utils import (
    Arg,
    Argv,
    _make_ptr,
    _raw_ctx,
    _raw_int,
    _spec_dispatcher9,
)

from variant_gates import ErrBuf, NO_OP_COMPILED, _op_on, _tmb_entry_error


def _embedding_dense_backward_go(
    grad_weight_ptr_obj: Arg,
    grad_output_ptr_obj: Arg,
    indices_ptr_obj: Arg,
    num_indices_obj: Arg,
    embedding_dim_obj: Arg,
    num_weights_obj: Arg,
    padding_idx_obj: Arg,
    scale_grad_by_freq_obj: Arg,
    device_context_ptr: Arg,
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
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["EmbeddingDenseBackwardF32I64"]():
            _spec_dispatcher9[
                _embedding_dense_backward_go,
                "EmbeddingDenseBackwardF32I64",
            ](argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
