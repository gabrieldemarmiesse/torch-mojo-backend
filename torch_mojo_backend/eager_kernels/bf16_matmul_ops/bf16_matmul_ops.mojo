# ===----------------------------------------------------------------------=== #
# Thin eager-mode bridge for the BF16 GEMM and BMM paths.
#
# Device-kernel bodies and dynamic fallback routing live in the internal v3
# module imported below. This Python-visible module only unpacks the runtime
# pointer/layout ABI and enqueues on the caller's DeviceContext. It performs no
# allocation, host read, or synchronization.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python._cpython import PyObjectPtr, Py_ssize_t

from bf16_gemm_v3_kernels import enqueue_bf16_bmm, enqueue_bf16_gemm
from op_utils import (
    _make_ptr,
    _raw_ctx,
    _raw_int,
    _raw_ret_none,
    _spec_dispatcher11,
    _spec_dispatcher13,
    _spec_unsupported,
)

from variant_gates import _op_on, _register_call


def _bf16_gemm_go(
    output_ptr_obj: PyObjectPtr,
    a_ptr_obj: PyObjectPtr,
    b_ptr_obj: PyObjectPtr,
    bias_ptr_obj: PyObjectPtr,
    m_obj: PyObjectPtr,
    n_obj: PyObjectPtr,
    k_obj: PyObjectPtr,
    transpose_a_obj: PyObjectPtr,
    transpose_b_obj: PyObjectPtr,
    has_bias_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var output = _make_ptr[DType.bfloat16](
        _raw_int(output_ptr_obj)
    ).as_unsafe_any_origin()
    var a = _make_ptr[DType.bfloat16](
        _raw_int(a_ptr_obj)
    ).as_unsafe_any_origin()
    var b = _make_ptr[DType.bfloat16](
        _raw_int(b_ptr_obj)
    ).as_unsafe_any_origin()
    var bias = _make_ptr[DType.bfloat16](
        _raw_int(bias_ptr_obj)
    ).as_unsafe_any_origin()
    var ctx = _raw_ctx(device_context_ptr)
    enqueue_bf16_gemm(
        output,
        a,
        b,
        bias,
        _raw_int(m_obj),
        _raw_int(n_obj),
        _raw_int(k_obj),
        _raw_int(transpose_a_obj) != 0,
        _raw_int(transpose_b_obj) != 0,
        _raw_int(has_bias_obj) != 0,
        ctx,
    )


def _bf16_bmm_go(
    output_ptr_obj: PyObjectPtr,
    a_ptr_obj: PyObjectPtr,
    b_ptr_obj: PyObjectPtr,
    batch_count_obj: PyObjectPtr,
    m_obj: PyObjectPtr,
    n_obj: PyObjectPtr,
    k_obj: PyObjectPtr,
    output_batch_stride_obj: PyObjectPtr,
    a_batch_stride_obj: PyObjectPtr,
    b_batch_stride_obj: PyObjectPtr,
    transpose_a_obj: PyObjectPtr,
    transpose_b_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var output = _make_ptr[DType.bfloat16](
        _raw_int(output_ptr_obj)
    ).as_unsafe_any_origin()
    var a = _make_ptr[DType.bfloat16](
        _raw_int(a_ptr_obj)
    ).as_unsafe_any_origin()
    var b = _make_ptr[DType.bfloat16](
        _raw_int(b_ptr_obj)
    ).as_unsafe_any_origin()
    var ctx = _raw_ctx(device_context_ptr)
    enqueue_bf16_bmm(
        output,
        a,
        b,
        _raw_int(batch_count_obj),
        _raw_int(m_obj),
        _raw_int(n_obj),
        _raw_int(k_obj),
        _raw_int(output_batch_stride_obj),
        _raw_int(a_batch_stride_obj),
        _raw_int(b_batch_stride_obj),
        _raw_int(transpose_a_obj) != 0,
        _raw_int(transpose_b_obj) != 0,
        ctx,
    )


@export
def PyInit_bf16_matmul_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("bf16_matmul_ops")
        comptime if _op_on["Bf16BmmBF16"]():
            _register_call(
                b,
                _spec_dispatcher13[_bf16_bmm_go, "Bf16BmmBF16"],
                docstring=(
                    "(output_ptr, a_ptr, b_ptr, batch_count, m, n, k,"
                    " output_batch_stride, a_batch_stride, b_batch_stride,"
                    " transpose_a, transpose_b, context_ptr); opt-in"
                    " BF16 strided BMM"
                ),
            )
        comptime if _op_on["Bf16GemmBF16"]():
            _register_call(
                b,
                _spec_dispatcher11[_bf16_gemm_go, "Bf16GemmBF16"],
                docstring=(
                    "(output_ptr, a_ptr, b_ptr, bias_ptr, m, n, k, transpose_a,"
                    " transpose_b, has_bias, context_ptr); BF16 2-D GEMM"
                ),
            )
        return b.finalize()
    except e:
        abort(t"failed to create bf16_matmul_ops python module: {e}")
