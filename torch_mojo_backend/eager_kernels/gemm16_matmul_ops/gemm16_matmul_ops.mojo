# ===----------------------------------------------------------------------=== #
# Thin eager-mode bridge for the 16-bit tensor-core GEMM and BMM paths.
#
# Device-kernel bodies and dynamic fallback routing live in the internal v3
# module imported below. This Python-visible module only unpacks the runtime
# pointer/layout ABI and enqueues on the caller's DeviceContext. It performs no
# allocation, host read, or synchronization.
#
# The family serves bfloat16 AND float16 from one source: `_GEMM16_DT`
# (gemm16_dtype.mojo) resolves the operand dtype from the DTYPE_ARG_0 define,
# so the loader's existing per-dtype specialization is the whole mechanism and
# nothing dtype-dependent travels at runtime. The module, its files and its OP
# names still spell `bf16` for one reason worth writing down: the entry
# module's stem is how scripts/compare_kernel_asm.py pairs kernels across two
# trees, and renaming it in the same change that parametrized the dtype would
# have made the bfloat16 byte-invariance check unverifiable.
# ===----------------------------------------------------------------------=== #

from std.os import abort

from gemm16_v3_kernels import enqueue_gemm16_bmm, enqueue_gemm16_gemm
from gemm16_dtype import _GEMM16_DT
from op_utils import (
    Arg,
    Argv,
    _make_ptr,
    _raw_ctx,
    _raw_int,
    _spec_dispatcher11,
    _spec_dispatcher13,
)

from variant_gates import ErrBuf, NO_OP_COMPILED, _op_on, _tmb_entry_error


def _bf16_gemm_go(
    output_ptr_obj: Arg,
    a_ptr_obj: Arg,
    b_ptr_obj: Arg,
    bias_ptr_obj: Arg,
    m_obj: Arg,
    n_obj: Arg,
    k_obj: Arg,
    transpose_a_obj: Arg,
    transpose_b_obj: Arg,
    has_bias_obj: Arg,
    device_context_ptr: Arg,
) raises:
    var output = _make_ptr[_GEMM16_DT](
        _raw_int(output_ptr_obj)
    ).as_unsafe_any_origin()
    var a = _make_ptr[_GEMM16_DT](_raw_int(a_ptr_obj)).as_unsafe_any_origin()
    var b = _make_ptr[_GEMM16_DT](_raw_int(b_ptr_obj)).as_unsafe_any_origin()
    var bias = _make_ptr[_GEMM16_DT](
        _raw_int(bias_ptr_obj)
    ).as_unsafe_any_origin()
    var ctx = _raw_ctx(device_context_ptr)
    enqueue_gemm16_gemm(
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
    output_ptr_obj: Arg,
    a_ptr_obj: Arg,
    b_ptr_obj: Arg,
    batch_count_obj: Arg,
    m_obj: Arg,
    n_obj: Arg,
    k_obj: Arg,
    output_batch_stride_obj: Arg,
    a_batch_stride_obj: Arg,
    b_batch_stride_obj: Arg,
    transpose_a_obj: Arg,
    transpose_b_obj: Arg,
    device_context_ptr: Arg,
) raises:
    var output = _make_ptr[_GEMM16_DT](
        _raw_int(output_ptr_obj)
    ).as_unsafe_any_origin()
    var a = _make_ptr[_GEMM16_DT](_raw_int(a_ptr_obj)).as_unsafe_any_origin()
    var b = _make_ptr[_GEMM16_DT](_raw_int(b_ptr_obj)).as_unsafe_any_origin()
    var ctx = _raw_ctx(device_context_ptr)
    enqueue_gemm16_bmm(
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
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["Bmm16"]():
            _spec_dispatcher13[_bf16_bmm_go, "Bmm16"](argv, argc)
            return 0
        comptime if _op_on["Gemm16"]():
            _spec_dispatcher11[_bf16_gemm_go, "Gemm16"](argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
