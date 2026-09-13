"""Thin eager bridge for runtime-dynamic BF16 GELU forward.

The optimized device kernel lives in ``activation_forward_kernels``.  This
module only validates and converts the raw Python pointer ABI, then enqueues
on the caller's supplied DeviceContext.  It performs no allocation, host
read, synchronization, or vendor-library call.
"""

from std.os import abort

from activation_forward_kernels import enqueue_gelu_forward_bf16
from op_utils import (
    Arg,
    Argv,
    _make_ptr,
    _raw_ctx,
    _raw_int,
    _spec_dispatcher5,
)

from variant_gates import ErrBuf, NO_OP_COMPILED, _op_on, _tmb_entry_error


def _gelu_forward_bf16_go(
    output_obj: Arg,
    input_obj: Arg,
    elements_obj: Arg,
    tanh_approx_obj: Arg,
    context_obj: Arg,
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
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["GeluForwardBF16"]():
            _spec_dispatcher5[_gelu_forward_bf16_go, "GeluForwardBF16"](
                argv, argc
            )
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
