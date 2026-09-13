# ===----------------------------------------------------------------------=== #
# Thin eager-mode native-dropout bridge for mojo_device (float32 GPU).
#
# Device-kernel bodies live in the Fable-owned internal module imported below.
# This Python-visible module only unpacks the pointer ABI, reconstructs the
# full-width RNG seed/counter, and enqueues work on the caller's DeviceContext.
# It performs no host reads or synchronization.
# ===----------------------------------------------------------------------=== #

from std.os import abort

from native_dropout_kernels import (
    enqueue_native_dropout_backward_f32,
    enqueue_native_dropout_f32,
)
from op_utils import (
    Arg,
    Argv,
    _make_ptr,
    _raw_ctx,
    _raw_f64,
    _raw_int,
    _spec_dispatcher10,
    _spec_dispatcher6,
)

from variant_gates import ErrBuf, NO_OP_COMPILED, _op_on, _tmb_entry_error


@always_inline
def _join_u64(lo: Int, hi: Int) -> UInt64:
    return UInt64(lo) | (UInt64(hi) << 32)


def _native_dropout_go(
    output_ptr_obj: Arg,
    mask_ptr_obj: Arg,
    input_ptr_obj: Arg,
    elements_obj: Arg,
    p_obj: Arg,
    seed_lo_obj: Arg,
    seed_hi_obj: Arg,
    offset_lo_obj: Arg,
    offset_hi_obj: Arg,
    device_context_ptr: Arg,
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
    grad_input_ptr_obj: Arg,
    grad_output_ptr_obj: Arg,
    mask_ptr_obj: Arg,
    elements_obj: Arg,
    scale_obj: Arg,
    device_context_ptr: Arg,
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
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["NativeDropoutF32"]():
            _spec_dispatcher10[_native_dropout_go, "NativeDropoutF32"](
                argv, argc
            )
            return 0
        comptime if _op_on["NativeDropoutBackwardF32"]():
            _spec_dispatcher6[
                _native_dropout_backward_go, "NativeDropoutBackwardF32"
            ](argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
