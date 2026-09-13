# ===----------------------------------------------------------------------=== #
# Thin eager-mode bridge for the on-device random generators of mojo_device.
#
# Kernel bodies live in `uniform_kernels.mojo`. This Python-visible module only
# unpacks the raw pointer/scalar ABI, rebuilds the full-width Philox seed and
# counter from their 32-bit halves, and enqueues work on the caller's
# DeviceContext. It performs no host reads and no synchronization, so a draw
# stays asynchronous: the generator state it needs was reserved on the host
# before the call (`_reserve_philox_state`).
# ===----------------------------------------------------------------------=== #

from std.os import abort

from op_utils import (
    Arg,
    Argv,
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _spec_dispatcher10,
)
from uniform_kernels import enqueue_uniform

from variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_out_on,
    _op_on,
    _tmb_entry_error,
)


# The dtypes `aten::uniform_` is defined for, minus complex (unsupported
# device-wide): ATen dispatches it over the floating-point types plus the two
# 16-bit floats (`AT_DISPATCH_FLOATING_TYPES_AND2` in `uniform_impl_`).
comptime UNIFORM_DTYPES = [
    DType.float32,
    DType.bfloat16,
    DType.float16,
    DType.float64,
]


@always_inline
def _join_u64(lo: Int, hi: Int) -> UInt64:
    """Rejoin a 64-bit value split into 32-bit halves by the Python caller.

    Keeping every Python integer below 2**32 avoids `Int` overflow in
    the raw CPython bridge while preserving all 64 bits of seed and counter.
    """
    return UInt64(lo) | (UInt64(hi) << 32)


def _uniform_go(
    dst_ptr_obj: Arg,
    from_obj: Arg,
    to_obj: Arg,
    numel_obj: Arg,
    dtype_obj: Arg,
    seed_lo_obj: Arg,
    seed_hi_obj: Arg,
    offset_lo_obj: Arg,
    offset_hi_obj: Arg,
    device_context_ptr: Arg,
) raises:
    var dst_addr = _raw_int(dst_ptr_obj)
    var from_value = _raw_f64(from_obj)
    var to_value = _raw_f64(to_obj)
    var size = _raw_int(numel_obj)
    var dtype = _raw_dtype_int(dtype_obj)
    var seed = _join_u64(_raw_int(seed_lo_obj), _raw_int(seed_hi_obj))
    var base_offset = _join_u64(
        _raw_int(offset_lo_obj), _raw_int(offset_hi_obj)
    )
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False

    comptime for dt in UNIFORM_DTYPES:
        comptime if _dtype_out_on[0, dt]():
            if dtype == dt:
                enqueue_uniform[dt](
                    dst_addr, from_value, to_value, size, seed, base_offset, ctx
                )
                handled = True
    if not handled:
        # A miss means Python selected the wrong immutable specialization.
        raise Error("unsupported dtype for on-device uniform_: ", dtype)


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["UniformFill"]():
            _spec_dispatcher10[_uniform_go, "UniformFill"](argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
