# ===----------------------------------------------------------------------=== #
# Thin eager-mode bridge for the fused flash-attention forward and backward.
#
# Device-kernel bodies live in `flash_attention_fwd_kernels` and
# `flash_attention_bwd_kernels`.  This Python-visible module only unpacks the
# pointer ABI and enqueues on the caller's DeviceContext.  It performs no host
# read and no synchronization.
#
# The one thing it does beyond unpacking is zero dq/dk/dv, because the backward
# contract says those arrive zeroed and a caller that forgets would get a
# plausible-looking wrong gradient rather than a crash.  The cost is already
# inside the measured 0.998x: the harness zeroes on every timed iteration too.
# ===----------------------------------------------------------------------=== #

from std.gpu.host import DeviceBuffer
from std.os import abort
from std.gpu.memory import AddressSpace
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python._cpython import PyObjectPtr, Py_ssize_t

from flash_attention_bwd_kernels import enqueue_flash_attention_bwd
from flash_attention_fwd_kernels import RowStrides, enqueue_flash_attention_fwd
from op_utils import (
    FLOAT_DTYPES,
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _raw_ret_none,
    _raw_tuple_int,
    _spec_unsupported,
)


@always_inline
def _raw_strides(t: PyObjectPtr, i: Int) -> RowStrides:
    """The `i`-th (batch, head, seq) element-stride triple of a flat tuple."""
    return RowStrides(
        _raw_tuple_int(t, 3 * i),
        _raw_tuple_int(t, 3 * i + 1),
        _raw_tuple_int(t, 3 * i + 2),
    )


def _flash_attention_forward_go(
    out_obj: PyObjectPtr,
    lse_obj: PyObjectPtr,
    q_obj: PyObjectPtr,
    k_obj: PyObjectPtr,
    v_obj: PyObjectPtr,
    dims_obj: PyObjectPtr,
    strides_obj: PyObjectPtr,
    scale_obj: PyObjectPtr,
    causal_obj: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    ctx_obj: PyObjectPtr,
) raises:
    var batch = _raw_tuple_int(dims_obj, 0)
    var heads = _raw_tuple_int(dims_obj, 1)
    var seq_q = _raw_tuple_int(dims_obj, 2)
    var seq_kv = _raw_tuple_int(dims_obj, 3)
    var head_dim = _raw_tuple_int(dims_obj, 4)
    # (q, k, v), each (batch, head, seq) in elements. head_dim's stride must be
    # 1; the Python caller declines anything else.
    var q_st = _raw_strides(strides_obj, 0)
    var k_st = _raw_strides(strides_obj, 1)
    var v_st = _raw_strides(strides_obj, 2)
    var scale = Float32(_raw_f64(scale_obj))
    var causal = _raw_int(causal_obj) != 0
    var dtype = _raw_dtype_int(dtype_obj)
    var ctx = _raw_ctx(ctx_obj)
    var lse = _make_ptr[DType.float32](_raw_int(lse_obj))

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        if dtype == dt:
            enqueue_flash_attention_fwd[dt](
                _make_ptr[dt](_raw_int(out_obj)).as_unsafe_any_origin(),
                lse.as_unsafe_any_origin(),
                _make_ptr[dt](_raw_int(q_obj))
                .as_unsafe_any_origin()
                .as_immutable(),
                _make_ptr[dt](_raw_int(k_obj))
                .as_unsafe_any_origin()
                .as_immutable(),
                _make_ptr[dt](_raw_int(v_obj))
                .as_unsafe_any_origin()
                .as_immutable(),
                q_st,
                k_st,
                v_st,
                batch,
                heads,
                seq_q,
                seq_kv,
                head_dim,
                scale,
                causal,
                ctx,
            )
            handled = True
    if not handled:
        raise Error(
            "unsupported dtype for fused flash attention forward: "
            + String(dtype)
        )


def _flash_attention_backward_go(
    dq_obj: PyObjectPtr,
    dk_obj: PyObjectPtr,
    dv_obj: PyObjectPtr,
    grad_obj: PyObjectPtr,
    q_obj: PyObjectPtr,
    k_obj: PyObjectPtr,
    v_obj: PyObjectPtr,
    out_obj: PyObjectPtr,
    lse_obj: PyObjectPtr,
    dims_obj: PyObjectPtr,
    strides_obj: PyObjectPtr,
    scale_obj: PyObjectPtr,
    causal_obj: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    ctx_obj: PyObjectPtr,
) raises:
    var batch = _raw_tuple_int(dims_obj, 0)
    var heads = _raw_tuple_int(dims_obj, 1)
    var seq_q = _raw_tuple_int(dims_obj, 2)
    var seq_kv = _raw_tuple_int(dims_obj, 3)
    var head_dim = _raw_tuple_int(dims_obj, 4)
    # (grad_output, query, key, value, out_fwd), each (batch, head, seq) in
    # elements. head_dim's stride must be 1; the Python caller declines anything
    # else.
    var g_st = _raw_strides(strides_obj, 0)
    var q_st = _raw_strides(strides_obj, 1)
    var k_st = _raw_strides(strides_obj, 2)
    var v_st = _raw_strides(strides_obj, 3)
    var o_st = _raw_strides(strides_obj, 4)
    var scale = Float32(_raw_f64(scale_obj))
    var causal = _raw_int(causal_obj) != 0
    var dtype = _raw_dtype_int(dtype_obj)
    var ctx = _raw_ctx(ctx_obj)
    var lse = _make_ptr[DType.float32](_raw_int(lse_obj))

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        if dtype == dt:
            var dq = _make_ptr[dt](_raw_int(dq_obj))
            var dk = _make_ptr[dt](_raw_int(dk_obj))
            var dv = _make_ptr[dt](_raw_int(dv_obj))
            # The kernel may accumulate into these; the contract is that they
            # arrive zeroed.
            ctx.enqueue_memset(
                DeviceBuffer(
                    ctx,
                    dq.address_space_cast[AddressSpace.GENERIC](),
                    batch * heads * seq_q * head_dim,
                    owning=False,
                ),
                Scalar[dt](0),
            )
            ctx.enqueue_memset(
                DeviceBuffer(
                    ctx,
                    dk.address_space_cast[AddressSpace.GENERIC](),
                    batch * heads * seq_kv * head_dim,
                    owning=False,
                ),
                Scalar[dt](0),
            )
            ctx.enqueue_memset(
                DeviceBuffer(
                    ctx,
                    dv.address_space_cast[AddressSpace.GENERIC](),
                    batch * heads * seq_kv * head_dim,
                    owning=False,
                ),
                Scalar[dt](0),
            )
            enqueue_flash_attention_bwd[dt](
                dq.as_unsafe_any_origin(),
                dk.as_unsafe_any_origin(),
                dv.as_unsafe_any_origin(),
                _make_ptr[dt](_raw_int(grad_obj))
                .as_unsafe_any_origin()
                .as_immutable(),
                _make_ptr[dt](_raw_int(q_obj))
                .as_unsafe_any_origin()
                .as_immutable(),
                _make_ptr[dt](_raw_int(k_obj))
                .as_unsafe_any_origin()
                .as_immutable(),
                _make_ptr[dt](_raw_int(v_obj))
                .as_unsafe_any_origin()
                .as_immutable(),
                _make_ptr[dt](_raw_int(out_obj))
                .as_unsafe_any_origin()
                .as_immutable(),
                lse.as_unsafe_any_origin().as_immutable(),
                g_st,
                q_st,
                k_st,
                v_st,
                o_st,
                batch,
                heads,
                seq_q,
                seq_kv,
                head_dim,
                scale,
                causal,
                ctx,
            )
            handled = True
    if not handled:
        raise Error(
            "unsupported dtype for fused flash attention backward: "
            + String(dtype)
        )


def _flash_attention_forward_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 11:
            raise Error("FlashAttentionForward expects exactly 11 arguments")
        _flash_attention_forward_go(
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
        )
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def _flash_attention_backward_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 15:
            raise Error("FlashAttentionBackward expects exactly 15 arguments")
        _flash_attention_backward_go(
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
            args[13],
            args[14],
        )
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


@export
def PyInit_flash_attention_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("flash_attention_ops")
        b.def_py_c_function(
            _flash_attention_forward_dispatcher,
            "FlashAttentionForward",
            docstring=(
                "(out_ptr, lse_ptr, q_ptr, k_ptr, v_ptr, (batch, heads, seq_q,"
                " seq_kv, head_dim), (batch, head, seq) x 3 for q, k and v,"
                " scale, is_causal, dtype, context_ptr); fused flash-attention"
                " forward writing the output and the per-row log-sum-exp the"
                " backward consumes. Q, K and V are read through the given"
                " element strides and their head_dim stride must be 1; the"
                " output and the log-sum-exp are dense."
            ),
        )
        b.def_py_c_function(
            _flash_attention_backward_dispatcher,
            "FlashAttentionBackward",
            docstring=(
                "(dq_ptr, dk_ptr, dv_ptr, grad_out_ptr, q_ptr, k_ptr, v_ptr,"
                " out_ptr, lse_ptr, (batch, heads, seq_q, seq_kv, head_dim),"
                " (batch, head, seq) x 5 for grad_out, q, k, v and out, scale,"
                " is_causal, dtype, context_ptr); fused flash-attention"
                " backward writing dQ, dK and dV. The five read operands are"
                " addressed through the given element strides and their"
                " head_dim stride must be 1; dQ, dK, dV and the log-sum-exp are"
                " dense."
            ),
        )
        return b.finalize()
    except e:
        abort(t"failed to create flash_attention_ops python module: {e}")
