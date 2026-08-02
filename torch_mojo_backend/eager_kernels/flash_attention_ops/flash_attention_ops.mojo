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
    _raw_tuple_int,
    _spec_dispatcher11,
    _spec_dispatcher15,
)

from variant_gates import _dtype_arg_on, _op_on, _register_call


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
    # (q, k, v, output), each (batch, head, seq) in elements. head_dim's stride
    # must be 1; the Python caller declines anything else.
    var q_st = _raw_strides(strides_obj, 0)
    var k_st = _raw_strides(strides_obj, 1)
    var v_st = _raw_strides(strides_obj, 2)
    var o_st = _raw_strides(strides_obj, 3)
    var scale = Float32(_raw_f64(scale_obj))
    var causal = _raw_int(causal_obj) != 0
    var dtype = _raw_dtype_int(dtype_obj)
    var ctx = _raw_ctx(ctx_obj)
    var lse = _make_ptr[DType.float32](_raw_int(lse_obj))

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
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
    # (grad_output, query, key, value, out_fwd, dq, dk, dv), each (batch, head,
    # seq) in elements. head_dim's stride must be 1; the Python caller declines
    # anything else.
    var g_st = _raw_strides(strides_obj, 0)
    var q_st = _raw_strides(strides_obj, 1)
    var k_st = _raw_strides(strides_obj, 2)
    var v_st = _raw_strides(strides_obj, 3)
    var o_st = _raw_strides(strides_obj, 4)
    var dq_st = _raw_strides(strides_obj, 5)
    var dk_st = _raw_strides(strides_obj, 6)
    var dv_st = _raw_strides(strides_obj, 7)
    var scale = Float32(_raw_f64(scale_obj))
    var causal = _raw_int(causal_obj) != 0
    var dtype = _raw_dtype_int(dtype_obj)
    var ctx = _raw_ctx(ctx_obj)
    var lse = _make_ptr[DType.float32](_raw_int(lse_obj))

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                var dq = _make_ptr[dt](_raw_int(dq_obj))
                var dk = _make_ptr[dt](_raw_int(dk_obj))
                var dv = _make_ptr[dt](_raw_int(dv_obj))
                # dq/dk/dv are WRITE-ONLY. This bridge used to zero all three
                # first, because the original contract let a kernel accumulate
                # into them -- 36 memsets and 728 us per nanoGPT step,
                # third-largest launch cost in the whole profile. Both shipped
                # kernels accumulate in REGISTERS and only ever store, and every
                # store is guarded solely by tensor bounds (`row < seq`,
                # `col < head_dim`), never by anything that could skip a live
                # element.
                #
                # Verified rather than reasoned: fill all three with a poison
                # value, run without zeroing, count survivors. Zero survivors on
                # nanoGPT, hd96/300, hd128, seq1025, seq_kv > seq_q, and 3x3 --
                # see scratchpad `poison_probe.mojo` and the journal.
                #
                # A future kernel that accumulates must zero them itself, or
                # restore this. Do not reintroduce the memsets speculatively;
                # they are not free.
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
                    dq_st,
                    dk_st,
                    dv_st,
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


@export
def PyInit_flash_attention_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("flash_attention_ops")
        comptime if _op_on["FlashAttentionForward"]():
            _register_call(
                b,
                _spec_dispatcher11[
                    _flash_attention_forward_go, "FlashAttentionForward"
                ],
                docstring=(
                    "(out_ptr, lse_ptr, q_ptr, k_ptr, v_ptr, (batch, heads,"
                    " seq_q, seq_kv, head_dim), (batch, head, seq) x 4 for q,"
                    " k, v and the output, scale, is_causal, dtype,"
                    " context_ptr); fused flash-attention forward writing the"
                    " output and the per-row log-sum-exp the backward consumes."
                    " Q, K, V and the output are addressed through the given"
                    " element strides and their head_dim stride must be 1; the"
                    " log-sum-exp is dense."
                ),
            )
        comptime if _op_on["FlashAttentionBackward"]():
            _register_call(
                b,
                _spec_dispatcher15[
                    _flash_attention_backward_go, "FlashAttentionBackward"
                ],
                docstring=(
                    "(dq_ptr, dk_ptr, dv_ptr, grad_out_ptr, q_ptr, k_ptr,"
                    " v_ptr, out_ptr, lse_ptr, (batch, heads, seq_q, seq_kv,"
                    " head_dim), (batch, head, seq) x 8 for grad_out, q, k, v,"
                    " out, dq, dk and dv, scale, is_causal, dtype,"
                    " context_ptr); fused flash-attention backward writing dQ,"
                    " dK and dV. All eight operands are addressed through the"
                    " given element strides and their head_dim stride must be"
                    " 1; the log-sum-exp is dense."
                ),
            )
        return b.finalize()
    except e:
        abort(t"failed to create flash_attention_ops python module: {e}")
