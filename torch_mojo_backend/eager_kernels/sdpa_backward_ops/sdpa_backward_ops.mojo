# ===----------------------------------------------------------------------=== #
# Thin eager-mode bridge for fused SDPA dropout/softmax backward.
#
# Device-kernel bodies live in the Fable-owned internal module imported below.
# This Python-visible module only validates and unpacks the pointer ABI, builds
# the optional mask pointer, and enqueues on the caller's DeviceContext.  It
# performs no allocation, host read, or synchronization.
# ===----------------------------------------------------------------------=== #

from std.os import abort

from sdpa_backward_gemm_kernels import enqueue_sdpa_ta_gemm_f32
from sdpa_dropout_softmax_backward_kernels import (
    enqueue_sdpa_dropout_softmax_backward,
    enqueue_sdpa_dropout_softmax_backward_f32,
)
from op_utils import (
    Arg,
    Argv,
    FLOAT_DTYPES,
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _raw_tuple_int,
    _spec_dispatcher12,
    _spec_dispatcher13,
    _spec_dispatcher7,
)

from variant_gates import ErrBuf, NO_OP_COMPILED, _op_on, _tmb_entry_error


def _sdpa_dropout_softmax_backward_go(
    output_ptr_obj: Arg,
    probabilities_ptr_obj: Arg,
    grad_after_dropout_ptr_obj: Arg,
    mask_ptr_obj: Arg,
    rows_obj: Arg,
    cols_obj: Arg,
    q_len_obj: Arg,
    has_mask_obj: Arg,
    causal_obj: Arg,
    dropout_scale_obj: Arg,
    score_scale_obj: Arg,
    dtype_obj: Arg,
    device_context_ptr: Arg,
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
            var mask: Optional[Pointer[Scalar[DType.bool], MutAnyOrigin]] = None
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


def _sdpa_dsb_f32_go(
    output_ptr_obj: Arg,
    probabilities_ptr_obj: Arg,
    grad_after_dropout_ptr_obj: Arg,
    mask_ptr_obj: Arg,
    rows_obj: Arg,
    cols_obj: Arg,
    has_mask_obj: Arg,
    dropout_scale_obj: Arg,
    score_scale_obj: Arg,
    causal_obj: Arg,
    q_len_obj: Arg,
    device_context_ptr: Arg,
) raises:
    var output = _make_ptr[DType.float32](
        _raw_int(output_ptr_obj)
    ).as_unsafe_any_origin()
    var probabilities = _make_ptr[DType.float32](
        _raw_int(probabilities_ptr_obj)
    ).as_unsafe_any_origin()
    var grad_after_dropout = _make_ptr[DType.float32](
        _raw_int(grad_after_dropout_ptr_obj)
    ).as_unsafe_any_origin()
    var mask_address = _raw_int(mask_ptr_obj)
    var has_mask = _raw_int(has_mask_obj) != 0
    var mask: Optional[Pointer[Scalar[DType.bool], MutAnyOrigin]] = None
    if has_mask:
        if mask_address == 0:
            raise Error(
                "SDPADropoutSoftmaxBackwardF32 requires a non-null mask when"
                " has_mask is true"
            )
        mask = _make_ptr[DType.bool](mask_address).as_unsafe_any_origin()
    elif mask_address != 0:
        raise Error(
            "SDPADropoutSoftmaxBackwardF32 requires a null mask when"
            " has_mask is false"
        )

    var ctx = _raw_ctx(device_context_ptr)
    enqueue_sdpa_dropout_softmax_backward_f32(
        output,
        probabilities,
        grad_after_dropout,
        mask,
        _raw_int(rows_obj),
        _raw_int(cols_obj),
        has_mask,
        _raw_f64(dropout_scale_obj),
        _raw_f64(score_scale_obj),
        ctx,
        causal=_raw_int(causal_obj) != 0,
        q_len=_raw_int(q_len_obj),
    )


def _sdpa_ta_gemm_go(
    c_ptr_obj: Arg,
    a_ptr_obj: Arg,
    b_ptr_obj: Arg,
    mask_ptr_obj: Arg,
    # (batch, m, n, k, has_mask, causal)
    params: Arg,
    drop_scale_obj: Arg,
    device_context_ptr: Arg,
) raises:
    var c = _make_ptr[DType.float32](_raw_int(c_ptr_obj)).as_unsafe_any_origin()
    var a = (
        _make_ptr[DType.float32](_raw_int(a_ptr_obj))
        .as_unsafe_any_origin()
        .as_imm()
    )
    var b = (
        _make_ptr[DType.float32](_raw_int(b_ptr_obj))
        .as_unsafe_any_origin()
        .as_imm()
    )
    var mask_address = _raw_int(mask_ptr_obj)
    var has_mask = _raw_tuple_int(params, 4) != 0
    var mask: Optional[Pointer[Scalar[DType.bool], ImmutAnyOrigin]] = None
    if has_mask:
        if mask_address == 0:
            raise Error(
                "SDPATransAGemmF32 requires a non-null mask when has_mask"
                " is true"
            )
        mask = (
            _make_ptr[DType.bool](mask_address).as_unsafe_any_origin().as_imm()
        )
    elif mask_address != 0:
        raise Error(
            "SDPATransAGemmF32 requires a null mask when has_mask is false"
        )

    var ctx = _raw_ctx(device_context_ptr)
    enqueue_sdpa_ta_gemm_f32(
        c,
        a,
        b,
        mask,
        _raw_tuple_int(params, 0),
        _raw_tuple_int(params, 1),
        _raw_tuple_int(params, 2),
        _raw_tuple_int(params, 3),
        _raw_tuple_int(params, 5) != 0,
        _raw_f64(drop_scale_obj),
        ctx,
    )


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["SDPADropoutSoftmaxBackward"]():
            _spec_dispatcher13[
                _sdpa_dropout_softmax_backward_go,
                "SDPADropoutSoftmaxBackward",
            ](argv, argc)
            return 0
        comptime if _op_on["SDPADropoutSoftmaxBackwardF32"]():
            _spec_dispatcher12[
                _sdpa_dsb_f32_go, "SDPADropoutSoftmaxBackwardF32"
            ](argv, argc)
            return 0
        comptime if _op_on["SDPATransAGemmF32"]():
            _spec_dispatcher7[_sdpa_ta_gemm_go, "SDPATransAGemmF32"](argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
