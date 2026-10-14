"""Thin eager bridge for runtime-dynamic multi-tensor optimizer kernels.

The Python boundary validates the full mutable ATen contract and passes one
flat tuple of raw tensor metadata. This module packs a bounded descriptor array
per by-value launch and enqueues on the tensors' existing DeviceContext. It does
no allocation, host read, synchronization, or vendor-library call.
"""

from std.collections import Array
from std.math import ceildiv
from max.gpu.host import DeviceContext

from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    _raw_ctx,
    _raw_int,
    _raw_tuple_f64,
    _raw_tuple_int,
    _raw_tuple_len,
    _spec_dispatcher4,
    _spec_dispatcher6,
    _spec_dispatcher7,
)
from tmb.kernels.optimizer.contract import (
    FUSED_OPT_DESC_CAP,
    FusedOptDesc,
    empty_fused_opt_desc,
)
from tmb.kernels.optimizer.kernels import (
    FUSED_ADAGRAD,
    FUSED_ADAM,
    FUSED_ADAMW,
    FUSED_SGD,
    enqueue_fused_optimizer,
    fused_opt_label,
)
from tmb.kernels.optimizer.amp_kernels import enqueue_amp_update_scale
from tmb.kernels.optimizer.foreach_clip_contract import (
    FOREACH_CHUNK_ELEMENTS,
    FOREACH_DESC_CAP,
    ForeachDesc,
    empty_foreach_desc,
)
from tmb.kernels.optimizer.foreach_clip_kernels import (
    enqueue_foreach_l2_norm_f32,
)
from tmb.kernels.foreach.apply import foreach_chunk_elements

from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)


comptime _FUSED_OPT_RECORD_FIELDS = 7
comptime _FOREACH_NORM_RECORD_FIELDS = 3

# The dtype pairs a fused optimizer compiles for: (param/grad, state). The
# state dtype differs only in Adam/AdamW's mixed-precision mode.
comptime _FUSED_OPT_PARAM_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
]


def _fused_opt_launch[
    pdt: DType, sdt: DType, algo: Int
](
    metadata_obj: Arg,
    record_count: Int,
    chunk_elements: Int,
    lr_ptr: Int,
    grad_scale_ptr: Int,
    found_inf_ptr: Int,
    hyper: Array[Float64, 5],
    flags: Int,
    ctx: DeviceContext,
) raises:
    var record = 0
    while record < record_count:
        # The complete array is encoded by value, so initialize unused slots.
        var descs = Array[FusedOptDesc, FUSED_OPT_DESC_CAP](
            fill=empty_fused_opt_desc()
        )
        var desc_count = 0
        var total_chunks = 0
        while record < record_count and desc_count < FUSED_OPT_DESC_CAP:
            var base = record * _FUSED_OPT_RECORD_FIELDS
            var numel = _raw_tuple_int(metadata_obj, base + 6)
            record += 1
            if numel == 0:
                continue
            total_chunks += ceildiv(numel, chunk_elements)
            descs[desc_count] = FusedOptDesc(
                _raw_tuple_int(metadata_obj, base + 0),
                _raw_tuple_int(metadata_obj, base + 1),
                _raw_tuple_int(metadata_obj, base + 2),
                _raw_tuple_int(metadata_obj, base + 3),
                _raw_tuple_int(metadata_obj, base + 4),
                _raw_tuple_int(metadata_obj, base + 5),
                numel,
                total_chunks,
            )
            desc_count += 1
        enqueue_fused_optimizer[pdt, sdt, algo](
            descs,
            desc_count,
            total_chunks,
            chunk_elements,
            lr_ptr,
            grad_scale_ptr,
            found_inf_ptr,
            hyper,
            flags,
            ctx,
        )


def _fused_opt_go[
    algo: Int
](
    metadata_obj: Arg,
    scalars_obj: Arg,
    flags_obj: Arg,
    lr_ptr_obj: Arg,
    grad_scale_ptr_obj: Arg,
    found_inf_ptr_obj: Arg,
    device_context_ptr: Arg,
) raises:
    """One bridge for `aten::_fused_{adam,adamw,sgd,adagrad}_`.

    `metadata_obj` is 7 ints per tensor (param, grad, three state addresses,
    step address, numel), already validated against ATen's contract by the
    op; `scalars_obj` the five host hyperparameters (see
    `kernels._fused_opt_range`). The (param, state) dtypes are the
    DTYPE_ARG_0 / DTYPE_ARG_1 defines, so each pair is its own build.
    """
    comptime label = fused_opt_label[algo]()
    var value_count = _raw_tuple_len(metadata_obj)
    if value_count % _FUSED_OPT_RECORD_FIELDS != 0:
        raise Error("invalid fused ", label, " metadata field count")
    var record_count = value_count // _FUSED_OPT_RECORD_FIELDS
    for record in range(record_count):
        var base = record * _FUSED_OPT_RECORD_FIELDS
        var numel = _raw_tuple_int(metadata_obj, base + 6)
        if numel < 0:
            raise Error("fused ", label, " tensor numel must be nonnegative")
        # param, grad and (Adam, Adagrad) the step are always read; unused
        # state slots are legitimately zero.
        if numel > 0 and (
            _raw_tuple_int(metadata_obj, base) == 0
            or _raw_tuple_int(metadata_obj, base + 1) == 0
        ):
            raise Error("fused ", label, " nonempty pointer must be nonzero")
    if _raw_tuple_len(scalars_obj) != 5:
        raise Error("fused ", label, " expects five scalar hyperparameters")
    var hyper = Array[Float64, 5](fill=0.0)
    for i in range(5):
        hyper[i] = _raw_tuple_f64(scalars_obj, i)
    var flags = _raw_int(flags_obj)
    var lr_ptr = _raw_int(lr_ptr_obj)
    var grad_scale_ptr = _raw_int(grad_scale_ptr_obj)
    var found_inf_ptr = _raw_int(found_inf_ptr_obj)
    var ctx = _raw_ctx(device_context_ptr)

    var total_elements = 0
    for record in range(record_count):
        total_elements += _raw_tuple_int(
            metadata_obj, record * _FUSED_OPT_RECORD_FIELDS + 6
        )
    var chunk_elements = foreach_chunk_elements(total_elements, ctx)

    comptime for pdt in _FUSED_OPT_PARAM_DTYPES:
        comptime if _dtype_arg_on[0, pdt]():
            comptime if _dtype_arg_on[1, pdt]():
                _fused_opt_launch[pdt, pdt, algo](
                    metadata_obj,
                    record_count,
                    chunk_elements,
                    lr_ptr,
                    grad_scale_ptr,
                    found_inf_ptr,
                    hyper,
                    flags,
                    ctx,
                )
                return
            comptime if (
                (algo == FUSED_ADAM or algo == FUSED_ADAMW)
                and pdt == DType.float32
                and _dtype_arg_on[1, DType.bfloat16]()
            ):
                _fused_opt_launch[pdt, DType.bfloat16, algo](
                    metadata_obj,
                    record_count,
                    chunk_elements,
                    lr_ptr,
                    grad_scale_ptr,
                    found_inf_ptr,
                    hyper,
                    flags,
                    ctx,
                )
                return
    raise Error("fused ", label, ": unsupported (param, state) dtype pair")


def _foreach_l2_norm_go(
    metadata_obj: Arg,
    partials_ptr_obj: Arg,
    partials_numel_obj: Arg,
    device_context_ptr: Arg,
) raises:
    var value_count = _raw_tuple_len(metadata_obj)
    if value_count == 0:
        raise Error("foreach norm requires a nonempty TensorList")
    if value_count % _FOREACH_NORM_RECORD_FIELDS != 0:
        raise Error("invalid foreach norm metadata field count")
    var record_count = value_count // _FOREACH_NORM_RECORD_FIELDS
    var required_partials = 0
    for validation_record in range(record_count):
        var base = validation_record * _FOREACH_NORM_RECORD_FIELDS
        var input_ptr = _raw_tuple_int(metadata_obj, base + 0)
        var output_ptr = _raw_tuple_int(metadata_obj, base + 1)
        var numel = _raw_tuple_int(metadata_obj, base + 2)
        if numel < 0:
            raise Error("foreach norm tensor numel must be nonnegative")
        if numel > 0 and input_ptr == 0:
            raise Error("foreach norm nonempty input pointer must be nonzero")
        if output_ptr == 0:
            raise Error("foreach norm output pointer must be nonzero")
        if numel > 0:
            required_partials += ceildiv(numel, FOREACH_CHUNK_ELEMENTS)
    var partials_numel = _raw_int(partials_numel_obj)
    if partials_numel < required_partials:
        raise Error("foreach norm scratch is smaller than required partials")
    var partials_ptr = _raw_int(partials_ptr_obj)
    if required_partials > 0 and partials_ptr == 0:
        raise Error("foreach norm scratch pointer must be nonzero")

    var ctx = _raw_ctx(device_context_ptr)
    var partial_offset = 0
    var record = 0
    while record < record_count:
        var descs = Array[ForeachDesc, FOREACH_DESC_CAP](
            fill=empty_foreach_desc()
        )
        var desc_count = 0
        var total_chunks = 0
        while record < record_count and desc_count < FOREACH_DESC_CAP:
            var base = record * _FOREACH_NORM_RECORD_FIELDS
            var numel = _raw_tuple_int(metadata_obj, base + 2)
            if numel > 0:
                total_chunks += ceildiv(numel, FOREACH_CHUNK_ELEMENTS)
            descs[desc_count] = ForeachDesc(
                _raw_tuple_int(metadata_obj, base + 0),
                _raw_tuple_int(metadata_obj, base + 1),
                numel,
                total_chunks,
            )
            record += 1
            desc_count += 1

        enqueue_foreach_l2_norm_f32(
            descs,
            desc_count,
            total_chunks,
            partials_ptr + partial_offset * 4,
            ctx,
        )
        partial_offset += total_chunks


def _amp_update_scale_go(
    scale_ptr_obj: Arg,
    growth_tracker_ptr_obj: Arg,
    found_inf_ptr_obj: Arg,
    factors_obj: Arg,
    growth_interval_obj: Arg,
    device_context_ptr: Arg,
) raises:
    if _raw_tuple_len(factors_obj) != 2:
        raise Error("amp update scale expects (growth, backoff) factors")
    enqueue_amp_update_scale(
        _raw_int(scale_ptr_obj),
        _raw_int(growth_tracker_ptr_obj),
        _raw_int(found_inf_ptr_obj),
        _raw_tuple_f64(factors_obj, 0),
        _raw_tuple_f64(factors_obj, 1),
        _raw_int(growth_interval_obj),
        _raw_ctx(device_context_ptr),
    )


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["FusedAdam"]():
            _spec_dispatcher7[_fused_opt_go[FUSED_ADAM], "FusedAdam"](
                argv, argc
            )
            return 0
        comptime if _op_on["FusedAdamW"]():
            _spec_dispatcher7[_fused_opt_go[FUSED_ADAMW], "FusedAdamW"](
                argv, argc
            )
            return 0
        comptime if _op_on["FusedSgd"]():
            _spec_dispatcher7[_fused_opt_go[FUSED_SGD], "FusedSgd"](argv, argc)
            return 0
        comptime if _op_on["FusedAdagrad"]():
            _spec_dispatcher7[_fused_opt_go[FUSED_ADAGRAD], "FusedAdagrad"](
                argv, argc
            )
            return 0
        comptime if _op_on["ForeachL2Norm"]():
            _spec_dispatcher4[_foreach_l2_norm_go, "ForeachL2Norm"](argv, argc)
            return 0
        comptime if _op_on["AmpUpdateScale"]():
            _spec_dispatcher6[_amp_update_scale_go, "AmpUpdateScale"](
                argv, argc
            )
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
