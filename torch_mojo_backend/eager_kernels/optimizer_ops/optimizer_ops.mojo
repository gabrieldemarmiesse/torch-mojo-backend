"""Thin eager bridge for runtime-dynamic multi-tensor optimizer kernels.

The Python boundary validates the full mutable ATen contract and passes one
flat tuple of raw tensor metadata. This module packs a bounded descriptor array
per by-value launch and enqueues on the tensors' existing DeviceContext. It does
no allocation, host read, synchronization, or vendor-library call.
"""

from std.collections import InlineArray
from std.math import ceildiv
from std.os import abort

from op_utils import (
    Arg,
    Argv,
    FLOAT_DTYPES,
    _raw_ctx,
    _raw_dtype_int,
    _raw_int,
    _raw_tuple_f64,
    _raw_tuple_int,
    _raw_tuple_len,
    _spec_dispatcher3,
    _spec_dispatcher4,
    _spec_dispatcher5,
    _spec_dispatcher8,
)
from optimizer_contract import (
    ADAMW_CHUNK_ELEMENTS,
    ADAMW_DESC_CAP,
    AdamWDesc,
    empty_adamw_desc,
)
from optimizer_kernels import enqueue_fused_adamw_f32
from foreach_clip_contract import (
    FOREACH_CHUNK_ELEMENTS,
    FOREACH_DESC_CAP,
    ForeachDesc,
    empty_foreach_desc,
)
from foreach_clip_kernels import enqueue_foreach_l2_norm_f32
from foreach_batched_kernels import (
    FEW_ADD,
    FEW_ADDCDIV,
    FEW_ADDCMUL,
    FEW_DESC_CAP,
    FEW_DIV,
    FEW_LERP,
    FEW_MUL,
    FEW_MUL_TENSOR,
    FEW_SQRT,
    ForeachEwDesc,
    _few_fields,
    _few_has_scalar,
    _few_label,
    empty_foreach_ew_desc,
    foreach_ew_chunk_elements,
    foreach_ew_enqueue,
)
from foreach_elementwise_kernels import (
    FOREACH_EW_SLOTS,
    enqueue_foreach_gather_scalars_f32,
)

from variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _dtype_supported,
    _op_on,
    _tmb_entry_error,
)


comptime _ADAMW_RECORD_FIELDS = 7
comptime _FOREACH_NORM_RECORD_FIELDS = 3


def _fused_adamw_go(
    metadata_obj: Arg,
    scalars_obj: Arg,
    dtype_mode_obj: Arg,
    flags_obj: Arg,
    lr_ptr_obj: Arg,
    grad_scale_ptr_obj: Arg,
    found_inf_ptr_obj: Arg,
    device_context_ptr: Arg,
) raises:
    var value_count = _raw_tuple_len(metadata_obj)
    if value_count % _ADAMW_RECORD_FIELDS != 0:
        raise Error("invalid fused AdamW metadata field count")
    if _raw_tuple_len(scalars_obj) != 5:
        raise Error("fused AdamW expects five scalar hyperparameters")
    if _raw_int(dtype_mode_obj) != 0:
        raise Error("fused AdamW currently supports homogeneous float32 state")
    var flags = _raw_int(flags_obj)
    if flags < 0 or flags > 3:
        raise Error("invalid fused AdamW flags")

    var ctx = _raw_ctx(device_context_ptr)
    if ctx.api() == "cpu":
        raise Error("fused AdamW requires a Mojo accelerator device")
    var lr_scalar = Float32(_raw_tuple_f64(scalars_obj, 0))
    var beta1 = Float32(_raw_tuple_f64(scalars_obj, 1))
    var beta2 = Float32(_raw_tuple_f64(scalars_obj, 2))
    var weight_decay = Float32(_raw_tuple_f64(scalars_obj, 3))
    var eps = Float32(_raw_tuple_f64(scalars_obj, 4))
    var amsgrad = flags & 1
    var maximize = (flags >> 1) & 1
    var lr_ptr = _raw_int(lr_ptr_obj)
    var grad_scale_ptr = _raw_int(grad_scale_ptr_obj)
    var found_inf_ptr = _raw_int(found_inf_ptr_obj)

    var record = 0
    var record_count = value_count // _ADAMW_RECORD_FIELDS
    while record < record_count:
        # The complete array is encoded by value, so initialize unused slots.
        var descs = InlineArray[AdamWDesc, ADAMW_DESC_CAP](
            fill=empty_adamw_desc()
        )
        var desc_count = 0
        var total_chunks = 0
        while record < record_count and desc_count < ADAMW_DESC_CAP:
            var base = record * _ADAMW_RECORD_FIELDS
            var numel = _raw_tuple_int(metadata_obj, base + 6)
            record += 1
            if numel < 0:
                raise Error("fused AdamW tensor numel must be nonnegative")
            if numel == 0:
                continue
            total_chunks += ceildiv(numel, ADAMW_CHUNK_ELEMENTS)
            descs[desc_count] = AdamWDesc(
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

        if desc_count > 0:
            enqueue_fused_adamw_f32(
                descs,
                desc_count,
                total_chunks,
                lr_scalar,
                lr_ptr,
                beta1,
                beta2,
                weight_decay,
                eps,
                amsgrad,
                maximize,
                grad_scale_ptr,
                found_inf_ptr,
                ctx,
            )


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
    if ctx.api() == "cpu":
        raise Error("foreach norm fast path requires a Mojo accelerator device")
    var partial_offset = 0
    var record = 0
    while record < record_count:
        var descs = InlineArray[ForeachDesc, FOREACH_DESC_CAP](
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


def _foreach_ew_validate(
    metadata_obj: Arg, record_fields: Int, op_name: StaticString
) raises -> Int:
    """Shared record-count/pointer validation for the batched foreach ops.

    Every record's trailing field is the numel; all leading fields are
    device pointers that must be nonzero when the numel is positive.
    """
    var value_count = _raw_tuple_len(metadata_obj)
    if value_count == 0:
        raise Error(op_name, " requires a nonempty TensorList")
    if value_count % record_fields != 0:
        raise Error("invalid ", op_name, " metadata field count")
    var record_count = value_count // record_fields
    for record in range(record_count):
        var base = record * record_fields
        var numel = _raw_tuple_int(metadata_obj, base + record_fields - 1)
        if numel < 0:
            raise Error(op_name, " tensor numel must be nonnegative")
        if numel > 0:
            for field in range(record_fields - 1):
                if _raw_tuple_int(metadata_obj, base + field) == 0:
                    raise Error(op_name, " nonempty pointer must be nonzero")
    return record_count


def _foreach_ew_go[
    op: Int
](
    metadata_obj: Arg,
    scalars_obj: Arg,
    aux_obj: Arg,
    dtype_obj: Arg,
    device_context_ptr: Arg,
) raises:
    """One bridge for the whole `aten::_foreach_*` elementwise family.

    The Python boundary has already validated ATen's mutable-TensorList
    contract and flattened the list into `metadata_obj`: `_few_fields[op]()`
    ints per tensor, the operand addresses then the numel. What else an op
    needs travels in the two small tuples, and which of them it reads is a
    compile-time fact:

    * `scalars_obj` -- one FP32 scalar per tensor for mul/add/div/addc
      (`.Scalar` fills the same tuple with one repeated value that
      `.ScalarList` fills per tensor), or the two lerp weights.
    * `aux_obj` -- ints: the lerp branch selector, or the address of the 0-d
      device scalar `_foreach_mul_.Tensor` multiplies by.

    Descriptors are packed `FEW_DESC_CAP` at a time; a longer list becomes
    several launches of the same compiled kernel. Nothing is allocated, read
    back, or synchronized here.
    """
    comptime fields = _few_fields[op]()
    comptime label = _few_label[op]()
    var record_count = _foreach_ew_validate(metadata_obj, fields, label)

    var dtype = _raw_dtype_int(dtype_obj)
    if not _dtype_supported[List[DType](FLOAT_DTYPES)](dtype):
        raise Error("mojo foreach ", label, ": unsupported dtype ", dtype)

    var weight = Float32(0.0)
    var one_minus_weight = Float32(0.0)
    comptime if _few_has_scalar[op]():
        if _raw_tuple_len(scalars_obj) != record_count:
            raise Error("mojo foreach ", label, " needs one scalar per tensor")
    comptime if op == FEW_LERP:
        if _raw_tuple_len(scalars_obj) != 2:
            raise Error("mojo foreach lerp needs both narrowed weights")
        weight = Float32(_raw_tuple_f64(scalars_obj, 0))
        one_minus_weight = Float32(_raw_tuple_f64(scalars_obj, 1))

    var low_branch = 0
    var scalar_addr = 0
    comptime if op == FEW_LERP:
        if _raw_tuple_len(aux_obj) != 1:
            raise Error("mojo foreach lerp needs its branch selector")
        low_branch = _raw_tuple_int(aux_obj, 0)
    comptime if op == FEW_MUL_TENSOR:
        if _raw_tuple_len(aux_obj) != 1:
            raise Error("mojo foreach multiply needs its scalar address")
        scalar_addr = _raw_tuple_int(aux_obj, 0)
        if scalar_addr == 0:
            raise Error("mojo foreach multiply scalar pointer must be nonzero")

    var ctx = _raw_ctx(device_context_ptr)
    if ctx.api() == "cpu":
        raise Error(
            "mojo foreach ", label, " requires a Mojo accelerator device"
        )

    # The chunk size is what fills the device, so it is derived from the whole
    # list's element count once, before any descriptor batch is packed.
    var total_elements = 0
    for record in range(record_count):
        total_elements += _raw_tuple_int(
            metadata_obj, record * fields + fields - 1
        )
    var chunk_elements = foreach_ew_chunk_elements(total_elements, ctx)

    var record = 0
    while record < record_count:
        var descs = InlineArray[ForeachEwDesc, FEW_DESC_CAP](
            fill=empty_foreach_ew_desc()
        )
        var scalars = InlineArray[Float32, FEW_DESC_CAP](fill=0.0)
        var desc_count = 0
        var total_chunks = 0
        while record < record_count and desc_count < FEW_DESC_CAP:
            var base = record * fields
            var numel = _raw_tuple_int(metadata_obj, base + fields - 1)
            if numel > 0:
                total_chunks += ceildiv(numel, chunk_elements)
            var addr1 = 0
            var addr2 = 0
            comptime if fields >= 3:
                addr1 = _raw_tuple_int(metadata_obj, base + 1)
            comptime if fields >= 4:
                addr2 = _raw_tuple_int(metadata_obj, base + 2)
            descs[desc_count] = ForeachEwDesc(
                _raw_tuple_int(metadata_obj, base),
                addr1,
                addr2,
                numel,
                total_chunks,
            )
            comptime if _few_has_scalar[op]():
                scalars[desc_count] = Float32(
                    _raw_tuple_f64(scalars_obj, record)
                )
            record += 1
            desc_count += 1

        comptime for dt in FLOAT_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if dtype == dt:
                    foreach_ew_enqueue[dt, op](
                        descs,
                        scalars,
                        desc_count,
                        total_chunks,
                        chunk_elements,
                        scalar_addr,
                        weight,
                        one_minus_weight,
                        low_branch,
                        ctx,
                    )


def _foreach_gather_scalars_go(
    metadata_obj: Arg,
    out_ptr_obj: Arg,
    device_context_ptr: Arg,
) raises:
    var record_count = _raw_tuple_len(metadata_obj)
    if record_count == 0:
        raise Error("foreach gather requires a nonempty TensorList")
    var out_addr = _raw_int(out_ptr_obj)
    if out_addr == 0:
        raise Error("foreach gather output pointer must be nonzero")
    for record in range(record_count):
        if _raw_tuple_int(metadata_obj, record) == 0:
            raise Error("foreach gather input pointers must be nonzero")
    var ctx = _raw_ctx(device_context_ptr)
    if ctx.api() == "cpu":
        raise Error("foreach gather requires a Mojo accelerator device")

    var record = 0
    while record < record_count:
        var in_addrs = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
        var base = record
        var slot = 0
        while record < record_count and slot < FOREACH_EW_SLOTS:
            in_addrs[slot] = _raw_tuple_int(metadata_obj, record)
            record += 1
            slot += 1
        var count = slot
        while slot < FOREACH_EW_SLOTS:
            in_addrs[slot] = in_addrs[0]
            slot += 1
        enqueue_foreach_gather_scalars_f32(out_addr, in_addrs, base, count, ctx)


comptime _FOREACH_EW_DOC = (
    "(metadata, scalars, aux, dtype, context_ptr); one batched in-place"
    " launch for one member of the foreach elementwise family"
)


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["FusedAdamW"]():
            _spec_dispatcher8[_fused_adamw_go, "FusedAdamW"](argv, argc)
            return 0
        comptime if _op_on["ForeachL2Norm"]():
            _spec_dispatcher4[_foreach_l2_norm_go, "ForeachL2Norm"](argv, argc)
            return 0
        comptime if _op_on["ForeachGatherScalars"]():
            _spec_dispatcher3[
                _foreach_gather_scalars_go, "ForeachGatherScalars"
            ](argv, argc)
            return 0
        comptime if _op_on["ForeachMul"]():
            _spec_dispatcher5[_foreach_ew_go[FEW_MUL], "ForeachMul"](argv, argc)
            return 0
        comptime if _op_on["ForeachAdd"]():
            _spec_dispatcher5[_foreach_ew_go[FEW_ADD], "ForeachAdd"](argv, argc)
            return 0
        comptime if _op_on["ForeachDiv"]():
            _spec_dispatcher5[_foreach_ew_go[FEW_DIV], "ForeachDiv"](argv, argc)
            return 0
        comptime if _op_on["ForeachMulTensor"]():
            _spec_dispatcher5[
                _foreach_ew_go[FEW_MUL_TENSOR], "ForeachMulTensor"
            ](argv, argc)
            return 0
        comptime if _op_on["ForeachLerp"]():
            _spec_dispatcher5[_foreach_ew_go[FEW_LERP], "ForeachLerp"](
                argv, argc
            )
            return 0
        comptime if _op_on["ForeachAddcmul"]():
            _spec_dispatcher5[_foreach_ew_go[FEW_ADDCMUL], "ForeachAddcmul"](
                argv, argc
            )
            return 0
        comptime if _op_on["ForeachAddcdiv"]():
            _spec_dispatcher5[_foreach_ew_go[FEW_ADDCDIV], "ForeachAddcdiv"](
                argv, argc
            )
            return 0
        comptime if _op_on["ForeachSqrt"]():
            _spec_dispatcher5[_foreach_ew_go[FEW_SQRT], "ForeachSqrt"](
                argv, argc
            )
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
