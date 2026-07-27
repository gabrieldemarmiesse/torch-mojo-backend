"""Thin eager bridge for runtime-dynamic multi-tensor optimizer kernels.

The Python boundary validates the full mutable ATen contract and passes one
flat tuple of raw tensor metadata. This module packs a bounded descriptor array
per by-value launch and enqueues on the tensors' existing DeviceContext. It does
no allocation, host read, synchronization, or vendor-library call.
"""

from std.collections import InlineArray
from std.math import ceildiv
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.python._cpython import PyObjectPtr, Py_ssize_t

from op_utils import (
    _raw_ctx,
    _raw_f64,
    _raw_int,
    _raw_ret_none,
    _raw_tuple_f64,
    _raw_tuple_int,
    _raw_tuple_len,
    _spec_unsupported,
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
from foreach_clip_kernels import (
    enqueue_foreach_l2_norm_f32,
    enqueue_foreach_mul_tensor_f32,
)
from foreach_elementwise_kernels import (
    FEA_ADDCDIV,
    FEA_ADDCMUL,
    FES_ADD,
    FES_DIV,
    FES_MUL,
    FOREACH_EW_CHUNK,
    FOREACH_EW_SLOTS,
    enqueue_foreach_addc_f32,
    enqueue_foreach_gather_scalars_f32,
    enqueue_foreach_lerp_f32,
    enqueue_foreach_scalar_f32,
    enqueue_foreach_sqrt_f32,
)


comptime _ADAMW_RECORD_FIELDS = 7
comptime _FOREACH_NORM_RECORD_FIELDS = 3
comptime _FOREACH_MUL_RECORD_FIELDS = 2
comptime _FOREACH_EW_SCALAR_RECORD_FIELDS = 2  # (ptr, numel)
comptime _FOREACH_EW_LERP_RECORD_FIELDS = 3  # (self_ptr, end_ptr, numel)
comptime _FOREACH_EW_ADDC_RECORD_FIELDS = 4  # (self, t1, t2, numel)
comptime _FOREACH_EW_SQRT_RECORD_FIELDS = 3  # (in_ptr, out_ptr, numel)


def _fused_adamw_go(
    metadata_obj: PyObjectPtr,
    scalars_obj: PyObjectPtr,
    dtype_mode_obj: PyObjectPtr,
    flags_obj: PyObjectPtr,
    lr_ptr_obj: PyObjectPtr,
    grad_scale_ptr_obj: PyObjectPtr,
    found_inf_ptr_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
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


def _fused_adamw_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 8:
            raise Error("FusedAdamW expects exactly eight arguments")
        _fused_adamw_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
        )
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def _foreach_l2_norm_go(
    metadata_obj: PyObjectPtr,
    partials_ptr_obj: PyObjectPtr,
    partials_numel_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
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


def _foreach_mul_tensor_go(
    metadata_obj: PyObjectPtr,
    scalar_ptr_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var value_count = _raw_tuple_len(metadata_obj)
    if value_count == 0:
        raise Error("foreach multiply requires a nonempty TensorList")
    if value_count % _FOREACH_MUL_RECORD_FIELDS != 0:
        raise Error("invalid foreach multiply metadata field count")
    var record_count = value_count // _FOREACH_MUL_RECORD_FIELDS
    for validation_record in range(record_count):
        var base = validation_record * _FOREACH_MUL_RECORD_FIELDS
        var tensor_ptr = _raw_tuple_int(metadata_obj, base + 0)
        var numel = _raw_tuple_int(metadata_obj, base + 1)
        if numel < 0:
            raise Error("foreach multiply tensor numel must be nonnegative")
        if numel > 0 and tensor_ptr == 0:
            raise Error("foreach multiply nonempty pointer must be nonzero")

    var ctx = _raw_ctx(device_context_ptr)
    if ctx.api() == "cpu":
        raise Error(
            "foreach multiply fast path requires a Mojo accelerator device"
        )
    var scalar_ptr = _raw_int(scalar_ptr_obj)
    if scalar_ptr == 0:
        raise Error("foreach multiply scalar pointer must be nonzero")

    var record = 0
    while record < record_count:
        var descs = InlineArray[ForeachDesc, FOREACH_DESC_CAP](
            fill=empty_foreach_desc()
        )
        var desc_count = 0
        var total_chunks = 0
        while record < record_count and desc_count < FOREACH_DESC_CAP:
            var base = record * _FOREACH_MUL_RECORD_FIELDS
            var numel = _raw_tuple_int(metadata_obj, base + 1)
            if numel > 0:
                total_chunks += ceildiv(numel, FOREACH_CHUNK_ELEMENTS)
            descs[desc_count] = ForeachDesc(
                _raw_tuple_int(metadata_obj, base + 0),
                0,
                numel,
                total_chunks,
            )
            record += 1
            desc_count += 1

        enqueue_foreach_mul_tensor_f32(
            descs, desc_count, total_chunks, scalar_ptr, ctx
        )


def _foreach_ew_validate(
    metadata_obj: PyObjectPtr, record_fields: Int, op_name: StaticString
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


@always_inline
def _foreach_ew_backfill(
    mut addrs: InlineArray[Int, FOREACH_EW_SLOTS], slot_count: Int
):
    """Give empty/padding slots a valid dummy address (never dereferenced:
    those slots own zero chunks). Metal still requires every pointer-typed
    argument to translate to a real buffer."""
    var first_addr = 0
    for slot in range(slot_count):
        if addrs[slot] != 0:
            first_addr = addrs[slot]
            break
    for slot in range(FOREACH_EW_SLOTS):
        if addrs[slot] == 0:
            addrs[slot] = first_addr


def _foreach_scalar_op_go(
    op_code_obj: PyObjectPtr,
    metadata_obj: PyObjectPtr,
    scalars_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var op_code = _raw_int(op_code_obj)
    if op_code < 0 or op_code > 2:
        raise Error("invalid foreach scalar op code")
    var record_count = _foreach_ew_validate(
        metadata_obj, _FOREACH_EW_SCALAR_RECORD_FIELDS, "foreach scalar op"
    )
    if _raw_tuple_len(scalars_obj) != record_count:
        raise Error("foreach scalar op needs one scalar per tensor")
    var ctx = _raw_ctx(device_context_ptr)
    if ctx.api() == "cpu":
        raise Error("foreach scalar op requires a Mojo accelerator device")

    var record = 0
    while record < record_count:
        var addrs = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
        var chunk_ends = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
        var numels = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
        var scalars = InlineArray[Float32, FOREACH_EW_SLOTS](fill=0.0)
        var slot = 0
        var total_chunks = 0
        while record < record_count and slot < FOREACH_EW_SLOTS:
            var base = record * _FOREACH_EW_SCALAR_RECORD_FIELDS
            var numel = _raw_tuple_int(metadata_obj, base + 1)
            if numel > 0:
                total_chunks += ceildiv(numel, FOREACH_EW_CHUNK)
            addrs[slot] = _raw_tuple_int(metadata_obj, base + 0)
            numels[slot] = numel
            chunk_ends[slot] = total_chunks
            scalars[slot] = Float32(_raw_tuple_f64(scalars_obj, record))
            record += 1
            slot += 1
        var used_slots = slot
        while slot < FOREACH_EW_SLOTS:
            chunk_ends[slot] = total_chunks
            slot += 1
        if total_chunks == 0:
            continue
        _foreach_ew_backfill(addrs, used_slots)
        if op_code == FES_MUL:
            enqueue_foreach_scalar_f32[FES_MUL](
                addrs, chunk_ends, numels, scalars, total_chunks, ctx
            )
        elif op_code == FES_ADD:
            enqueue_foreach_scalar_f32[FES_ADD](
                addrs, chunk_ends, numels, scalars, total_chunks, ctx
            )
        else:
            enqueue_foreach_scalar_f32[FES_DIV](
                addrs, chunk_ends, numels, scalars, total_chunks, ctx
            )


def _foreach_lerp_scalar_go(
    metadata_obj: PyObjectPtr,
    weight_obj: PyObjectPtr,
    one_minus_weight_obj: PyObjectPtr,
    low_branch_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var record_count = _foreach_ew_validate(
        metadata_obj, _FOREACH_EW_LERP_RECORD_FIELDS, "foreach lerp"
    )
    var weight = Float32(_raw_f64(weight_obj))
    var one_minus_weight = Float32(_raw_f64(one_minus_weight_obj))
    var low_branch = _raw_int(low_branch_obj)
    var ctx = _raw_ctx(device_context_ptr)
    if ctx.api() == "cpu":
        raise Error("foreach lerp requires a Mojo accelerator device")

    var record = 0
    while record < record_count:
        var self_addrs = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
        var end_addrs = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
        var chunk_ends = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
        var numels = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
        var slot = 0
        var total_chunks = 0
        while record < record_count and slot < FOREACH_EW_SLOTS:
            var base = record * _FOREACH_EW_LERP_RECORD_FIELDS
            var numel = _raw_tuple_int(metadata_obj, base + 2)
            if numel > 0:
                total_chunks += ceildiv(numel, FOREACH_EW_CHUNK)
            self_addrs[slot] = _raw_tuple_int(metadata_obj, base + 0)
            end_addrs[slot] = _raw_tuple_int(metadata_obj, base + 1)
            numels[slot] = numel
            chunk_ends[slot] = total_chunks
            record += 1
            slot += 1
        var used_slots = slot
        while slot < FOREACH_EW_SLOTS:
            chunk_ends[slot] = total_chunks
            slot += 1
        if total_chunks == 0:
            continue
        _foreach_ew_backfill(self_addrs, used_slots)
        _foreach_ew_backfill(end_addrs, used_slots)
        enqueue_foreach_lerp_f32(
            self_addrs,
            end_addrs,
            chunk_ends,
            numels,
            weight,
            one_minus_weight,
            low_branch,
            total_chunks,
            ctx,
        )


def _foreach_addc_op_go(
    op_code_obj: PyObjectPtr,
    metadata_obj: PyObjectPtr,
    scalars_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var op_code = _raw_int(op_code_obj)
    if op_code < 0 or op_code > 1:
        raise Error("invalid foreach addc op code")
    var record_count = _foreach_ew_validate(
        metadata_obj, _FOREACH_EW_ADDC_RECORD_FIELDS, "foreach addc op"
    )
    if _raw_tuple_len(scalars_obj) != record_count:
        raise Error("foreach addc op needs one scalar per tensor")
    var ctx = _raw_ctx(device_context_ptr)
    if ctx.api() == "cpu":
        raise Error("foreach addc op requires a Mojo accelerator device")

    var record = 0
    while record < record_count:
        var self_addrs = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
        var first_addrs = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
        var second_addrs = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
        var chunk_ends = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
        var numels = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
        var scalars = InlineArray[Float32, FOREACH_EW_SLOTS](fill=0.0)
        var slot = 0
        var total_chunks = 0
        while record < record_count and slot < FOREACH_EW_SLOTS:
            var base = record * _FOREACH_EW_ADDC_RECORD_FIELDS
            var numel = _raw_tuple_int(metadata_obj, base + 3)
            if numel > 0:
                total_chunks += ceildiv(numel, FOREACH_EW_CHUNK)
            self_addrs[slot] = _raw_tuple_int(metadata_obj, base + 0)
            first_addrs[slot] = _raw_tuple_int(metadata_obj, base + 1)
            second_addrs[slot] = _raw_tuple_int(metadata_obj, base + 2)
            numels[slot] = numel
            chunk_ends[slot] = total_chunks
            scalars[slot] = Float32(_raw_tuple_f64(scalars_obj, record))
            record += 1
            slot += 1
        var used_slots = slot
        while slot < FOREACH_EW_SLOTS:
            chunk_ends[slot] = total_chunks
            slot += 1
        if total_chunks == 0:
            continue
        _foreach_ew_backfill(self_addrs, used_slots)
        _foreach_ew_backfill(first_addrs, used_slots)
        _foreach_ew_backfill(second_addrs, used_slots)
        if op_code == FEA_ADDCMUL:
            enqueue_foreach_addc_f32[FEA_ADDCMUL](
                self_addrs,
                first_addrs,
                second_addrs,
                chunk_ends,
                numels,
                scalars,
                total_chunks,
                ctx,
            )
        else:
            enqueue_foreach_addc_f32[FEA_ADDCDIV](
                self_addrs,
                first_addrs,
                second_addrs,
                chunk_ends,
                numels,
                scalars,
                total_chunks,
                ctx,
            )


def _foreach_sqrt_go(
    metadata_obj: PyObjectPtr, device_context_ptr: PyObjectPtr
) raises:
    var record_count = _foreach_ew_validate(
        metadata_obj, _FOREACH_EW_SQRT_RECORD_FIELDS, "foreach sqrt"
    )
    var ctx = _raw_ctx(device_context_ptr)
    if ctx.api() == "cpu":
        raise Error("foreach sqrt requires a Mojo accelerator device")

    var record = 0
    while record < record_count:
        var in_addrs = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
        var out_addrs = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
        var chunk_ends = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
        var numels = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
        var slot = 0
        var total_chunks = 0
        while record < record_count and slot < FOREACH_EW_SLOTS:
            var base = record * _FOREACH_EW_SQRT_RECORD_FIELDS
            var numel = _raw_tuple_int(metadata_obj, base + 2)
            if numel > 0:
                total_chunks += ceildiv(numel, FOREACH_EW_CHUNK)
            in_addrs[slot] = _raw_tuple_int(metadata_obj, base + 0)
            out_addrs[slot] = _raw_tuple_int(metadata_obj, base + 1)
            numels[slot] = numel
            chunk_ends[slot] = total_chunks
            record += 1
            slot += 1
        var used_slots = slot
        while slot < FOREACH_EW_SLOTS:
            chunk_ends[slot] = total_chunks
            slot += 1
        if total_chunks == 0:
            continue
        _foreach_ew_backfill(in_addrs, used_slots)
        _foreach_ew_backfill(out_addrs, used_slots)
        enqueue_foreach_sqrt_f32(
            in_addrs, out_addrs, chunk_ends, numels, total_chunks, ctx
        )


def _foreach_gather_scalars_go(
    metadata_obj: PyObjectPtr,
    out_ptr_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
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


def _foreach_gather_scalars_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 3:
            raise Error("ForeachGatherScalars expects exactly three arguments")
        _foreach_gather_scalars_go(args[0], args[1], args[2])
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def _foreach_scalar_op_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 4:
            raise Error("ForeachScalarOp expects exactly four arguments")
        _foreach_scalar_op_go(args[0], args[1], args[2], args[3])
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def _foreach_lerp_scalar_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 5:
            raise Error("ForeachLerpScalar expects exactly five arguments")
        _foreach_lerp_scalar_go(args[0], args[1], args[2], args[3], args[4])
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def _foreach_addc_op_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 4:
            raise Error("ForeachAddcOp expects exactly four arguments")
        _foreach_addc_op_go(args[0], args[1], args[2], args[3])
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def _foreach_sqrt_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 2:
            raise Error("ForeachSqrt expects exactly two arguments")
        _foreach_sqrt_go(args[0], args[1])
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def _foreach_l2_norm_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 4:
            raise Error("ForeachL2Norm expects exactly four arguments")
        _foreach_l2_norm_go(args[0], args[1], args[2], args[3])
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def _foreach_mul_tensor_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 3:
            raise Error("ForeachMulTensor expects exactly three arguments")
        _foreach_mul_tensor_go(args[0], args[1], args[2])
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


@export
def PyInit_optimizer_ops() abi("C") -> PythonObject:
    try:
        var builder = PythonModuleBuilder("optimizer_ops")
        builder.def_py_c_function(
            _fused_adamw_dispatcher,
            "FusedAdamW",
            docstring=(
                "(metadata, scalars, dtype_mode, flags, lr_ptr, "
                "grad_scale_ptr, found_inf_ptr, context_ptr); fused FP32 AdamW"
            ),
        )
        builder.def_py_c_function(
            _foreach_l2_norm_dispatcher,
            "ForeachL2Norm",
            docstring=(
                "(metadata, partials_ptr, partials_numel, context_ptr); "
                "runtime-dynamic FP32 foreach L2 norms"
            ),
        )
        builder.def_py_c_function(
            _foreach_mul_tensor_dispatcher,
            "ForeachMulTensor",
            docstring=(
                "(metadata, scalar_ptr, context_ptr); runtime-dynamic FP32 "
                "foreach in-place device-scalar multiply"
            ),
        )
        builder.def_py_c_function(
            _foreach_scalar_op_dispatcher,
            "ForeachScalarOp",
            docstring=(
                "(op_code, metadata, scalars, context_ptr); batched FP32 "
                "in-place foreach mul/add/div by one host scalar per tensor"
            ),
        )
        builder.def_py_c_function(
            _foreach_lerp_scalar_dispatcher,
            "ForeachLerpScalar",
            docstring=(
                "(metadata, weight, one_minus_weight, low_branch, "
                "context_ptr); batched FP32 in-place foreach scalar lerp"
            ),
        )
        builder.def_py_c_function(
            _foreach_addc_op_dispatcher,
            "ForeachAddcOp",
            docstring=(
                "(op_code, metadata, scalars, context_ptr); batched FP32 "
                "in-place foreach addcmul/addcdiv with per-tensor scalars"
            ),
        )
        builder.def_py_c_function(
            _foreach_gather_scalars_dispatcher,
            "ForeachGatherScalars",
            docstring=(
                "(in_ptrs, out_ptr, context_ptr); batched FP32 gather of "
                "one scalar per input tensor into a contiguous output"
            ),
        )
        builder.def_py_c_function(
            _foreach_sqrt_dispatcher,
            "ForeachSqrt",
            docstring=(
                "(metadata, context_ptr); batched FP32 out-of-place "
                "foreach sqrt"
            ),
        )
        return builder.finalize()
    except e:
        abort(t"failed to create optimizer_ops python module: {e}")
