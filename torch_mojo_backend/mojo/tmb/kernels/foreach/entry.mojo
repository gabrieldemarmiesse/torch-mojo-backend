"""Eager bridge of the foreach family: every `aten::_foreach_*` elementwise
overload, one generic multi-tensor apply (`apply.mojo`).

One build per (kind, operand variant, dtype):
  * OP           the element math, a `functors.foreach_math` kind;
  * FE_VARIANT   the operand variant (`apply.fe_variant`);
  * DTYPE_ARG_0  the lists' dtype.

Slots: (metadata, scalars, device_addr, aux_addr, context_ptr).
  * metadata    five ints per tensor: the first, second and third operand
                addresses, the output address (0 in place), the numel;
  * scalars     one int64 per tensor: the host scalar as float64 bits for a
                floating list, as its value for an integer or bool list (an
                empty tuple when the overload has none);
  * device_addr the 0-d device tensor operand (0 if none);
  * aux_addr    GradScaler's found_inf flag (0 if none).
The ops layer has already validated ATen's contract (lengths, dtypes,
devices, shapes, aliasing); this packs descriptors FE_DESC_CAP at a time and
enqueues on the tensors' DeviceContext. Nothing is allocated, read back or
synchronized here.
"""

from std.collections import Array
from std.math import ceildiv
from std.sys.defines import get_defined_int
from std.sys.info import size_of

from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    _raw_ctx,
    _raw_f64,
    _raw_int,
    _raw_tuple_int,
    _raw_tuple_len,
    _spec_dispatcher5,
)
from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _dtype_out_on,
    _op_on,
    _tmb_entry_error,
)
from tmb.kernels.foreach.apply import (
    FE_DESC_CAP,
    ForeachDesc,
    empty_foreach_desc,
    fe_lists,
    fe_out_of_place,
    foreach_apply_enqueue,
    foreach_chunk_elements,
)
from tmb.kernels.foreach.contract import FE_REDUCE_CHUNK, FR_L2, FR_MAX
from tmb.kernels.foreach.reduce import (
    foreach_reduce_acc,
    foreach_reduce_enqueue,
)
from tmb.kernels.foreach.functors import (
    FOREACH_OTHER_KINDS,
    FOREACH_UNARY_KINDS,
)


comptime _VARIANT = get_defined_int["FE_VARIANT", 0]()
comptime _NORM = get_defined_int["FE_NORM", FR_L2]()
comptime _ROOT = get_defined_int["FE_ROOT", 1]() != 0
comptime _RECORD_FIELDS = 5

# Every storage dtype a foreach list can have (bool is handled on its own:
# uint8 storage, int32 arithmetic).
comptime _DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
]


def _foreach_go[
    kind: StaticString
](
    metadata_obj: Arg,
    scalars_obj: Arg,
    device_addr_obj: Arg,
    aux_addr_obj: Arg,
    device_context_ptr: Arg,
) raises:
    comptime V = _VARIANT
    comptime lists = fe_lists[V]()
    comptime out_of_place = fe_out_of_place[V]()
    comptime assert lists >= 1, "FE_VARIANT names no input list"
    var value_count = _raw_tuple_len(metadata_obj)
    if value_count == 0 or value_count % _RECORD_FIELDS != 0:
        raise Error("mojo foreach ", kind, ": invalid metadata field count")
    var record_count = value_count // _RECORD_FIELDS
    var scalar_count = _raw_tuple_len(scalars_obj)
    if scalar_count != 0 and scalar_count != record_count:
        raise Error("mojo foreach ", kind, ": one scalar per tensor expected")
    var total_elements = 0
    for record in range(record_count):
        var base = record * _RECORD_FIELDS
        var numel = _raw_tuple_int(metadata_obj, base + 4)
        if numel < 0:
            raise Error("mojo foreach ", kind, ": negative numel")
        if numel > 0:
            var missing = _raw_tuple_int(metadata_obj, base) == 0
            if lists >= 2 and _raw_tuple_int(metadata_obj, base + 1) == 0:
                missing = True
            if lists >= 3 and _raw_tuple_int(metadata_obj, base + 2) == 0:
                missing = True
            if out_of_place and _raw_tuple_int(metadata_obj, base + 3) == 0:
                missing = True
            if missing:
                raise Error(
                    "mojo foreach ", kind, ": nonempty pointer must be nonzero"
                )
        total_elements += numel
    var device_addr = _raw_int(device_addr_obj)
    var aux_addr = _raw_int(aux_addr_obj)
    var ctx = _raw_ctx(device_context_ptr)

    # The chunk size is what fills the device, so it is derived from the whole
    # list's element count once, before any descriptor batch is packed.
    var chunk_elements = foreach_chunk_elements(total_elements, ctx)

    var record = 0
    while record < record_count:
        var descs = Array[ForeachDesc, FE_DESC_CAP](fill=empty_foreach_desc())
        var scalars = Array[Int64, FE_DESC_CAP](fill=0)
        var desc_count = 0
        var total_chunks = 0
        while record < record_count and desc_count < FE_DESC_CAP:
            var base = record * _RECORD_FIELDS
            var numel = _raw_tuple_int(metadata_obj, base + 4)
            total_chunks += ceildiv(numel, chunk_elements)
            descs[desc_count] = ForeachDesc(
                _raw_tuple_int(metadata_obj, base),
                _raw_tuple_int(metadata_obj, base + 1),
                _raw_tuple_int(metadata_obj, base + 2),
                _raw_tuple_int(metadata_obj, base + 3),
                numel,
                total_chunks,
            )
            if scalar_count != 0:
                scalars[desc_count] = Int64(_raw_tuple_int(scalars_obj, record))
            record += 1
            desc_count += 1

        var launched = False
        comptime for dt in _DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                foreach_apply_enqueue[dt, kind, V, False](
                    descs,
                    scalars,
                    desc_count,
                    total_chunks,
                    chunk_elements,
                    device_addr,
                    aux_addr,
                    ctx,
                )
                launched = True
        comptime if _dtype_arg_on[0, DType.bool]():
            foreach_apply_enqueue[DType.uint8, kind, V, True](
                descs,
                scalars,
                desc_count,
                total_chunks,
                chunk_elements,
                device_addr,
                aux_addr,
                ctx,
            )
            launched = True
        if not launched:
            raise Error(
                "mojo foreach ", kind, ": no dtype compiled (DTYPE_ARG_0)"
            )


# The floating dtypes a norm reads and writes (and accumulates in their
# opmath), and every dtype max takes (bool as its uint8 storage).
comptime _NORM_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
]
comptime _MAX_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
]

# A norm / powsum reads any of these (cast to the floating result dtype
# first): every max dtype plus the wide unsigned ones, which max refuses.
comptime _NORM_INPUT_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
]


def _reduce_batches[
    dtype: DType, out_dtype: DType, norm: Int
](
    metadata_obj: Arg,
    partials_addr: Int,
    p: Float64,
    one_element_abs: Bool,
    device_context_ptr: Arg,
) raises:
    comptime acc = foreach_reduce_acc[dtype, out_dtype, norm]()
    var record_count = _raw_tuple_len(metadata_obj) // 3
    var ctx = _raw_ctx(device_context_ptr)
    var record = 0
    var partial_offset = 0
    while record < record_count:
        var descs = Array[ForeachDesc, FE_DESC_CAP](fill=empty_foreach_desc())
        var desc_count = 0
        var total_chunks = 0
        while record < record_count and desc_count < FE_DESC_CAP:
            var base = record * 3
            var numel = _raw_tuple_int(metadata_obj, base + 2)
            total_chunks += ceildiv(numel, FE_REDUCE_CHUNK)
            descs[desc_count] = ForeachDesc(
                _raw_tuple_int(metadata_obj, base),
                0,
                0,
                _raw_tuple_int(metadata_obj, base + 1),
                numel,
                total_chunks,
            )
            record += 1
            desc_count += 1
        foreach_reduce_enqueue[dtype, acc, out_dtype, norm, _ROOT](
            descs,
            desc_count,
            total_chunks,
            partials_addr + partial_offset * size_of[acc](),
            p,
            one_element_abs,
            ctx,
        )
        partial_offset += total_chunks


def _reduce_go[
    max_op: Bool
](
    metadata_obj: Arg,
    partials_obj: Arg,
    p_obj: Arg,
    one_element_abs_obj: Arg,
    device_context_ptr: Arg,
) raises:
    """`_foreach_norm` / `_foreach_powsum` (FE_NORM, FE_ROOT; DTYPE_OUT the
    result dtype) or `_foreach_max`. Slots: (metadata, partials, p,
    one_element_abs, context): three ints per tensor (input, 0-d output,
    numel), the scratch of one accumulator per FE_REDUCE_CHUNK chunk of the
    list, the order of an FR_LP norm (float64 bits), and whether a
    one-element tensor takes linalg_vector_norm's |x| shortcut."""
    var value_count = _raw_tuple_len(metadata_obj)
    if value_count == 0 or value_count % 3 != 0:
        raise Error("mojo foreach reduction: invalid metadata field count")
    for record in range(value_count // 3):
        var numel = _raw_tuple_int(metadata_obj, record * 3 + 2)
        if numel < 0 or _raw_tuple_int(metadata_obj, record * 3 + 1) == 0:
            raise Error("mojo foreach reduction: invalid tensor record")
        if numel > 0 and _raw_tuple_int(metadata_obj, record * 3) == 0:
            raise Error("mojo foreach reduction: nonempty pointer is zero")
    var partials_addr = _raw_int(partials_obj)
    var p = _raw_f64(p_obj)
    var one_element_abs = _raw_int(one_element_abs_obj) != 0
    var launched = False
    comptime if max_op:
        comptime for dt in _MAX_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                _reduce_batches[dt, dt, FR_MAX](
                    metadata_obj, partials_addr, p, False, device_context_ptr
                )
                launched = True
        comptime if _dtype_arg_on[0, DType.bool]():
            _reduce_batches[DType.uint8, DType.uint8, FR_MAX](
                metadata_obj, partials_addr, p, False, device_context_ptr
            )
            launched = True
    else:
        # The input: any dtype (an integer or bool list reaches a norm
        # through the slow route with a floating `dtype`); the result: a
        # floating dtype.
        comptime for dt in _NORM_INPUT_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                comptime for odt in _NORM_DTYPES:
                    comptime if _dtype_out_on[0, odt]():
                        _reduce_batches[dt, odt, _NORM](
                            metadata_obj,
                            partials_addr,
                            p,
                            one_element_abs,
                            device_context_ptr,
                        )
                        launched = True
        comptime if _dtype_arg_on[0, DType.bool]():
            comptime for odt in _NORM_DTYPES:
                comptime if _dtype_out_on[0, odt]():
                    _reduce_batches[DType.uint8, odt, _NORM](
                        metadata_obj,
                        partials_addr,
                        p,
                        one_element_abs,
                        device_context_ptr,
                    )
                    launched = True
    if not launched:
        raise Error("mojo foreach reduction: no dtype compiled")


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kind per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["norm"]():
            _spec_dispatcher5[_reduce_go[False], "foreach norm"](argv, argc)
            return 0
        comptime if _op_on["max"]():
            _spec_dispatcher5[_reduce_go[True], "foreach max"](argv, argc)
            return 0
        comptime for kind in FOREACH_UNARY_KINDS:
            comptime if _op_on[kind]():
                _spec_dispatcher5[_foreach_go[kind], "foreach"](argv, argc)
                return 0
        comptime for kind in FOREACH_OTHER_KINDS:
            comptime if _op_on[kind]():
                _spec_dispatcher5[_foreach_go[kind], "foreach"](argv, argc)
                return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
