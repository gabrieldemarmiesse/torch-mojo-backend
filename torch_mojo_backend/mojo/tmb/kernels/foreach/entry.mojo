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

from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    _raw_ctx,
    _raw_int,
    _raw_tuple_int,
    _raw_tuple_len,
    _spec_dispatcher5,
)
from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
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
from tmb.kernels.foreach.functors import (
    FOREACH_OTHER_KINDS,
    FOREACH_UNARY_KINDS,
)


comptime _VARIANT = get_defined_int["FE_VARIANT", 0]()
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


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kind per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
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
