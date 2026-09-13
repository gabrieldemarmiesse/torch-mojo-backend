# ===----------------------------------------------------------------------=== #
# Dynamic CPU/GPU binary search for aten::searchsorted and aten::bucketize.
#
# One logical worker searches one input value.  A 1-D boundary is shared by
# every value; an N-D boundary selects the matching flattened prefix row.
# Optional sorter entries are relative indices within the final dimension,
# matching ATen.  Python validates the sorter's device, shape and dtype
# before this raw-pointer bridge is called, but NOT its index values (that
# would need a device-to-host sync); a sorter entry outside the boundary
# range is clamped in `_binary_search_position` instead of trusted raw.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.host import DeviceContext
from std.sys.info import has_accelerator
from std.utils.coord import Coord
from std.utils.static_tuple import StaticTuple

from op_utils import (
    Arg,
    Argv,
    GS_THREADS,
    _enqueue_cached,
    _gs_blocks,
    _make_ptr,
    _parallel_for,
    _raw_ctx,
    _raw_dtype_int,
    _raw_int,
    _spec_dispatcher13,
)
from variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _dtype_out_on,
    _op_on,
    _tmb_entry_error,
)


comptime SEARCH_DTYPES = [
    DType.float32,
    DType.bfloat16,
    DType.float16,
    DType.int32,
    DType.int64,
]
comptime OUTPUT_DTYPES = [DType.int32, DType.int64]


@always_inline
def _should_advance[
    dtype: DType, right: Bool
](boundary: SIMD[dtype, 1], value: SIMD[dtype, 1]) -> Bool:
    # Match ATen's negated comparison exactly.  Besides using one compare,
    # this is what makes a NaN in either operand advance the search.
    comptime if right:
        return not boundary.gt(value)[0]
    else:
        return not boundary.ge(value)[0]


@always_inline
def _binary_search_position[
    dtype: DType,
    boundaries_are_1d: Bool,
    has_sorter: Bool,
    right: Bool,
](
    boundaries: Pointer[Scalar[dtype], ImmutAnyOrigin],
    values: Pointer[Scalar[dtype], ImmutAnyOrigin],
    sorter: Pointer[Scalar[DType.int64], ImmutAnyOrigin],
    value_index: Int,
    boundary_size: Int,
    values_per_batch: Int,
) -> Int:
    var boundary_base = 0
    comptime if not boundaries_are_1d:
        boundary_base = (value_index // values_per_batch) * boundary_size

    var value = SIMD[dtype, 1](values[unsafe_offset=value_index])
    var low = 0
    var high = boundary_size
    while low < high:
        var mid = low + ((high - low) >> 1)
        var boundary_index = mid
        comptime if has_sorter:
            # Python checks the sorter's device/shape/dtype but not its
            # values (no device-to-host sync per call): clamp the gathered
            # index so an out-of-range entry reads some in-bounds boundary
            # (an unspecified result) instead of out of bounds.
            var raw_index = Int(sorter[unsafe_offset=boundary_base + mid])
            boundary_index = max(0, min(raw_index, boundary_size - 1))
        var boundary = SIMD[dtype, 1](
            boundaries[unsafe_offset=boundary_base + boundary_index]
        )
        var advance = _should_advance[dtype, right](boundary, value)
        # Conditional expressions lower to selects: the data-dependent update
        # is branchless even though every lane takes a different search path.
        low = mid + 1 if advance else low
        high = high if advance else mid
    return low


@__name(
    t"binary_search_global_{dtype}_{out_dtype}_d1{boundaries_are_1d}_s{has_sorter}_r{right}_t{GS_THREADS}"
)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(GS_THREADS))
)
def _binary_search_kernel[
    dtype: DType,
    out_dtype: DType,
    boundaries_are_1d: Bool,
    has_sorter: Bool,
    right: Bool,
](
    out_ptr: Pointer[Scalar[out_dtype], MutAnyOrigin],
    boundaries: Pointer[Scalar[dtype], ImmutAnyOrigin],
    values: Pointer[Scalar[dtype], ImmutAnyOrigin],
    sorter: Pointer[Scalar[DType.int64], ImmutAnyOrigin],
    num_values_arg: Int64,
    boundary_size_arg: Int64,
    values_per_batch_arg: Int64,
):
    var num_values = Int(num_values_arg)
    var boundary_size = Int(boundary_size_arg)
    var values_per_batch = Int(values_per_batch_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < num_values:
        out_ptr[unsafe_offset=i] = _binary_search_position[
            dtype, boundaries_are_1d, has_sorter, right
        ](
            boundaries,
            values,
            sorter,
            i,
            boundary_size,
            values_per_batch,
        ).cast[
            out_dtype
        ]()
        i += stride


@always_inline
def _searchsorted[
    dtype: DType,
    out_dtype: DType,
    boundaries_are_1d: Bool,
    has_sorter: Bool,
    right: Bool,
](
    out_addr: Int,
    boundaries_addr: Int,
    values_addr: Int,
    sorter_addr: Int,
    num_values: Int,
    boundary_size: Int,
    values_per_batch: Int,
    ctx: DeviceContext,
) raises:
    var out = _make_ptr[out_dtype](out_addr).as_unsafe_any_origin()
    var boundaries = (
        _make_ptr[dtype](boundaries_addr).as_unsafe_any_origin().as_imm()
    )
    var values = _make_ptr[dtype](values_addr).as_unsafe_any_origin().as_imm()
    var sorter = (
        _make_ptr[DType.int64](sorter_addr).as_unsafe_any_origin().as_imm()
    )

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(
            out,
            boundaries,
            values,
            sorter,
            boundary_size,
            values_per_batch,
        )
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value())
            out[unsafe_offset=i] = _binary_search_position[
                dtype, boundaries_are_1d, has_sorter, right
            ](
                boundaries,
                values,
                sorter,
                i,
                boundary_size,
                values_per_batch,
            ).cast[
                out_dtype
            ]()

        _parallel_for[func](num_values, ctx)
        return

    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        _enqueue_cached[
            _binary_search_kernel[
                dtype, out_dtype, boundaries_are_1d, has_sorter, right
            ]
        ](
            ctx,
            String(
                t"binary_search_{dtype}_{out_dtype}_d1{boundaries_are_1d}_s{has_sorter}_r{right}"
            ),
            _gs_blocks(num_values),
            1,
            1,
            GS_THREADS,
            out,
            boundaries,
            values,
            sorter,
            Int64(num_values),
            Int64(boundary_size),
            Int64(values_per_batch),
        )


@always_inline
def _dispatch_searchsorted[
    dtype: DType, out_dtype: DType
](
    out_addr: Int,
    boundaries_addr: Int,
    values_addr: Int,
    sorter_addr: Int,
    num_values: Int,
    boundary_size: Int,
    values_per_batch: Int,
    boundaries_are_1d: Bool,
    has_sorter: Bool,
    right: Bool,
    ctx: DeviceContext,
) raises:
    @always_inline
    @parameter
    def _run[d1: Bool, sorter: Bool, upper: Bool]() raises:
        _searchsorted[dtype, out_dtype, d1, sorter, upper](
            out_addr,
            boundaries_addr,
            values_addr,
            sorter_addr,
            num_values,
            boundary_size,
            values_per_batch,
            ctx,
        )

    if boundaries_are_1d:
        if has_sorter:
            if right:
                _run[True, True, True]()
            else:
                _run[True, True, False]()
        elif right:
            _run[True, False, True]()
        else:
            _run[True, False, False]()
    elif has_sorter:
        if right:
            _run[False, True, True]()
        else:
            _run[False, True, False]()
    elif right:
        _run[False, False, True]()
    else:
        _run[False, False, False]()


def _searchsorted_go(
    out_obj: Arg,
    boundaries_obj: Arg,
    values_obj: Arg,
    sorter_obj: Arg,
    num_values_obj: Arg,
    boundary_size_obj: Arg,
    values_per_batch_obj: Arg,
    boundaries_are_1d_obj: Arg,
    has_sorter_obj: Arg,
    right_obj: Arg,
    dtype_obj: Arg,
    out_dtype_obj: Arg,
    ctx_obj: Arg,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var out_dtype = _raw_dtype_int(out_dtype_obj)
    var handled = False
    comptime for dt in SEARCH_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                comptime for odt in OUTPUT_DTYPES:
                    comptime if _dtype_out_on[0, odt]():
                        if out_dtype == odt:
                            _dispatch_searchsorted[dt, odt](
                                _raw_int(out_obj),
                                _raw_int(boundaries_obj),
                                _raw_int(values_obj),
                                _raw_int(sorter_obj),
                                _raw_int(num_values_obj),
                                _raw_int(boundary_size_obj),
                                _raw_int(values_per_batch_obj),
                                Bool(_raw_int(boundaries_are_1d_obj)),
                                Bool(_raw_int(has_sorter_obj)),
                                Bool(_raw_int(right_obj)),
                                _raw_ctx(ctx_obj),
                            )
                            handled = True
    if not handled:
        raise Error(
            "unsupported dtype specialization for searchsorted: "
            + String(dtype)
            + " -> "
            + String(out_dtype)
        )


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["Searchsorted"]():
            _spec_dispatcher13[_searchsorted_go, "Searchsorted"](argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
