# ===----------------------------------------------------------------------=== #
# The device passes of aten::_unique / _unique2 / unique_dim /
# unique_consecutive / unique_dim_consecutive (ATen's native/cuda/UniqueCub.cu
# and Unique.cu). The op sorts with the repo's stable sort, then:
#
#   UniqueMarks   marks[i] = row(i) != row(i - 1)   (adjacent difference)
#   (cumsum of the marks through the dispatcher: the group id of every row)
#   UniqueSelect  one representative row per group -- the first of its run,
#                 or the last (cub's run_length_encode keeps the last, which
#                 return_counts goes through) -- its source row, the inverse
#                 index of every input row and the start of every run
#   UniqueCounts  counts[g] = start[g + 1] - start[g]
#
# A "row" is `inner` consecutive elements (1 for the flat ops) read through
# `perm` (the sort's permutation; address 0 means the identity). Rows compare
# with `!=` element by element, so a NaN never equals anything, as on CUDA.
# ===----------------------------------------------------------------------=== #

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
)
from std.sys.info import has_accelerator, has_apple_gpu_accelerator
from std.utils.static_tuple import StaticTuple

from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    GS_THREADS,
    _enqueue_cached,
    _gs_blocks,
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_int,
    _raw_tuple_int,
    _spec_dispatcher4,
    _spec_dispatcher5,
    _spec_dispatcher6,
)
from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)

comptime UNIQUE_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.int64,
    DType.int32,
    DType.int16,
    DType.int8,
    DType.uint8,
    DType.bool,
]


@always_inline
def _row(perm: Pointer[Int64, MutAnyOrigin], has_perm: Bool, i: Int) -> Int:
    return Int(perm[unsafe_offset=i]) if has_perm else i


@__name(t"unique_marks_{dtype}_t{GS_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(GS_THREADS))
)
def _marks_kernel[
    dtype: DType
](
    marks: Pointer[Int64, MutAnyOrigin],
    data: Pointer[Scalar[dtype], MutAnyOrigin],
    perm: Pointer[Int64, MutAnyOrigin],
    has_perm: Int64,
    n: Int64,
    inner: Int64,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var step = Int(grid_dim.x) * Int(block_dim.x)
    var w = Int(inner)
    while i < Int(n):
        var differs = False
        if i > 0:
            var a = _row(perm, has_perm != 0, i) * w
            var b = _row(perm, has_perm != 0, i - 1) * w
            for k in range(w):
                if data[unsafe_offset=a + k] != data[unsafe_offset=b + k]:
                    differs = True
                    break
        marks[unsafe_offset=i] = 1 if differs else 0
        i += step


@__name(t"unique_select_{dtype}_t{GS_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(GS_THREADS))
)
def _select_kernel[
    dtype: DType
](
    out_vals: Pointer[Scalar[dtype], MutAnyOrigin],
    out_rows: Pointer[Int64, MutAnyOrigin],
    inverse: Pointer[Int64, MutAnyOrigin],
    starts: Pointer[Int64, MutAnyOrigin],
    data: Pointer[Scalar[dtype], MutAnyOrigin],
    perm: Pointer[Int64, MutAnyOrigin],
    gid: Pointer[Int64, MutAnyOrigin],
    # bit 0 has_perm, 1 take_last, 2 out_vals, 3 out_rows, 4 inverse,
    # 5 starts
    flags: Int64,
    n: Int64,
    inner: Int64,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var step = Int(grid_dim.x) * Int(block_dim.x)
    var w = Int(inner)
    var f = Int(flags)
    var has_perm = f & 1 != 0
    while i < Int(n):
        var g = gid[unsafe_offset=i]
        var is_start = i == 0 or gid[unsafe_offset=i - 1] != g
        var is_end = i == Int(n) - 1 or gid[unsafe_offset=i + 1] != g
        var r = _row(perm, has_perm, i)
        var chosen = is_end if f & 2 != 0 else is_start
        if chosen:
            if f & 4 != 0:
                for k in range(w):
                    out_vals[unsafe_offset=Int(g) * w + k] = data[
                        unsafe_offset=r * w + k
                    ]
            if f & 8 != 0:
                out_rows[unsafe_offset=Int(g)] = Int64(r)
        if f & 16 != 0:
            inverse[unsafe_offset=r] = g
        if f & 32 != 0 and is_start:
            starts[unsafe_offset=Int(g)] = Int64(i)
        i += step


@__name(t"unique_counts_t{GS_THREADS}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(GS_THREADS))
)
def _counts_kernel(
    counts: Pointer[Int64, MutAnyOrigin],
    starts: Pointer[Int64, MutAnyOrigin],
    m: Int64,
    n: Int64,
):
    var g = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var step = Int(grid_dim.x) * Int(block_dim.x)
    while g < Int(m):
        var stop = n if g + 1 == Int(m) else starts[unsafe_offset=g + 1]
        counts[unsafe_offset=g] = stop - starts[unsafe_offset=g]
        g += step


def _ptr[dt: DType](a: Arg) -> Pointer[Scalar[dt], MutAnyOrigin]:
    return _make_ptr[dt](_raw_int(a)).as_unsafe_any_origin()


def _marks_go(
    marks_o: Arg,
    data_o: Arg,
    perm_o: Arg,
    params: Arg,  # (n, inner)
    dtype_o: Arg,
    ctx_o: Arg,
) raises:
    var dtype = _raw_dtype_int(dtype_o)
    var ctx = _raw_ctx(ctx_o)
    var n = _raw_tuple_int(params, 0)
    var handled = False
    comptime for dt in UNIQUE_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                handled = True
                comptime if not has_accelerator():
                    raise Error("no GPU accelerator available at compile time")
                elif dt == DType.float64 and has_apple_gpu_accelerator():
                    raise Error("float64 is not supported on Apple GPU")
                else:
                    _enqueue_cached[_marks_kernel[dt]](
                        ctx,
                        _gs_blocks(n),
                        1,
                        1,
                        GS_THREADS,
                        _ptr[DType.int64](marks_o),
                        _ptr[dt](data_o),
                        _ptr[DType.int64](perm_o),
                        Int64(1 if _raw_int(perm_o) != 0 else 0),
                        Int64(n),
                        Int64(_raw_tuple_int(params, 1)),
                    )
    if not handled:
        raise Error("UniqueMarks: unsupported dtype ", dtype)


def _select_go(
    ptrs: Arg,  # (out_vals, out_rows, inverse, starts, data, perm, gid)
    params: Arg,  # (take_last, n, inner)
    dtype_o: Arg,
    ctx_o: Arg,
) raises:
    var dtype = _raw_dtype_int(dtype_o)
    var ctx = _raw_ctx(ctx_o)
    var n = _raw_tuple_int(params, 1)
    var flags = 0
    if _raw_tuple_int(ptrs, 5) != 0:
        flags |= 1
    if _raw_tuple_int(params, 0) != 0:
        flags |= 2
    for b in range(4):
        if _raw_tuple_int(ptrs, b) != 0:
            flags |= 4 << b
    var handled = False
    comptime for dt in UNIQUE_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                handled = True
                comptime if not has_accelerator():
                    raise Error("no GPU accelerator available at compile time")
                elif dt == DType.float64 and has_apple_gpu_accelerator():
                    raise Error("float64 is not supported on Apple GPU")
                else:
                    _enqueue_cached[_select_kernel[dt]](
                        ctx,
                        _gs_blocks(n),
                        1,
                        1,
                        GS_THREADS,
                        _make_ptr[dt](
                            _raw_tuple_int(ptrs, 0)
                        ).as_unsafe_any_origin(),
                        _make_ptr[DType.int64](
                            _raw_tuple_int(ptrs, 1)
                        ).as_unsafe_any_origin(),
                        _make_ptr[DType.int64](
                            _raw_tuple_int(ptrs, 2)
                        ).as_unsafe_any_origin(),
                        _make_ptr[DType.int64](
                            _raw_tuple_int(ptrs, 3)
                        ).as_unsafe_any_origin(),
                        _make_ptr[dt](
                            _raw_tuple_int(ptrs, 4)
                        ).as_unsafe_any_origin(),
                        _make_ptr[DType.int64](
                            _raw_tuple_int(ptrs, 5)
                        ).as_unsafe_any_origin(),
                        _make_ptr[DType.int64](
                            _raw_tuple_int(ptrs, 6)
                        ).as_unsafe_any_origin(),
                        Int64(flags),
                        Int64(n),
                        Int64(_raw_tuple_int(params, 2)),
                    )
    if not handled:
        raise Error("UniqueSelect: unsupported dtype ", dtype)


def _counts_go(
    counts_o: Arg, starts_o: Arg, m_o: Arg, n_o: Arg, ctx_o: Arg
) raises:
    var m = _raw_int(m_o)
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        _enqueue_cached[_counts_kernel](
            _raw_ctx(ctx_o),
            _gs_blocks(m),
            1,
            1,
            GS_THREADS,
            _ptr[DType.int64](counts_o),
            _ptr[DType.int64](starts_o),
            Int64(m),
            Int64(_raw_int(n_o)),
        )


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["UniqueMarks"]():
            _spec_dispatcher6[_marks_go, "UniqueMarks"](argv, argc)
            return 0
        comptime if _op_on["UniqueSelect"]():
            _spec_dispatcher4[_select_go, "UniqueSelect"](argv, argc)
            return 0
        comptime if _op_on["UniqueCounts"]():
            _spec_dispatcher5[_counts_go, "UniqueCounts"](argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
