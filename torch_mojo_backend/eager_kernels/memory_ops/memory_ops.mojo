"""Memory movement kernels: strided copy and strided fill for any rank up to
MAX_RANK, dispatched on element size (copy) or dtype (fill). These used to
live in tensor_holder.mojo behind Python entry points; the native backend
calls them through `tmb_call` for `.contiguous()`, `copy_`, `fill_` on
non-contiguous tensors and every strided materialization.

Slots (all Int): CopyStrided(dst_ptr, src_ptr, shape8, dst_strides8,
src_strides8, itemsize, ctx_ptr); StridedFill(dst_ptr, value_f64_bits,
shape8, dst_strides8, dtype, ctx_ptr). A `*8` argument is a tuple slot
`[8, d0..d7]` leading-padded to MAX_RANK.
"""
from std.sys.info import has_apple_gpu_accelerator
from std.utils import IndexList

from max.gpu.host import DeviceContext

from op_utils import (
    MAX_RANK,
    Arg,
    Argv,
    _copy_strided,
    _fill_bits,
    _fill_bits_dtype,
    _fill_layout,
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _raw_tuple_int,
    _spec_dispatcher6,
    _spec_dispatcher7,
)
from variant_gates import ErrBuf, NO_OP_COMPILED, _op_on, _tmb_entry_error

comptime ALL_DTYPES = [
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
    DType.bool,
]


def _copy_strided_go(
    dst_ptr: Arg,
    src_ptr: Arg,
    shape_t: Arg,
    dst_strides_t: Arg,
    src_strides_t: Arg,
    itemsize_o: Arg,
    ctx_ptr: Arg,
) raises:
    var dst_addr = _raw_int(dst_ptr)
    var src_addr = _raw_int(src_ptr)
    var shape = IndexList[MAX_RANK](1)
    var dst_strides = IndexList[MAX_RANK](0)
    var src_strides = IndexList[MAX_RANK](0)
    for i in range(MAX_RANK):
        shape[i] = _raw_tuple_int(shape_t, i)
        dst_strides[i] = _raw_tuple_int(dst_strides_t, i)
        src_strides[i] = _raw_tuple_int(src_strides_t, i)
    var itemsize = _raw_int(itemsize_o)
    var ctx = _raw_ctx(ctx_ptr)
    if itemsize == 4:
        _copy_strided[DType.uint32](
            dst_addr, src_addr, shape, dst_strides, src_strides, ctx
        )
    elif itemsize == 2:
        _copy_strided[DType.uint16](
            dst_addr, src_addr, shape, dst_strides, src_strides, ctx
        )
    elif itemsize == 8:
        _copy_strided[DType.uint64](
            dst_addr, src_addr, shape, dst_strides, src_strides, ctx
        )
    elif itemsize == 1:
        _copy_strided[DType.uint8](
            dst_addr, src_addr, shape, dst_strides, src_strides, ctx
        )
    else:
        raise Error("CopyStrided: unsupported element size ", itemsize)


@always_inline
def _strided_fill[
    dtype: DType
](
    dst_addr: Int,
    value: Float64,
    shape: IndexList[MAX_RANK],
    dst_strides: IndexList[MAX_RANK],
    ctx: DeviceContext,
) raises:
    comptime if dtype == DType.float64 and has_apple_gpu_accelerator():
        if ctx.api() != "cpu":
            raise Error("float64 is not supported on Apple GPU")
    comptime BITS = _fill_bits_dtype[dtype]()
    _fill_layout[BITS](
        dst_addr, _fill_bits[dtype, BITS](value), shape, dst_strides, ctx
    )


def _strided_fill_go(
    dst_ptr: Arg,
    value_o: Arg,
    shape_t: Arg,
    dst_strides_t: Arg,
    dtype_o: Arg,
    ctx_ptr: Arg,
) raises:
    var dst_addr = _raw_int(dst_ptr)
    var value = _raw_f64(value_o)
    var shape = IndexList[MAX_RANK](1)
    var dst_strides = IndexList[MAX_RANK](0)
    for i in range(MAX_RANK):
        shape[i] = _raw_tuple_int(shape_t, i)
        dst_strides[i] = _raw_tuple_int(dst_strides_t, i)
    var dtype = _raw_dtype_int(dtype_o)
    var ctx = _raw_ctx(ctx_ptr)
    var handled = False
    comptime for dt in ALL_DTYPES:
        if dtype == dt:
            _strided_fill[dt](dst_addr, value, shape, dst_strides, ctx)
            handled = True
    if not handled:
        raise Error("StridedFill: unsupported dtype ", dtype)


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    try:
        comptime if _op_on["CopyStrided"]():
            _spec_dispatcher7[_copy_strided_go, "CopyStrided"](argv, argc)
            return 0
        comptime if _op_on["StridedFill"]():
            _spec_dispatcher6[_strided_fill_go, "StridedFill"](argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
