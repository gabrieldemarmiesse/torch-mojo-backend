"""Core aten ops: empty/empty_strided, transfers, views, scalar readback,
fills, record_stream.

Each op takes the record stack of its schema (abi.mojo) and writes its
result records. Views and the two empty factories never launch a kernel;
transfers are MAX copies; fills are memsets on contiguous memory. The
value-producing factories (arange, normal_, rand...) are ops_factories.mojo.
"""
from std.utils import IndexList

from abi import (
    Values,
    Value,
    T,
    IntList,
    DoubleList,
    v_is_none,
    v_int,
    v_int_or,
    v_f64,
    v_f64_or,
    v_bool,
    v_bool_or,
    v_scalar_is_integral,
    v_scalar_is_bool,
    v_dtype_or,
    v_device_index,
    v_memory_format_or,
    v_generator,
    v_stream,
    v_string,
    v_tensor,
    v_opt_tensor,
    v_tensor_list,
    ret_tensor,
    ret_ref,
    ret_int,
    ret_bool,
    ret_f64,
    ret_scalar_int,
    ret_scalar_f64,
    ret_scalar_bool,
    ret_tensor_list,
    contiguous_strides,
    strides_for_memory_format,
    new_strided,
    new_tensor,
    new_like,
    own,
    own_if_new,
    call_op,
    tensor_arg,
    bool_arg,
    new_like_dtype,
    new_scalar,
    view_strided,
    set_sizes_strides,
    retain,
    release,
    cpu_empty,
    default_dtype,
    unsupported,
    check,
    max_dtype,
    torch_dtype,
    is_floating,
    MEMORY_FORMAT_CONTIGUOUS,
    TAG_NONE,
    TAG_TENSOR,
    TAG_TENSOR_REF,
)
from device import (
    copy_d2d,
    copy_from_host,
    copy_to_host,
    ctx_for,
    current_device,
    dev,
    read_bytes_sync,
    record_stream,
)
from op_utils import MAX_RANK
from ops_common import cast_to, contiguous, copy_strided_into, fill_value
from registry import Site, impl, op_address_of


def _target_device(v: Value) -> Int:
    """Device? argument of a factory: its index, else the current device."""
    var i = v_device_index(v)
    return i if i >= 0 else current_device()


def _shape_of(sizes: IntList) raises -> IndexList[MAX_RANK]:
    if len(sizes) > MAX_RANK:
        raise Error(
            "tensor rank ",
            len(sizes),
            " exceeds the mojo device limit of ",
            MAX_RANK,
        )
    var shape = IndexList[MAX_RANK](1)
    var pad = MAX_RANK - len(sizes)
    for i in range(len(sizes)):
        if sizes[i] < 0:
            raise Error("negative dimension ", sizes[i])
        shape[pad + i] = sizes[i]
    return shape


# aten::empty.memory_format(SymInt[] size, *, ScalarType? dtype, Layout? layout,
#   Device? device, bool? pin_memory, MemoryFormat? memory_format) -> Tensor
def op_empty_memory_format(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var sizes = IntList(args[unsafe_offset=0])
    var stype = v_dtype_or(args[unsafe_offset=1], default_dtype())
    var device = _target_device(args[unsafe_offset=3])
    var mf = v_memory_format_or(args[unsafe_offset=5], MEMORY_FORMAT_CONTIGUOUS)
    var shape = _shape_of(sizes)
    var rank = len(sizes)
    ret_tensor(
        rets,
        0,
        new_strided(
            shape,
            strides_for_memory_format(shape, rank, mf),
            rank,
            stype,
            device,
        ),
    )


# aten::empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype,
#   Layout? layout, Device? device, bool? pin_memory) -> Tensor
def op_empty_strided(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var sizes = IntList(args[unsafe_offset=0])
    var strides = IntList(args[unsafe_offset=1])
    var stype = v_dtype_or(args[unsafe_offset=2], default_dtype())
    var device = _target_device(args[unsafe_offset=4])
    var shape = _shape_of(sizes)
    ret_tensor(
        rets,
        0,
        new_strided(
            shape, _strides_of(strides, len(sizes)), len(sizes), stype, device
        ),
    )


def _strides_of(strides: IntList, rank: Int) raises -> IndexList[MAX_RANK]:
    if len(strides) != rank:
        raise Error("expected ", rank, " strides, got ", len(strides))
    var strd = IndexList[MAX_RANK](0)
    var pad = MAX_RANK - rank
    for i in range(rank):
        strd[pad + i] = strides[i]
    return strd


def _viewed_as(t: T, like: T) raises -> T:
    """A view of the contiguous buffer `t` carrying `like`'s logical shape
    (same dtype, same element count). `copy_strided_into` walks one shape
    with both operands' strides and so needs them equal, while `copy_` is
    allowed to feed it a source of another shape: `dst(2,3).copy_(src(1,2,3))`
    broadcasts to the same elements in the same order."""
    return view_strided(
        t,
        like.shape,
        contiguous_strides(like.shape, like.rank),
        like.rank,
        t.offset,
    )


def _device_copy(dst: T, src: T) raises:
    """mojo -> mojo, same device: any layouts, any dtype pair, and any two
    logical shapes of the same element count (op_copy_from checked that)."""
    if src.stype != dst.stype:
        var dense = own_if_new(contiguous(src), src)
        var tmp = own_if_new(cast_to(dense.t, dst.stype), dense.t)
        if dst.contig:
            copy_d2d(
                ctx_for(dst.device),
                dst.ptr,
                tmp.t.ptr,
                dst.numel * dst.itemsize,
            )
        else:
            var shaped = own(_viewed_as(tmp.t, dst))
            copy_strided_into(dst, shaped.t)
        return
    # Two contiguous buffers of equal numel and dtype hold their elements in
    # the same order whatever their shapes: a flat byte copy is exact.
    if src.contig and dst.contig:
        copy_d2d(
            ctx_for(dst.device), dst.ptr, src.ptr, dst.numel * dst.itemsize
        )
    elif dst.same_shape(src):
        copy_strided_into(dst, src)
    else:
        var dense = own_if_new(contiguous(src), src)
        var shaped = own(_viewed_as(dense.t, dst))
        copy_strided_into(dst, shaped.t)


def _host_copy(dst: T, src: T) raises:
    """dst = src for two host tensors, through torch's CPU copy_ (layouts and
    dtypes handled by ATen)."""
    var args = List[Value]()
    args.append(tensor_arg(dst))
    args.append(tensor_arg(src))
    args.append(bool_arg(False))
    _ = call_op("aten::copy_", "", args^, 1)  # Results releases copy_'s handle


# aten::_copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> Tensor
def op_copy_from(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var src = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=1])
    if src.numel != dst.numel:
        raise Error(
            "_copy_from: element count mismatch (",
            src.numel,
            " vs ",
            dst.numel,
            ")",
        )
    if dst.numel == 0:
        ret_ref(rets, 0, dst)
        return
    if dst.on_mojo() and src.on_mojo():
        if dst.device != src.device:
            unsupported("copy between two mojo devices")
        _device_copy(dst, src)
    elif dst.on_mojo():
        if not src.on_cpu():
            unsupported(
                "copy from a device "
                + String(src.device_type)
                + " tensor to the mojo device"
            )
        # host -> device: a dense host copy in dst's dtype (torch's CPU copy_
        # does the layout / dtype work), one H2D, then a device relayout if needed
        var host = own_if_new(
            src.copy() if src.contig
            and src.stype
            == dst.stype else cpu_empty(dst.shape, dst.rank, dst.stype),
            src,
        )
        if host.t.h != src.h:
            _host_copy(host.t, src)
        var nbytes = dst.numel * dst.itemsize
        # `host` is dense in dst's dtype, so a contiguous dst takes the bytes
        # whatever the two logical shapes are.
        if dst.contig:
            copy_from_host(
                dst.device, ctx_for(dst.device), dst.ptr, host.t.ptr, nbytes
            )
        else:
            var tmp = own(new_like(dst))
            copy_from_host(
                dst.device, ctx_for(dst.device), tmp.t.ptr, host.t.ptr, nbytes
            )
            copy_strided_into(dst, tmp.t)
    elif src.on_mojo():
        if not dst.on_cpu():
            unsupported(
                "copy from the mojo device to a device "
                + String(dst.device_type)
                + " tensor"
            )
        # device -> host: one D2H of a dense copy in src's dtype, then torch's
        # CPU copy_ for the host layout / dtype when dst is not that already
        var dense = own_if_new(contiguous(src), src)
        var nbytes = src.numel * src.itemsize
        # Both dense in the same dtype: the bytes land in the right order
        # whatever the two logical shapes are.
        if dst.contig and dst.stype == src.stype:
            copy_to_host(ctx_for(src.device), dense.t.ptr, dst.ptr, nbytes)
        else:
            var host = own(cpu_empty(src.shape, src.rank, src.stype))
            copy_to_host(ctx_for(src.device), dense.t.ptr, host.t.ptr, nbytes)
            _host_copy(dst, host.t)
    else:
        raise Error("_copy_from: neither tensor is on the mojo device")
    ret_ref(rets, 0, dst)


def _infer_view_shape(t: T, sizes: IntList) raises -> IndexList[MAX_RANK]:
    var shape = IndexList[MAX_RANK](1)
    var pad = MAX_RANK - len(sizes)
    var known = 1
    var infer = -1
    for i in range(len(sizes)):
        if sizes[i] == -1:
            if infer >= 0:
                raise Error("only one dimension can be inferred")
            infer = i
        else:
            known *= sizes[i]
            shape[pad + i] = sizes[i]
    if infer >= 0:
        if known == 0 or t.numel % known != 0:
            raise Error("shape is invalid for input of size ", t.numel)
        shape[pad + infer] = t.numel // known
    elif known != t.numel:
        raise Error("shape is invalid for input of size ", t.numel)
    return shape


def _view_strides(
    t: T, shape: IndexList[MAX_RANK], rank: Int
) raises -> IndexList[MAX_RANK]:
    """Strides of `view(shape)` over t, per at::detail::computeStride; the
    contiguous case is the common one and handled first."""
    if t.contig:
        return contiguous_strides(shape, rank)
    # Non-contiguous: only shapes that keep every stride group intact are views.
    var out = IndexList[MAX_RANK](0)
    var tensor_d = t.rank - 1
    var view_d = rank - 1
    var chunk_base_stride = t.stride(tensor_d) if t.rank > 0 else 1
    var tensor_numel = 1
    var view_numel = 1
    while tensor_d >= 0:
        tensor_numel *= t.dim(tensor_d)
        if tensor_d == 0 or (
            t.dim(tensor_d - 1) != 1
            and t.stride(tensor_d - 1) != tensor_numel * chunk_base_stride
        ):
            while view_d >= 0 and (
                view_numel < tensor_numel
                or shape[MAX_RANK - rank + view_d] == 1
            ):
                out[MAX_RANK - rank + view_d] = view_numel * chunk_base_stride
                view_numel *= shape[MAX_RANK - rank + view_d]
                view_d -= 1
            if view_numel != tensor_numel:
                unsupported(
                    "view of a non-contiguous tensor with incompatible size and"
                    " stride; use reshape"
                )
            if tensor_d > 0:
                chunk_base_stride = t.stride(tensor_d - 1)
                tensor_numel = 1
                view_numel = 1
        tensor_d -= 1
    if view_d != -1:
        unsupported(
            "view of a non-contiguous tensor with incompatible size and stride;"
            " use reshape"
        )
    return out


# aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)
def op_view(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var sizes = IntList(args[unsafe_offset=1])
    if len(sizes) > MAX_RANK:
        raise Error("rank ", len(sizes), " exceeds the mojo device limit")
    var shape = _infer_view_shape(t, sizes)
    ret_tensor(
        rets,
        0,
        view_strided(
            t, shape, _view_strides(t, shape, len(sizes)), len(sizes), t.offset
        ),
    )


# aten::_reshape_alias(Tensor(a) self, SymInt[] size, SymInt[] stride) -> Tensor(a)
def op_reshape_alias(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var sizes = IntList(args[unsafe_offset=1])
    var strides = IntList(args[unsafe_offset=2])
    var shape = _shape_of(sizes)
    ret_tensor(
        rets,
        0,
        view_strided(
            t, shape, _strides_of(strides, len(sizes)), len(sizes), t.offset
        ),
    )


# aten::as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)
def op_as_strided(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var sizes = IntList(args[unsafe_offset=1])
    var strides = IntList(args[unsafe_offset=2])
    var offset = v_int_or(args[unsafe_offset=3], t.offset)
    var shape = _shape_of(sizes)
    ret_tensor(
        rets,
        0,
        view_strided(
            t, shape, _strides_of(strides, len(sizes)), len(sizes), offset
        ),
    )


# aten::_local_scalar_dense(Tensor self) -> Scalar
def op_local_scalar_dense(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    if t.numel != 1:
        raise Error(
            "a Tensor with ", t.numel, " elements cannot be converted to Scalar"
        )
    var buf = InlineArray[UInt8, 16](fill=0)
    read_bytes_sync(ctx_for(t.device), t.ptr, Int(buf.unsafe_ptr()), t.itemsize)
    var p = buf.unsafe_ptr()
    if t.dtype == DType.bool:
        ret_scalar_bool(rets, 0, p[] != 0)
    elif t.dtype.is_floating_point():
        var v: Float64 = 0
        if t.dtype == DType.float32:
            v = Float64(p.unsafe_bitcast[Float32]()[])
        elif t.dtype == DType.bfloat16:
            v = Float64(p.unsafe_bitcast[BFloat16]()[])
        elif t.dtype == DType.float16:
            v = Float64(p.unsafe_bitcast[Float16]()[])
        else:
            v = p.unsafe_bitcast[Float64]()[]
        ret_scalar_f64(rets, 0, v)
    else:
        var v: Int = 0
        if t.dtype == DType.int64:
            v = Int(p.unsafe_bitcast[Int64]()[])
        elif t.dtype == DType.int32:
            v = Int(p.unsafe_bitcast[Int32]()[])
        elif t.dtype == DType.int16:
            v = Int(p.unsafe_bitcast[Int16]()[])
        elif t.dtype == DType.int8:
            v = Int(p.unsafe_bitcast[Int8]()[])
        elif t.dtype == DType.uint8:
            v = Int(p[])
        elif t.dtype == DType.uint16:
            v = Int(p.unsafe_bitcast[UInt16]()[])
        elif t.dtype == DType.uint32:
            v = Int(p.unsafe_bitcast[UInt32]()[])
        else:
            v = Int(p.unsafe_bitcast[UInt64]()[])
        ret_scalar_int(rets, 0, v)


# aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
def op_fill_scalar_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    # The Scalar record, not a Float64: a bool tensor fills on nonzero truth
    # and an int64 one keeps every bit (ops_common.FillScalar).
    fill_value(t, args[unsafe_offset=1].copy())
    ret_ref(rets, 0, t)


# aten::zero_(Tensor(a!) self) -> Tensor(a!)
def op_zero_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    fill_value(t, 0.0)
    ret_ref(rets, 0, t)


# aten::record_stream(Tensor(a!) self, Stream s) -> ()
def op_record_stream(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var st = v_stream(args[unsafe_offset=1])
    var handle = t.storage_ctx()
    if handle != 0:
        record_stream(handle, st[0], st[1])


def register_core(site: Site) raises:
    impl[op_empty_memory_format, "empty.memory_format"](site)
    impl[op_empty_strided, "empty_strided"](site)
    impl[op_copy_from, "_copy_from"](site)
    impl[op_view, "view"](site)
    impl[op_view, "_unsafe_view"](site)
    impl[op_reshape_alias, "_reshape_alias"](site)
    impl[op_as_strided, "as_strided"](site)
    impl[op_local_scalar_dense, "_local_scalar_dense"](site)
    impl[op_fill_scalar_, "fill_.Scalar"](site)
    impl[op_zero_, "zero_"](site)
    impl[op_record_stream, "record_stream"](site)


@export
def tmb_op_address() abi("C") -> Int:
    """Entry of this file's one-op extension: the address of the op the
    TMB_OP define selected (registry.mojo)."""
    return op_address_of[register_core]()
