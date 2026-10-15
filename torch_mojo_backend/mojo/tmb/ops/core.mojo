"""Core aten ops: empty/empty_strided, transfers, views, scalar readback,
fills, record_stream.

Each op takes the record stack of its schema (abi.mojo) and writes its
result records. Views and the two empty factories never launch a kernel;
transfers are MAX copies; fills are memsets on contiguous memory. The
value-producing factories (arange, normal_, rand...) are tmb/ops/factories.mojo.
"""
from std.ffi import external_call
from std.utils import IndexList

from tmb.backend.abi import (
    ST_BOOL,
    ST_BFLOAT16,
    ST_INT16,
    ST_INT32,
    ST_INT64,
    ST_INT8,
    ST_UINT8,
    ST_UINT64,
    TAG_COMPLEX,
    TAG_DOUBLE,
    TAG_SCALAR_BOOL,
    TAG_SCALAR_INT,
    TAG_INT,
    TAG_BOOL,
    complex_of,
    TAG_SCALAR_DOUBLE,
    dtype_name,
    ST_FLOAT16,
    ST_FLOAT32,
    ST_FLOAT64,
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
    v_tensor_c,
    v_opt_tensor,
    v_tensor_list,
    v_complex,
    is_complex_stype,
    ret_scalar_complex,
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
    none_arg,
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
    Owned,
    Results,
    f64_bits,
    TAG_NONE,
    TAG_TENSOR,
    TAG_TENSOR_REF,
)
from tmb.backend.device import (
    copy_d2d,
    copy_peer,
    copy_from_host,
    copy_to_host,
    wait_for_host_read,
    ctx_for,
    current_device,
    current_stream,
    drain_copy,
    dev,
    read_bytes_sync,
    record_stream,
)
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.kernels.common.float8 import fp8_bits_to_f64, is_float8
from tmb.ops.common import (
    real_fill_value,
    is_float_stype,
    check_real_scalar_fits,
    check_scalar_fits,
    assert_no_internal_overlap,
    assert_no_partial_overlap,
    cx_unary_into,
    fill_complex,
    cast_complex,
    cast_into,
    cast_to,
    conj_copy,
    contiguous,
    copy_strided_into,
    fill_value,
    is_cast_dtype_on,
    is_cast_pair_on,
)
from tmb.backend.registry import Site, impl


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


# aten::scalar_tensor(Scalar s, *, ScalarType? dtype, Layout? layout,
#   Device? device, bool? pin_memory) -> Tensor
def op_scalar_tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """A 0-d tensor holding `s`. Its dtype defaults to the default float
    dtype whatever the Scalar's kind (CUDA's scalar_tensor). The value
    converts as at::detail::scalar_tensor_static does -- what Python item
    assignment (`x[i] = v`, `x[mask] = v`, `x[:] = v`) builds its value
    with on the host for CUDA, and through this op on this device:
    - half / bfloat16: value.to<double>() (a complex with an imaginary part
      raises), then an unchecked narrowing (overflow is inf);
    - complex: the checked complex conversion;
    - bool: the value's truth;
    - every other dtype: value.to<scalar_t>(), checked (a complex with no
      imaginary part converts as its real part does; uint64 takes doubles
      up to 2**64 as the host static_cast does -- `real_fill_value`).
    An explicit torch.scalar_tensor(70000., dtype=half) is checked on CUDA
    (a device fill_) but inf here: both entry points reach this one op."""
    var v = args[unsafe_offset=0].copy()
    var stype = v_dtype_or(args[unsafe_offset=1], default_dtype())
    var device = _target_device(args[unsafe_offset=3])
    var c = v_complex(v)
    if is_complex_stype(stype):
        check_scalar_fits(stype, c[0], c[1])
    elif stype == ST_FLOAT16 or stype == ST_BFLOAT16:
        check_real_scalar_fits(ST_FLOAT64, v)  # value.to<double>()
    elif stype != ST_BOOL:
        check_real_scalar_fits(stype, v)
    var out = own(new_tensor(IndexList[MAX_RANK](1), 0, stype, device))
    if is_complex_stype(stype):
        fill_complex(out.t, v)
    elif stype == ST_BOOL:
        var truth = Int64(1) if c[0] != 0.0 or c[1] != 0.0 else Int64(0)
        fill_value(out.t, Value(TAG_SCALAR_BOOL, 0, truth, 0))
    elif stype == ST_FLOAT16 or stype == ST_BFLOAT16:
        # the double, narrowed unchecked (through float, as c10 does)
        var dbl = Float64(v.a) if (
            v.tag == TAG_SCALAR_INT or v.tag == TAG_INT
        ) else c[0]
        fill_value(out.t, dbl)
    else:
        fill_value(out.t, real_fill_value(stype, v))
    ret_tensor(rets, 0, out.take())


# aten::empty.memory_format(SymInt[] size, *, ScalarType? dtype, Layout? layout,
#   Device? device, bool? pin_memory, MemoryFormat? memory_format) -> Tensor
def op_empty_memory_format(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    if v_bool_or(args[unsafe_offset=4], False):
        raise Error("Only dense CPU tensors can be pinned")
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


# aten::_make_dep_token(*, ScalarType? dtype=None, Layout? layout=None,
#   Device? device=None, bool? pin_memory=None,
#   MemoryFormat? memory_format=None) -> Tensor
def op_make_dep_token(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """ATen's CPU kernel (the only one upstream has): `at::empty({}, ...)`,
    a 0-d placeholder that orders `_functional_assert_async` calls."""
    if v_bool_or(args[unsafe_offset=3], False):
        raise Error("Only dense CPU tensors can be pinned")
    var stype = v_dtype_or(args[unsafe_offset=0], default_dtype())
    var device = _target_device(args[unsafe_offset=2])
    var shape = IndexList[MAX_RANK](1)
    var mf = v_memory_format_or(args[unsafe_offset=4], MEMORY_FORMAT_CONTIGUOUS)
    ret_tensor(
        rets,
        0,
        new_strided(
            shape, strides_for_memory_format(shape, 0, mf), 0, stype, device
        ),
    )


# aten::empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype,
#   Layout? layout, Device? device, bool? pin_memory) -> Tensor
def op_empty_strided(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    if v_bool_or(args[unsafe_offset=5], False):
        raise Error("Only dense CPU tensors can be pinned")
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


def record_tensor_stream(t: T) raises:
    var handle = t.storage_ctx()
    if handle != 0:
        record_stream(handle, t.device, current_stream(t.device))


def copy_between_devices(dst: T, src: T) raises:
    """Requires contiguous buffers of equal dtype and size."""
    record_tensor_stream(src)
    record_tensor_stream(dst)
    var nbytes = src.numel * src.itemsize
    try:
        if copy_peer(dst.device, dst.ptr, src.device, src.ptr, nbytes):
            return
    except e:
        _ = drain_copy(dst.device, src.device)
        raise e
    var host = own(cpu_empty(src.shape, src.rank, src.stype))
    try:
        copy_to_host(ctx_for(src.device), src.ptr, host.t.ptr, nbytes)
        copy_from_host(
            dst.device, ctx_for(dst.device), dst.ptr, host.t.ptr, nbytes
        )
    except e:
        if not drain_copy(dst.device, src.device):
            _ = retain(host.t)
        _ = host^
        raise e
    _ = host^


def _device_copy(dst: T, src: T) raises:
    """Shapes may differ; op_copy_from checks equal element counts."""
    if src.device != dst.device:
        record_tensor_stream(src)
        record_tensor_stream(dst)
        var dense = own_if_new(contiguous(src), src)
        if dst.contig and dst.stype == src.stype:
            copy_between_devices(dst, dense.t)
        else:
            var moved = own(
                new_tensor(src.shape, src.rank, src.stype, dst.device)
            )
            copy_between_devices(moved.t, dense.t)
            _device_copy(dst, moved.t)
            _ = moved^
        _ = dense^
        return
    if src.stype != dst.stype:
        if (
            src.contig
            and dst.contig
            and not src.is_complex()
            and not dst.is_complex()
            and is_cast_pair_on(src.dtype, dst.dtype, src)
            and (
                src.ptr + src.numel * src.itemsize <= dst.ptr
                or dst.ptr + dst.numel * dst.itemsize <= src.ptr
            )
        ):
            # The caller already owns the destination. Convert into it
            # directly when no source bytes can be overwritten before read.
            cast_into(dst, src)
            return
        var dense = own_if_new(contiguous(src), src)
        var tmp = own_if_new(cast_for_copy(dense.t, dst.stype), dense.t)
        if dst.contig:
            copy_d2d(
                ctx_for(dst.device),
                dst.ptr,
                tmp.t.ptr,
                dst.numel * dst.itemsize,
            )
            _ = tmp^  # alive past the launch
        else:
            var shaped = own(_viewed_as(tmp.t, dst))
            _ = tmp^  # alive past the view construction
            copy_strided_into(dst, shaped.t)
            _ = shaped^  # alive past the launch
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
        _ = dense^  # alive past the view construction
        copy_strided_into(dst, shaped.t)
        _ = shaped^  # alive past the launch


def _host_copy(dst: T, src: T) raises:
    """dst = src for two host tensors, through torch's CPU copy_ (layouts and
    dtypes handled by ATen)."""
    var args = List[Value]()
    args.append(tensor_arg(dst))
    args.append(tensor_arg(src))
    args.append(bool_arg(False))
    _ = call_op("aten::copy_", "", args^, 1)  # Results releases copy_'s handle


def _copy_broadcast_source(src: T, dst: T) raises -> T:
    # _copy_from also accepts equal-element-count reshapes. Unequal counts
    # require normal trailing-dimension broadcasting, validated before writes.
    if src.numel == dst.numel:
        return src.copy()
    if src.rank > dst.rank:
        raise Error("_copy_from: source shape cannot broadcast to destination")
    var strides = IndexList[MAX_RANK](0)
    for i in range(dst.rank):
        var j = i - (dst.rank - src.rank)
        if j < 0:
            continue
        if src.dim(j) == dst.dim(i):
            strides[MAX_RANK - dst.rank + i] = src.stride(j)
        elif src.dim(j) != 1:
            raise Error(
                "_copy_from: source shape cannot broadcast to destination"
            )
    return view_strided(src, dst.shape, strides, dst.rank, src.offset)


def cast_for_copy(src: T, stype: Int32, saturate: Bool = True) raises -> T:
    """`src` converted to `stype`. `saturate` is copy_'s float -> e4m3fn
    rule; an out= / in-place store of a TensorIterator op converts in its
    kernel (c10's static_cast) and passes False."""
    if src.is_complex() or is_complex_stype(stype):
        # `saturate` cannot apply: a complex source never saturates (stock
        # CUDA's SATFINITE path is keyed on a float / half / bfloat16 source)
        # and no complex dtype is float8.
        return cast_complex(src, stype)
    if is_cast_pair_on(src.dtype, max_dtype(stype), src):
        return cast_to(src, stype, saturate)
    # CPU torch preserves integer precision for pairs outside CastSpec.
    var host_src = own(cpu_empty(src.shape, src.rank, src.stype))
    copy_to_host(
        ctx_for(src.device), src.ptr, host_src.t.ptr, src.numel * src.itemsize
    )
    var host_dst = own(cpu_empty(src.shape, src.rank, stype))
    _host_copy(host_dst.t, host_src.t)
    _ = host_src^
    var out = own(new_like_dtype(src, stype))
    copy_from_host(
        src.device,
        ctx_for(src.device),
        out.t.ptr,
        host_dst.t.ptr,
        out.t.numel * out.t.itemsize,
    )
    _ = host_dst^
    return out.take()


def _copy_through_toggled(
    op: StaticString, input: T, dst: T, non_blocking: Bool
) raises:
    """`dst.copy_(input)` for a `dst` with a math bit: written through its
    view with the bit toggled off (`_conj` / `_neg_view`), from `input` with
    the same bit toggled (conj of a real input is the input itself)."""
    var plain = call_op(op, "", [tensor_arg(dst)], 1)
    var flipped: Results
    if op == "aten::_conj" and not input.is_complex():
        flipped = call_op("aten::alias", "", [tensor_arg(input)], 1)
    else:
        flipped = call_op(op, "", [tensor_arg(input)], 1)
    _ = call_op(
        "aten::copy_",
        "",
        [plain[0].copy(), flipped[0].copy(), bool_arg(non_blocking)],
        1,
    )
    _ = plain^  # their handles are the copy's arguments
    _ = flipped^


def _resolved_source(input: T) raises -> Owned:
    """A fresh tensor holding `input`'s logical values: its math bits (conj
    on a complex tensor, neg on any) applied to its raw storage."""
    var conj = input.is_complex() and input.is_conj()
    var neg = input.is_neg()
    if not input.on_mojo():
        var name = "aten::resolve_conj" if conj else "aten::resolve_neg"
        var r = call_op(name, "", [tensor_arg(input)], 1)
        var first = own(r.take_tensor(0))
        if not (conj and neg):
            return first^
        var r2 = call_op("aten::resolve_neg", "", [tensor_arg(first.t)], 1)
        var both = own(r2.take_tensor(0))
        _ = first^  # alive through the second call
        return both^
    var fresh: Owned
    if conj:
        fresh = own(conj_copy(input))
    else:
        fresh = own(new_like(input))
        copy_strided_into(fresh.t, input)
    if neg:
        if input.is_complex():
            cx_unary_into("neg", fresh.t, fresh.t)
        else:
            _ = call_op("aten::neg_", "", [tensor_arg(fresh.t)], 1)
    return fresh^


def _same_strides(a: T, b: T) -> Bool:
    for d in range(a.rank):
        if a.stride(d) != b.stride(d):
            return False
    return True


def own_borrowed(t: T) -> Owned:
    """An `Owned` over a borrowed tensor: never released by it."""
    var o = own(t.copy())
    o.live = False
    return o^


def _check_copy_broadcast(src: T, dst: T) raises:
    """copy_'s TensorIterator shape check (infer_size, scanning from the
    last dim): src must broadcast to dst."""
    if src.numel == dst.numel and src.same_shape(dst):
        return
    var n = max(src.rank, dst.rank)
    for k in range(n):
        var i = n - 1 - k
        var ia = i - (n - dst.rank)
        var ib = i - (n - src.rank)
        var a = dst.dim(ia) if ia >= 0 else 1
        var b = src.dim(ib) if ib >= 0 else 1
        if a != b and a != 1 and b != 1:
            raise Error(
                "The size of tensor a (",
                a,
                ") must match the size of tensor b (",
                b,
                ") at non-singleton dimension ",
                i,
            )
    if src.rank > dst.rank or not _broadcasts_to(src, dst):
        raise Error("_copy_from: source shape cannot broadcast to destination")


def _broadcasts_to(src: T, dst: T) -> Bool:
    for i in range(src.rank):
        var j = i + (dst.rank - src.rank)
        if src.dim(i) != 1 and src.dim(i) != dst.dim(j):
            return False
    return True


def _st_token(st: Int32) -> String:
    """c10::toString(ScalarType): what AT_DISPATCH's not-implemented
    message names."""
    if st == ST_FLOAT32:
        return "Float"
    if st == ST_FLOAT64:
        return "Double"
    if st == ST_FLOAT16:
        return "Half"
    if st == ST_BFLOAT16:
        return "BFloat16"
    if st == ST_INT64:
        return "Long"
    if st == ST_INT32:
        return "Int"
    if st == ST_INT16:
        return "Short"
    if st == ST_INT8:
        return "Char"
    if st == ST_UINT8:
        return "Byte"
    if st == ST_BOOL:
        return "Bool"
    return dtype_name(st)


def _toggled_view(op: StaticString, t: T) raises -> Owned:
    """`aten::_conj` / `aten::_neg_view` of t: the same data, that bit
    toggled."""
    var r = call_op(String(op), "", [tensor_arg(t)], 1)
    return own(r.take_tensor(0))


def _copy_bits(input: T, dst: T, non_blocking: Bool) raises -> Bool:
    """`dst.copy_(input)` when either carries a lazy math bit (conj on a
    complex tensor, neg on any): the stored values end up the logical ones,
    as copy_impl makes them. True when it handled the copy. A bit on `dst`
    is written through the toggled view; a bit on `input` is resolved into a
    fresh tensor. Each step re-enters copy_ with one bit fewer. (The
    Conjugate and Negative keys let `_copy_from` through with its bits for
    mojo tensors: shim_views.cpp.)"""
    var bits = (
        (dst.is_complex() and dst.is_conj())
        or dst.is_neg()
        or (input.is_complex() and input.is_conj())
        or input.is_neg()
    )
    if not bits:
        return False
    _check_copy_broadcast(input, dst)
    if dst.numel == 0:
        return True  # copy_impl returns before choosing a kernel
    var s_conj = input.is_complex() and input.is_conj()
    var s_neg = input.is_neg()
    var d_conj = dst.is_complex() and dst.is_conj()
    var d_neg = dst.is_neg()
    if (s_neg and d_neg) or (s_conj and d_conj):
        # a bit on both sides cancels: copy between the views without it
        var src_v = own_borrowed(input)
        var dst_v = own_borrowed(dst)
        if s_neg and d_neg:
            src_v = _toggled_view("aten::_neg_view", src_v.t)
            dst_v = _toggled_view("aten::_neg_view", dst_v.t)
        if s_conj and d_conj:
            src_v = _toggled_view("aten::_conj", src_v.t)
            dst_v = _toggled_view("aten::_conj", dst_v.t)
        _ = call_op(
            "aten::copy_",
            "",
            [tensor_arg(dst_v.t), tensor_arg(src_v.t), bool_arg(non_blocking)],
            1,
        )
        _ = src_v^
        _ = dst_v^
        return True
    if input.on_mojo() and dst.on_mojo():
        # copy_device_to_device's kernels: neg into bool, and neg + conj
        # from a real source, have no CUDA instantiation (stock CUDA raises
        # these; matched so code behaves the same on both)
        var neg_differs = input.is_neg() != dst.is_neg()
        var conj_differs = (input.is_complex() and input.is_conj()) != (
            dst.is_complex() and dst.is_conj()
        )
        # (neg alone: a neg kernel on dst's dtype; neg and conj: the fused
        # neg_conj kernel on the source's)
        if neg_differs and not conj_differs and dst.stype == ST_BOOL:
            unsupported("\"neg_cuda\" not implemented for 'Bool'")
        if neg_differs and conj_differs and not input.is_complex():
            unsupported(
                '"neg_conj_cuda" not implemented for \''
                + _st_token(input.stype)
                + "'"
            )
    if (s_conj or s_neg) and (d_conj or d_neg):
        # a different bit on each side: not emulated (declined cleanly)
        unsupported(
            "a copy whose source and destination carry different math bits"
        )
    if (
        s_conj
        and s_neg
        and not dst.is_complex()
        and not is_float_stype(dst.stype)
        and dst.stype != ST_BOOL
    ):
        # CUDA runs the fused neg+conj in the source dtype, then converts
        # out-of-range values the device's way: not emulated
        unsupported(
            "a neg + conj view copied into an integer tensor (resolve it first)"
        )
    if input.stype != dst.stype or not (
        input.on_mojo() and dst.on_mojo() and input.device == dst.device
    ):
        # convert first, then apply the math bits (copy_'s semantics: the
        # destination's logical values are the converted source's), so a
        # negation never happens in the narrower or unsigned source dtype:
        # the source's raw values (its bits toggled off, views) converted
        # into a fresh tensor of dst's dtype, which then takes the bits
        if (
            input.on_mojo()
            and dst.on_cpu()
            and (s_conj or s_neg)
            and non_blocking
            and input.stype != dst.stype
        ):
            # Stock CUDA's non-blocking device -> host copy with a math bit
            # and a dtype change converts on the GPU and drops the bit
            # (measured: _neg_view(int8 [5]) into a host int64 gives [5]); a
            # wrong value is not replicated: declined, before any write.
            unsupported(
                "a non-blocking device-to-host copy of a conj / neg view with"
                " a dtype change (copy blocking, or resolve the view first)"
            )
        if input.on_mojo() and dst.on_cpu() and (s_conj or s_neg):
            # device -> host: CUDA resolves the bits in the SOURCE dtype
            # (uint8 -255 wraps to 1), then converts on the host
            var rawd = own_borrowed(input)
            if s_conj:
                rawd = _toggled_view("aten::_conj", rawd.t)
            if s_neg:
                rawd = _toggled_view("aten::_neg_view", rawd.t)
            var host = own(cpu_empty(rawd.t.shape, rawd.t.rank, input.stype))
            _ = call_op(
                "aten::copy_",
                "",
                [tensor_arg(host.t), tensor_arg(rawd.t), bool_arg(False)],
                1,
            )
            _ = rawd^
            if s_conj:
                _ = call_op("aten::conj_physical_", "", [tensor_arg(host.t)], 1)
            if s_neg:
                _ = call_op("aten::neg_", "", [tensor_arg(host.t)], 1)
            _ = call_op(
                "aten::copy_",
                "",
                [tensor_arg(dst), tensor_arg(host.t), bool_arg(False)],
                1,
            )
            _ = host^
            return True
        var raw = own_borrowed(input)
        if s_conj:
            raw = _toggled_view("aten::_conj", raw.t)
        if s_neg:
            raw = _toggled_view("aten::_neg_view", raw.t)
        var conv = own(
            new_tensor(raw.t.shape, raw.t.rank, dst.stype, dst.device)
        ) if dst.on_mojo() else own(
            cpu_empty(raw.t.shape, raw.t.rank, dst.stype)
        )
        _ = call_op(
            "aten::copy_",
            "",
            [tensor_arg(conv.t), tensor_arg(raw.t), bool_arg(non_blocking)],
            1,
        )
        _ = raw^
        if s_conj and conv.t.is_complex():
            if conv.t.on_mojo():
                cx_unary_into("conj", conv.t, conv.t)
            else:
                _ = call_op("aten::conj_physical_", "", [tensor_arg(conv.t)], 1)
        if s_neg:
            if conv.t.on_mojo() and conv.t.is_complex():
                cx_unary_into("neg", conv.t, conv.t)
            elif conv.t.stype != ST_BOOL:  # -x is as true as x
                _ = call_op("aten::neg_", "", [tensor_arg(conv.t)], 1)
        _ = call_op(
            "aten::copy_",
            "",
            [tensor_arg(dst), tensor_arg(conv.t), bool_arg(non_blocking)],
            1,
        )
        _ = conv^
        return True
    if dst.is_complex() and dst.is_conj():
        _copy_through_toggled("aten::_conj", input, dst, non_blocking)
        return True
    if dst.is_neg():
        _copy_through_toggled("aten::_neg_view", input, dst, non_blocking)
        return True
    if not (input.is_complex() and input.is_conj()) and not input.is_neg():
        return False
    var resolved = _resolved_source(input)
    _ = call_op(
        "aten::copy_",
        "",
        [tensor_arg(dst), tensor_arg(resolved.t), bool_arg(non_blocking)],
        1,
    )
    _ = resolved^
    return True


# aten::_copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> Tensor
def op_copy_from(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var input = v_tensor_c(args[unsafe_offset=0])
    var dst = v_tensor_c(args[unsafe_offset=1])
    var non_blocking = v_bool_or(args[unsafe_offset=2], False)
    # copy_impl: nothing to do for the same tensor, or for a view of the
    # very same data (dtype and math bits included)
    if dst.impl() == input.impl():
        ret_ref(rets, 0, dst)
        return
    if (
        dst.on_mojo()
        and input.on_mojo()
        and dst.device == input.device
        and dst.ptr == input.ptr
        and dst.stype == input.stype
        and dst.same_shape(input)
        and _same_strides(dst, input)
        and dst.is_neg() == input.is_neg()
        and (not dst.is_complex() or dst.is_conj() == input.is_conj())
    ):
        ret_ref(rets, 0, dst)
        return
    # its TensorIterator's memory-overlap checks, on the operands as given
    if dst.on_mojo():
        assert_no_internal_overlap(dst)
        if input.on_mojo() and input.device == dst.device:
            assert_no_partial_overlap(dst, input)
    if _copy_bits(input, dst, non_blocking):
        ret_ref(rets, 0, dst)
        return
    var expanded = own_if_new(_copy_broadcast_source(input, dst), input)
    var src = expanded.t.copy()
    if dst.numel == 0:
        ret_ref(rets, 0, dst)
        return
    if dst.on_mojo() and src.on_mojo():
        _device_copy(dst, src)
    elif dst.on_mojo():
        if not src.on_cpu():
            unsupported(
                "copy from a device "
                + String(src.device_type)
                + " tensor to the mojo device"
            )
        if (
            non_blocking
            and src.stype != dst.stype
            and is_cast_pair_on(src.dtype, dst.dtype, dst)
        ):
            # Copy.cu: a non-blocking transfer converts on the GPU (where
            # float -> e4m3fn saturates on sm_89+), a blocking one on the CPU.
            var packed = own(cpu_empty(src.shape, src.rank, src.stype))
            wait_for_host_read(ctx_for(dst.device), src.ptr)
            _host_copy(packed.t, src)
            var raw = own(
                new_tensor(src.shape, src.rank, src.stype, dst.device)
            )
            copy_from_host(
                dst.device,
                ctx_for(dst.device),
                raw.t.ptr,
                packed.t.ptr,
                src.numel * src.itemsize,
            )
            _ = packed^
            _device_copy(dst, raw.t)
            _ = raw^  # alive past the launch
            _ = expanded^
            ret_ref(rets, 0, dst)
            return
        # host -> device: a dense host copy in dst's dtype (torch's CPU copy_
        # does the layout / dtype work), one H2D, then a device relayout if needed
        var host = own_if_new(
            src.copy() if src.contig
            and src.stype
            == dst.stype else cpu_empty(dst.shape, dst.rank, dst.stype),
            src,
        )
        if host.t.h != src.h:
            wait_for_host_read(ctx_for(dst.device), src.ptr)
            _host_copy(host.t, src)
        var nbytes = dst.numel * dst.itemsize
        # `host` is dense in dst's dtype, so a contiguous dst takes the bytes
        # whatever the two logical shapes are.
        if dst.contig:
            copy_from_host(
                dst.device,
                ctx_for(dst.device),
                dst.ptr,
                host.t.ptr,
                nbytes,
                non_blocking,
            )
            _ = host^  # alive past the launch
        else:
            var tmp = own(new_like(dst))
            copy_from_host(
                dst.device,
                ctx_for(dst.device),
                tmp.t.ptr,
                host.t.ptr,
                nbytes,
                non_blocking,
            )
            _ = host^  # alive past the launch
            copy_strided_into(dst, tmp.t)
            _ = tmp^  # alive past the launch
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
        if (
            non_blocking
            and src.stype != dst.stype
            and is_cast_pair_on(src.dtype, dst.dtype, src)
        ):
            # A non-blocking download converts on the GPU (Copy.cu).
            var conv = own(cast_to(dense.t, dst.stype))
            _ = dense^
            var landed = own(cpu_empty(src.shape, src.rank, dst.stype))
            copy_to_host(
                ctx_for(src.device),
                conv.t.ptr,
                landed.t.ptr,
                conv.t.numel * conv.t.itemsize,
            )
            _ = conv^
            _host_copy(dst, landed.t)
            _ = landed^
            _ = expanded^
            ret_ref(rets, 0, dst)
            return
        var nbytes = src.numel * src.itemsize
        # Both dense in the same dtype: the bytes land in the right order
        # whatever the two logical shapes are.
        if dst.contig and dst.stype == src.stype:
            copy_to_host(
                ctx_for(src.device), dense.t.ptr, dst.ptr, nbytes, non_blocking
            )
            _ = dense^  # alive past the launch
        else:
            var host = own(cpu_empty(src.shape, src.rank, src.stype))
            copy_to_host(ctx_for(src.device), dense.t.ptr, host.t.ptr, nbytes)
            _ = dense^  # alive past the launch
            _host_copy(dst, host.t)
            _ = host^  # alive past the copy
    else:
        raise Error("_copy_from: neither tensor is on the mojo device")
    _ = expanded^
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
# Unregistered: ATen's own kernel serves `view` / `_unsafe_view` /
# `_reshape_alias` / `as_strided` here (see register_core). Kept as the Mojo
# route for a device or a torch version that needs one.
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
    var t = v_tensor_c(args[unsafe_offset=0])
    if t.numel != 1:
        raise Error(
            "a Tensor with ", t.numel, " elements cannot be converted to Scalar"
        )
    var buf = Array[UInt8, 16](fill=0)
    read_bytes_sync(ctx_for(t.device), t.ptr, Int(buf.unsafe_ptr()), t.itemsize)
    var p = buf.unsafe_ptr()
    if t.is_complex():
        var re: Float64
        var im: Float64
        if t.dtype == DType.float32:
            re = Float64(p.unsafe_bitcast[Float32]()[])
            im = Float64(p.unsafe_bitcast[Float32]()[unsafe_offset=1])
        else:
            re = p.unsafe_bitcast[Float64]()[]
            im = p.unsafe_bitcast[Float64]()[unsafe_offset=1]
        if t.is_conj():
            im = -im
        ret_scalar_complex(rets, 0, re, im)
        return
    if t.dtype == DType.bool:
        ret_scalar_bool(rets, 0, p[] != 0)
    elif is_float8(t.dtype):
        ret_scalar_f64(rets, 0, fp8_bits_to_f64(t.dtype, p[]))
    elif t.dtype.is_floating_point():
        var v: Float64
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
        var v: Int
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
    var t = v_tensor_c(args[unsafe_offset=0])
    if t.is_complex():
        var c = v_complex(args[unsafe_offset=1])
        check_scalar_fits(t.stype, c[0], c[1])
        fill_complex(t, args[unsafe_offset=1].copy())
        ret_ref(rets, 0, t)
        return
    # The Scalar record, not a Float64: a bool tensor fills on nonzero truth
    # and an int64 one keeps every bit (ops_common.FillScalar). Converted
    # checked first, as CUDA's fill_ (value.to<scalar_t>()).
    var v = args[unsafe_offset=1].copy()
    check_real_scalar_fits(t.stype, v)
    v = real_fill_value(t.stype, v)
    if v.tag == TAG_COMPLEX:
        # a complex Scalar into bool: its truth reads either part
        var c = v_complex(v)
        var truth = Int64(1) if c[0] != 0.0 or c[1] != 0.0 else Int64(0)
        v = Value(TAG_SCALAR_BOOL, 0, truth, 0)
    fill_value(t, v)
    ret_ref(rets, 0, t)


# aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)
# aten::zero_(Tensor(a!) self) -> Tensor(a!)
def op_zero_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor_c(args[unsafe_offset=0])
    if t.is_complex() and not t.contig:
        fill_value(t.pairs(), 0.0)
    else:
        fill_value(t, 0.0)
    ret_ref(rets, 0, t)


# aten::record_stream(Tensor(a!) self, Stream s) -> ()
def op_record_stream(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor_c(args[unsafe_offset=0])
    var st = v_stream(args[unsafe_offset=1])
    var handle = t.storage_ctx()
    if handle != 0:
        record_stream(handle, st[0], st[1])


def _aten_view(site: Site, name: StaticString) raises:
    """Register ATen's own kernel for one metadata-only view op
    (shim_views.cpp): storage sharing plus new sizes/strides, no device code,
    and unboxed -- which skips boxing a `SymInt[]` into a heap c10::List per
    call and the round trip into Mojo."""
    var s = String(name)
    check(
        external_call["tmb_library_impl_aten_view", Int32](
            site.lib, s.as_c_string_span().ptr()
        ),
        "registering ATen's view kernel",
    )


# aten::_copy_from_and_resize(Tensor self, Tensor dst) -> Tensor
def op_copy_from_and_resize(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """What a CPU fallback uses to write a result back (MPS's kernel):
    `dst.resize_as_(self)`, then `dst.copy_(self)`."""
    var src = v_tensor_c(args[unsafe_offset=0])
    var dst = v_tensor_c(args[unsafe_offset=1])
    _ = call_op(
        "aten::resize_as_",
        "",
        [tensor_arg(dst), tensor_arg(src), none_arg()],
        1,
    )
    _ = call_op(
        "aten::copy_",
        "",
        [tensor_arg(dst), tensor_arg(src), bool_arg(False)],
        1,
    )
    ret_ref(rets, 0, dst)


def register_core(site: Site) raises:
    impl[op_empty_memory_format, "empty.memory_format"](site)
    impl[op_empty_strided, "empty_strided"](site)
    impl[op_copy_from, "_copy_from"](site)
    impl[op_copy_from_and_resize, "_copy_from_and_resize"](site)
    impl[op_make_dep_token, "_make_dep_token"](site)
    # The four metadata-only views are ATen's own kernels, registered through
    # the shim (`_unsafe_view` is CompositeExplicitAutograd upstream and would
    # work unregistered, but the test suite counts native calls per op).
    _aten_view(site, "view")
    _aten_view(site, "_unsafe_view")
    _aten_view(site, "_reshape_alias")
    _aten_view(site, "as_strided")
    _aten_view(site, "view_as_real")
    _aten_view(site, "view_as_complex")
    impl[op_local_scalar_dense, "_local_scalar_dense"](site)
    impl[op_fill_scalar_, "fill_.Scalar"](site)
    impl[op_zero_, "zero_"](site)
    impl[op_scalar_tensor, "scalar_tensor"](site)
    impl[op_record_stream, "record_stream"](site)
