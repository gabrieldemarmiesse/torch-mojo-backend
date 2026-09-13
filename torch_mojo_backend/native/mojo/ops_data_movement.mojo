"""aten ops: data_movement group (see docs/native_backend.md).

clone / _to_copy / cat / stack / repeat / tril / triu / select_scatter /
scatter.src / scatter.value / index.Tensor / nonzero / set_.source_Tensor /
empty_permuted -- ported from eager_kernels/aten_fast.py's
fast_aten_cat/stack/repeat/tril/triu/select_scatter/scatter_src/
scatter_value/index/nonzero/clone, mojo_device/aten_ops/inplace.py's
set_.source_Tensor, factories.py's empty_permuted and transfer.py's
_to_copy. Kernel families: data_movement_ops (CatN, NarrowCopyDst, TileCopy,
RepeatTiled, TriangularCopy, GatherRows, ScatterDim, PermuteCopy, CastSpec).

`nonzero` and the boolean-mask branch of `index.Tensor` are data-dependent
(the output shape depends on tensor CONTENTS, not just metadata) and have no
kernel route: they round-trip through the host exactly as the old eager path
did, via `cpu_empty` + `copy_to_host` + a plain host loop + `copy_from_host`.
"""
from std.ffi import external_call
from std.memory import unsafe_memcpy
from std.utils import IndexList

from abi import (
    DEVICE_TYPE_CPU,
    DEVICE_TYPE_PRIVATEUSE1,
    MEMORY_FORMAT_PRESERVE,
    ST_INT32,
    ST_INT64,
    IntList,
    Owned,
    T,
    Value,
    Values,
    check,
    contiguous_strides,
    cpu_empty,
    default_dtype,
    dense_strides_like,
    dtype_code,
    dtype_itemsize,
    is_dense,
    max_dtype,
    new_like,
    new_like_dtype,
    new_strided,
    new_tensor,
    own,
    own_if_new,
    release,
    ret_owned,
    ret_ref,
    set_sizes_strides,
    strides_equal,
    strides_for_memory_format,
    unsupported,
    v_device_index,
    v_device_type,
    v_dtype_or,
    v_f64,
    v_int,
    v_int_or,
    v_memory_format_or,
    v_opt_tensor_list_present,
    v_tensor,
    v_tensor_list,
    view_strided,
)
from device import (
    copy_d2d,
    copy_from_host,
    copy_to_host,
    ctx_for,
    ctx_ptr,
    current_device,
)
from kernels import KernelCall
from op_utils import MAX_RANK
from ops_common import (
    cast_into,
    fill_value,
    cast_to,
    contiguous,
    copy_strided_into,
    release_if_new,
    resize_out,
)
from registry import Site, impl, op_address_of

# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _resolve_device(v: Value) -> Int:
    """Device? argument of a factory: its index, else the current device."""
    var i = v_device_index(v)
    return i if i >= 0 else current_device()


def _shape_from_values(values: List[Int]) raises -> IndexList[MAX_RANK]:
    var n = len(values)
    if n > MAX_RANK:
        raise Error(
            "tensor rank ", n, " exceeds the mojo device limit of ", MAX_RANK
        )
    var shape = IndexList[MAX_RANK](1)
    var pad = MAX_RANK - n
    for i in range(n):
        if values[i] < 0:
            raise Error("negative dimension ", values[i])
        shape[pad + i] = values[i]
    return shape


def _pad8_list(values: List[Int], fill: Int) -> List[Int]:
    var pad = MAX_RANK - len(values)
    var out = List[Int](capacity=MAX_RANK)
    for _ in range(pad):
        out.append(fill)
    for v in values:
        out.append(v)
    return out^


def _row_major(shape: List[Int]) -> List[Int]:
    var n = len(shape)
    var strides = List[Int](capacity=n)
    for _ in range(n):
        strides.append(0)
    var acc = 1
    for k in range(n):
        var i = n - 1 - k
        strides[i] = acc
        acc *= shape[i]
    return strides^


def _is_cast_dtype(dt: DType) -> Bool:
    """The dtypes the fast CastSpec kernel supports on either end (mirrors
    data_movement_ops.mojo's `CAST_DTYPES`)."""
    return (
        dt == DType.float32
        or dt == DType.float16
        or dt == DType.bfloat16
        or dt == DType.int64
        or dt == DType.int32
        or dt == DType.uint8
        or dt == DType.bool
    )


def _is_scatter_dtype(dt: DType) -> Bool:
    """Mirrors data_movement_ops.mojo's `SCATTER_DTYPES` (no uint16/32/64)."""
    return (
        dt == DType.float16
        or dt == DType.bfloat16
        or dt == DType.float32
        or dt == DType.float64
        or dt == DType.int8
        or dt == DType.int16
        or dt == DType.int32
        or dt == DType.int64
        or dt == DType.uint8
        or dt == DType.bool
    )


def _read_f64_at(ptr: Int, i: Int, dt: DType) raises -> Float64:
    """Read element `i` of a host buffer of dtype `dt` as a Float64. Exact
    for every dtype except int64/uint64 magnitudes beyond 2**53 (the same
    bridge `_local_scalar_dense`'s scalar readback already uses); `!= 0.0` on
    the result is an exact nonzero test for every dtype regardless, since a
    nonzero value of any of these types never rounds to exactly 0.0."""
    if dt == DType.float32:
        return Float64(
            Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=ptr)[
                unsafe_offset=i
            ]
        )
    if dt == DType.bfloat16:
        return Float64(
            Pointer[BFloat16, MutUntrackedOrigin](unsafe_from_address=ptr)[
                unsafe_offset=i
            ]
        )
    if dt == DType.float16:
        return Float64(
            Pointer[Float16, MutUntrackedOrigin](unsafe_from_address=ptr)[
                unsafe_offset=i
            ]
        )
    if dt == DType.float64:
        return Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=ptr)[
            unsafe_offset=i
        ]
    if dt == DType.int64:
        return Float64(
            Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=ptr)[
                unsafe_offset=i
            ]
        )
    if dt == DType.int32:
        return Float64(
            Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=ptr)[
                unsafe_offset=i
            ]
        )
    if dt == DType.int16:
        return Float64(
            Pointer[Int16, MutUntrackedOrigin](unsafe_from_address=ptr)[
                unsafe_offset=i
            ]
        )
    if dt == DType.int8:
        return Float64(
            Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=ptr)[
                unsafe_offset=i
            ]
        )
    if dt == DType.uint8:
        return Float64(
            Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=ptr)[
                unsafe_offset=i
            ]
        )
    if dt == DType.bool:
        var b = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=ptr)[
            unsafe_offset=i
        ]
        return 1.0 if b != 0 else 0.0
    if dt == DType.uint16:
        return Float64(
            Pointer[UInt16, MutUntrackedOrigin](unsafe_from_address=ptr)[
                unsafe_offset=i
            ]
        )
    if dt == DType.uint32:
        return Float64(
            Pointer[UInt32, MutUntrackedOrigin](unsafe_from_address=ptr)[
                unsafe_offset=i
            ]
        )
    if dt == DType.uint64:
        return Float64(
            Pointer[UInt64, MutUntrackedOrigin](unsafe_from_address=ptr)[
                unsafe_offset=i
            ]
        )
    raise Error("unsupported dtype in a host round trip")


def _write_f64_at(ptr: Int, i: Int, v: Float64, dt: DType) raises:
    if dt == DType.float32:
        Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=ptr)[
            unsafe_offset=i
        ] = Float32(v)
    elif dt == DType.bfloat16:
        Pointer[BFloat16, MutUntrackedOrigin](unsafe_from_address=ptr)[
            unsafe_offset=i
        ] = BFloat16(v)
    elif dt == DType.float16:
        Pointer[Float16, MutUntrackedOrigin](unsafe_from_address=ptr)[
            unsafe_offset=i
        ] = Float16(v)
    elif dt == DType.float64:
        Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=ptr)[
            unsafe_offset=i
        ] = v
    elif dt == DType.int64:
        Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=ptr)[
            unsafe_offset=i
        ] = Int64(Int(v))
    elif dt == DType.int32:
        Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=ptr)[
            unsafe_offset=i
        ] = Int32(Int(v))
    elif dt == DType.int16:
        Pointer[Int16, MutUntrackedOrigin](unsafe_from_address=ptr)[
            unsafe_offset=i
        ] = Int16(Int(v))
    elif dt == DType.int8:
        Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=ptr)[
            unsafe_offset=i
        ] = Int8(Int(v))
    elif dt == DType.uint8:
        Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=ptr)[
            unsafe_offset=i
        ] = UInt8(Int(v))
    elif dt == DType.bool:
        Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=ptr)[
            unsafe_offset=i
        ] = UInt8(1) if v != 0.0 else UInt8(0)
    elif dt == DType.uint16:
        Pointer[UInt16, MutUntrackedOrigin](unsafe_from_address=ptr)[
            unsafe_offset=i
        ] = UInt16(Int(v))
    elif dt == DType.uint32:
        Pointer[UInt32, MutUntrackedOrigin](unsafe_from_address=ptr)[
            unsafe_offset=i
        ] = UInt32(Int(v))
    elif dt == DType.uint64:
        Pointer[UInt64, MutUntrackedOrigin](unsafe_from_address=ptr)[
            unsafe_offset=i
        ] = UInt64(Int(v))
    else:
        raise Error("unsupported dtype in a host round trip")


# ---------------------------------------------------------------------------
# Materializing a contiguous copy: a fast PermuteCopy gather for rank<=4
# (mirrors the old `TorchMojoTensor._materialize_contiguous`), else the
# general rank<=8 CopyStrided kernel. ALWAYS a fresh handle (never aliases
# `t`), so it is always safe to wrap the result in `own()`.
# ---------------------------------------------------------------------------


def _permute_materialize(t: T) raises -> T:
    var out = new_like(t)
    var ctx = ctx_for(t.device)
    var cp = ctx_ptr(ctx)
    var pad = 4 - t.rank
    var dims = List[Int](capacity=4)
    var strides = List[Int](capacity=4)
    for i in range(4):
        if i < pad:
            dims.append(1)
            strides.append(0)
        else:
            dims.append(t.dim(i - pad))
            strides.append(t.stride(i - pad))
    var call = KernelCall("data_movement_ops", "PermuteCopy")
    call.arg_dtype(0, t.dtype)
    call.out_dtype(t.dtype)
    call.int(out.ptr)
    call.int(t.ptr)
    call.tuple(dims)
    call.tuple(strides)
    call.int(t.itemsize)
    call.int(cp)
    call.run()
    _ = ctx
    return out^


def _materialize_contiguous(t: T) raises -> T:
    if t.contig:
        var out = new_like(t)
        if t.numel > 0:
            var ctx = ctx_for(t.device)
            copy_d2d(ctx, out.ptr, t.ptr, t.numel * t.itemsize)
            _ = ctx
        return out^
    if t.numel > 0 and t.rank >= 1 and t.rank <= 4:
        return _permute_materialize(t)
    var out = new_like(t)
    if t.numel > 0:
        copy_strided_into(out, t)
    return out^


def _wanted_strides(t: T, mf: Int) raises -> IndexList[MAX_RANK]:
    """The strides a MemoryFormat asks a COPY of `t` to have.

    `preserve_format` is the only one that reads `t`'s own layout: a dense
    input keeps its strides exactly (a channels-last tensor stays
    channels-last through `.clone()` / `.half()`), anything else falls back to
    torch's `infer_dense_strides`.
    """
    if mf != MEMORY_FORMAT_PRESERVE:
        return strides_for_memory_format(t.shape, t.rank, mf)
    if is_dense(t.shape, t.strides, t.rank):
        return t.strides
    return dense_strides_like(t.shape, t.strides, t.rank)


def _materialize_as(t: T, strides: IndexList[MAX_RANK]) raises -> T:
    """A fresh copy of `t`'s values laid out with `strides` (always a new
    handle, so it is always safe to `own()`)."""
    if strides_equal(strides, contiguous_strides(t.shape, t.rank), t.rank):
        return _materialize_contiguous(t)
    var out = own(new_strided(t.shape, strides, t.rank, t.stype, t.device))
    if t.numel > 0:
        if strides_equal(strides, t.strides, t.rank) and is_dense(
            t.shape, t.strides, t.rank
        ):
            # Same layout, densely packed: the two buffers hold the same bytes
            # in the same order.
            var ctx = ctx_for(t.device)
            copy_d2d(ctx, out.t.ptr, t.ptr, t.numel * t.itemsize)
            _ = ctx
        else:
            copy_strided_into(out.t, t)
    return out.take()


# aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
def op_clone(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var mf = v_memory_format_or(args[unsafe_offset=1], MEMORY_FORMAT_PRESERVE)
    var out = own(_materialize_as(t, _wanted_strides(t, mf)))
    ret_owned(rets, 0, out)


# ---------------------------------------------------------------------------
# _to_copy: dtype casts (same mojo device) and device moves. A genuine
# device CHANGE is handled here too (this op is only ever reached when
# `self` already carries the PrivateUse1 dispatch key -- a CPU tensor
# arriving here would already be a bug in the dispatcher), by the same
# host-bounce `_copy_from` uses on its mojo<->cpu paths.
# ---------------------------------------------------------------------------


def _flat_view(t: T) raises -> T:
    """A 1-D contiguous view over a DENSE tensor's elements, in memory order."""
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 1] = t.numel
    var strides = IndexList[MAX_RANK](0)
    strides[MAX_RANK - 1] = 1
    return view_strided(t, shape, strides, 1, t.offset)


def _to_copy_same_device(
    t: T, stype: Int32, want: IndexList[MAX_RANK]
) raises -> T:
    if stype == t.stype:
        return _materialize_as(t, want)
    var dst_dtype = max_dtype(stype)
    if not (_is_cast_dtype(t.dtype) and _is_cast_dtype(dst_dtype)):
        return _relayout_owned(own(_host_cast(t, stype)), want)
    if (
        not strides_equal(want, contiguous_strides(t.shape, t.rank), t.rank)
        and strides_equal(want, t.strides, t.rank)
        and is_dense(t.shape, t.strides, t.rank)
    ):
        # Result and source share one memory order, so the cast reads and
        # writes packed buffers: one CastSpec over flat views, no relayout
        # pass (the channels-last `x.half()` of a CNN).
        var packed = own(new_strided(t.shape, want, t.rank, stype, t.device))
        var flat_src = own(_flat_view(t))
        var flat_dst = own(_flat_view(packed.t))
        cast_into(flat_dst.t, flat_src.t)
        _ = flat_src^  # alive past the launch (the specs read pointers)
        _ = flat_dst^
        return packed.take()
    var src = own_if_new(contiguous(t), t)
    var out = own(new_like_dtype(src.t, stype))
    cast_into(out.t, src.t)
    _ = src^
    return _relayout_owned(out^, want)


def _relayout_owned(var contig: Owned, want: IndexList[MAX_RANK]) raises -> T:
    """Hand `contig` (a fresh CONTIGUOUS result) back in `want`'s layout,
    copying only when the two differ."""
    if strides_equal(
        want, contiguous_strides(contig.t.shape, contig.t.rank), contig.t.rank
    ):
        return contig.take()
    var out = _materialize_as(contig.t, want)
    _ = contig^  # `contig` owns the storage `out` was just read from
    return out^


def _host_cast(t: T, stype: Int32) raises -> T:
    """Exotic dtype pair (outside CastSpec's dtype set, e.g. float64,
    int8/16, uint16/32/64): cast element-by-element on the host through a
    Float64 bridge -- the same precision tradeoff `_read_f64_at` documents.
    """
    var src = contiguous(t)
    var numel = src.numel
    var src_dtype = src.dtype
    var src_shape = src.shape
    var src_rank = src.rank
    var src_device = src.device
    var host_src = own(cpu_empty(src_shape, src_rank, src.stype))
    if numel > 0:
        var ctx = ctx_for(src_device)
        copy_to_host(ctx, src.ptr, host_src.t.ptr, numel * src.itemsize)
        _ = ctx
    release_if_new(src, t)
    var host_dst = own(cpu_empty(src_shape, src_rank, stype))
    var dst_dtype = max_dtype(stype)
    for i in range(numel):
        _write_f64_at(
            host_dst.t.ptr,
            i,
            _read_f64_at(host_src.t.ptr, i, src_dtype),
            dst_dtype,
        )
    var out = new_tensor(src_shape, src_rank, stype, src_device)
    if numel > 0:
        var ctx2 = ctx_for(src_device)
        copy_from_host(
            src_device,
            ctx2,
            out.ptr,
            host_dst.t.ptr,
            numel * dtype_itemsize(dst_dtype),
        )
        _ = ctx2
    # `host_src`/`host_dst` are plain CPU allocations (not this backend's
    # stream-ordered device allocator): keep them alive through their last
    # read above, or a hot allocator can reuse the bytes underneath a
    # "finished" copy that only just enqueued.
    _ = host_src
    _ = host_dst
    return out^


def _download_to_cpu(t: T) raises -> T:
    var out = cpu_empty(t.shape, t.rank, t.stype)
    if t.numel > 0:
        var ctx = ctx_for(t.device)
        copy_to_host(ctx, t.ptr, out.ptr, t.numel * t.itemsize)
        _ = ctx
    return out^


def _upload_cross_device(t: T, target_device: Int) raises -> T:
    var host = own(_download_to_cpu(t))
    var out = new_tensor(t.shape, t.rank, t.stype, target_device)
    if t.numel > 0:
        var ctx = ctx_for(target_device)
        copy_from_host(
            target_device, ctx, out.ptr, host.t.ptr, t.numel * t.itemsize
        )
        _ = ctx
    # A plain CPU allocation: keep it alive through the read above (see the
    # comment in `_host_cast`).
    _ = host
    return out^


def _host_materialize_contiguous(t: T) raises -> T:
    """A contiguous copy of a (possibly strided) CPU tensor's bytes with no
    mojo device/kernel involved -- for a foreign CPU `self` moving onto the
    mojo device, whose `device` field names no mojo index `ctx_for` could
    use."""
    var out = cpu_empty(t.shape, t.rank, t.stype)
    if t.numel == 0:
        return out^
    var itemsize = t.itemsize
    if t.contig:
        unsafe_memcpy(
            dest=Pointer[UInt8, MutUntrackedOrigin](
                unsafe_from_address=out.ptr
            ),
            src=Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=t.ptr),
            count=t.numel * itemsize,
        )
        return out^
    for i in range(t.numel):
        var rem = i
        var src_off = 0
        for d in range(t.rank - 1, -1, -1):
            var extent = t.dim(d)
            src_off += (rem % extent) * t.stride(d)
            rem = rem // extent
        unsafe_memcpy(
            dest=Pointer[UInt8, MutUntrackedOrigin](
                unsafe_from_address=out.ptr + i * itemsize
            ),
            src=Pointer[UInt8, MutUntrackedOrigin](
                unsafe_from_address=t.ptr + src_off * itemsize
            ),
            count=itemsize,
        )
    return out^


def _upload_from_cpu(t: T, stype: Int32, target_device: Int) raises -> T:
    """A real (non-mojo) CPU tensor moving onto the mojo device: a host
    cast (when the dtype changes) then one upload -- mirrors the old eager
    path's `mojo_device__to_copy` "not isinstance(tensor, TorchMojoTensor)"
    branch."""
    var out = new_tensor(t.shape, t.rank, stype, target_device)
    if t.numel == 0:
        return out^
    if t.contig and stype == t.stype:
        # The common case: read straight off the dispatcher's own (borrowed,
        # definitely-live) tensor, exactly like `_copy_from`'s H2D path --
        # no scratch allocation needed.
        var ctx0 = ctx_for(target_device)
        copy_from_host(
            target_device, ctx0, out.ptr, t.ptr, t.numel * t.itemsize
        )
        _ = ctx0
        return out^
    var contiguous_cpu = own(_host_materialize_contiguous(t))
    if stype == t.stype:
        var ctx = ctx_for(target_device)
        copy_from_host(
            target_device,
            ctx,
            out.ptr,
            contiguous_cpu.t.ptr,
            t.numel * t.itemsize,
        )
        _ = ctx
    else:
        var dst_dtype = max_dtype(stype)
        var casted = own(cpu_empty(t.shape, t.rank, stype))
        for i in range(t.numel):
            _write_f64_at(
                casted.t.ptr,
                i,
                _read_f64_at(contiguous_cpu.t.ptr, i, t.dtype),
                dst_dtype,
            )
        var ctx2 = ctx_for(target_device)
        copy_from_host(
            target_device,
            ctx2,
            out.ptr,
            casted.t.ptr,
            t.numel * dtype_itemsize(dst_dtype),
        )
        _ = ctx2
        _ = casted
    _ = contiguous_cpu
    return out^


# aten::_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None,
#   Device? device=None, bool? pin_memory=None, bool non_blocking=False,
#   MemoryFormat? memory_format=None) -> Tensor
def op_to_copy(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var stype = v_dtype_or(args[unsafe_offset=1], t.stype)
    var mf = v_memory_format_or(args[unsafe_offset=6], MEMORY_FORMAT_PRESERVE)
    var want = _wanted_strides(t, mf)
    var contig = contiguous_strides(t.shape, t.rank)
    var dev_v = args[unsafe_offset=3].copy()
    var dev_type = v_device_type(dev_v)
    if (
        dev_type != -1
        and dev_type != DEVICE_TYPE_CPU
        and dev_type != DEVICE_TYPE_PRIVATEUSE1
    ):
        unsupported(
            "aten::_to_copy to a device type this backend does not know"
        )

    if not t.on_mojo():
        # `_to_copy` reaches this PrivateUse1 kernel even for a foreign CPU
        # `self` whenever the target device resolves to `mojo` (dispatch
        # keys on the Device argument too, not just on `self`'s own tensor
        # key) -- so a plain `cpu_tensor.to(mojo_device)` lands here just
        # like a same-device dtype cast does.
        if not t.on_cpu():
            unsupported("aten::_to_copy from a non-cpu, non-mojo device")
        if dev_type != DEVICE_TYPE_PRIVATEUSE1:
            raise Error(
                "aten::_to_copy: self is a cpu tensor with no mojo target"
                " device"
            )
        var target_index = v_device_index(dev_v)
        if target_index < 0:
            raise Error("aten::_to_copy: no explicit mojo device index given")
        var uploaded = own(_upload_from_cpu(t, stype, target_index))
        var out = own(_relayout_owned(uploaded^, want))
        ret_owned(rets, 0, out)
        return

    var target_index2 = -1
    if dev_type == DEVICE_TYPE_PRIVATEUSE1:
        target_index2 = v_device_index(dev_v)
    var cross = target_index2 >= 0 and target_index2 != t.device
    # A device move copies the whole dense buffer verbatim, so `want` is laid
    # out where kernels can run: on the source for a download to the host, on
    # the destination for a move to another mojo device.
    var staged = own(_to_copy_same_device(t, stype, contig if cross else want))
    if dev_type == DEVICE_TYPE_CPU:
        var host = own(_download_to_cpu(staged.t))
        _ = staged^
        if not strides_equal(want, contig, t.rank):
            # `staged` was dense in `want`'s order, so its bytes landed in the
            # host buffer in that order: the layout is metadata from here.
            set_sizes_strides(host.t, t.shape, want, t.rank, 0)
        ret_owned(rets, 0, host)
        return
    if not cross:
        ret_owned(rets, 0, staged)
        return
    var moved = own(_upload_cross_device(staged.t, target_index2))
    _ = staged^
    var out3 = own(_relayout_owned(moved^, want))
    ret_owned(rets, 0, out3)


# ---------------------------------------------------------------------------
# cat / stack: one batched CatN launch when every real input is contiguous
# and the device is a GPU, else a NarrowCopyDst (contiguous input) or a
# strided view + CopyStrided (non-contiguous input) per input.
# ---------------------------------------------------------------------------

comptime _CAT_VECTOR_BYTES = 16


def _is_legacy_empty(t: T) -> Bool:
    return t.rank == 1 and t.numel == 0


def _cat_vector_width(
    out_t: T, ins: List[T], lens: List[Int], dst_stride: Int
) -> Int:
    var itemsize = out_t.itemsize
    var width = _CAT_VECTOR_BYTES // itemsize
    if width <= 1 or width * itemsize != _CAT_VECTOR_BYTES:
        return 1
    if (
        out_t.ptr % _CAT_VECTOR_BYTES != 0
        or (dst_stride * itemsize) % _CAT_VECTOR_BYTES != 0
    ):
        return 1
    for i in range(len(ins)):
        var copy_len = lens[i]
        if copy_len == 0:
            continue
        if (
            ins[i].ptr % _CAT_VECTOR_BYTES != 0
            or (copy_len * itemsize) % _CAT_VECTOR_BYTES != 0
        ):
            return 1
    return width


def _cat_impl(ins: List[T], dim: Int) raises -> Owned:
    var n = len(ins)
    var first = ins[0].copy()
    var rank = first.rank
    if rank == 0:
        unsupported("aten::cat/stack of 0-d tensors")
    if dim < 0 or dim >= rank:
        raise Error("cat: dim out of range")
    for i in range(1, n):
        var b = ins[i].copy()
        if b.dtype != first.dtype or b.device != first.device or b.rank != rank:
            unsupported("aten::cat/stack with mismatched dtype/device/rank")
        for d in range(rank):
            if d != dim and b.dim(d) != first.dim(d):
                raise Error(
                    "cat: sizes of tensors must match except in dimension ", dim
                )
    var cat_size = 0
    for i in range(n):
        cat_size += ins[i].dim(dim)
    var out_shape = first.shape
    out_shape[MAX_RANK - rank + dim] = cat_size
    var out = own(new_tensor(out_shape, rank, first.stype, first.device))
    if out.t.numel == 0:
        return out^
    var inner = 1
    for d in range(dim + 1, rank):
        inner *= first.dim(d)
    var outer = 1
    for d in range(dim):
        outer *= first.dim(d)
    var dst_stride = cat_size * inner
    var ctx = ctx_for(first.device)
    var all_contig = True
    for i in range(n):
        if not ins[i].contig:
            all_contig = False
    if ctx.api() != "cpu" and outer > 0 and dst_stride > 0 and all_contig:
        var srcs = List[Int](capacity=n)
        var lens = List[Int](capacity=n)
        for i in range(n):
            srcs.append(ins[i].ptr)
            lens.append(ins[i].dim(dim) * inner)
        var width = _cat_vector_width(out.t, ins, lens, dst_stride)
        var cp = ctx_ptr(ctx)
        var call = KernelCall("data_movement_ops", "CatN")
        call.arg_dtype(0, first.dtype)
        call.out_dtype(out.t.dtype)
        call.int(out.t.ptr)
        call.tuple(srcs)
        call.tuple(lens)
        call.int(outer)
        call.int(dst_stride)
        call.int(first.itemsize)
        call.int(width)
        call.int(cp)
        call.run()
        _ = ctx
        return out^
    var offset = 0
    for i in range(n):
        var b = ins[i].copy()
        var copy_len = b.dim(dim) * inner
        if copy_len > 0 and outer > 0:
            if b.contig:
                var cp2 = ctx_ptr(ctx)
                var call2 = KernelCall("data_movement_ops", "NarrowCopyDst")
                call2.arg_dtype(0, b.dtype)
                call2.out_dtype(out.t.dtype)
                call2.int(out.t.ptr)
                call2.int(b.ptr)
                call2.int(outer)
                call2.int(dst_stride)
                call2.int(copy_len)
                call2.int(offset)
                call2.int(b.itemsize)
                call2.int(cp2)
                call2.run()
            else:
                var slot = own(
                    view_strided(
                        out.t,
                        b.shape,
                        out.t.strides,
                        rank,
                        out.t.offset + offset,
                    )
                )
                copy_strided_into(slot.t, b)
        offset += copy_len
    _ = ctx
    return out^


# aten::cat(Tensor[] tensors, int dim=0) -> Tensor
def op_cat(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var all_tensors = v_tensor_list(args[unsafe_offset=0])
    var dim_in = v_int_or(args[unsafe_offset=1], 0)
    var real = List[T]()
    for x in all_tensors:
        if not _is_legacy_empty(x):
            real.append(x.copy())
    if len(real) == 0:
        unsupported("aten::cat of only legacy-empty tensors")
    var rank = real[0].rank
    var dim = dim_in + rank if dim_in < 0 else dim_in
    var out = _cat_impl(real, dim)
    ret_owned(rets, 0, out)


# aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
def op_cat_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """DDP's reducer flattens its buckets with this overload."""
    var all_tensors = v_tensor_list(args[unsafe_offset=0])
    var dim_in = v_int_or(args[unsafe_offset=1], 0)
    var out = v_tensor(args[unsafe_offset=2])
    var real = List[T]()
    for x in all_tensors:
        if not _is_legacy_empty(x):
            real.append(x.copy())
    if len(real) == 0:
        unsupported("aten::cat.out of only legacy-empty tensors")
    var rank = real[0].rank
    var dim = dim_in + rank if dim_in < 0 else dim_in
    var result = _cat_impl(real, dim)
    if result.t.dtype != out.dtype:
        raise Error("cat.out: out dtype must match the inputs")
    if not out.on_mojo() or out.device != result.t.device:
        raise Error("cat.out: out must be on the inputs' mojo device")
    # Only a mismatching `out` is resized. Resizing resets sizes, strides and
    # offset, so doing it unconditionally would send `cat(..., out=base[4:8])`
    # to the front of `base`.
    if not out.same_shape(result.t):
        resize_out(out, result.t.shape, result.t.rank)
    copy_strided_into(out, result.t)
    ret_ref(rets, 0, out)


def _unsqueeze_view(t: T, dim: Int) raises -> T:
    var new_rank = t.rank + 1
    if new_rank > MAX_RANK:
        raise Error("stack: result rank exceeds the mojo device limit")
    var new_shape = IndexList[MAX_RANK](1)
    var new_strides = IndexList[MAX_RANK](0)
    var pad_new = MAX_RANK - new_rank
    var pad_old = MAX_RANK - t.rank
    var k = 0
    for i in range(new_rank):
        if i == dim:
            new_shape[pad_new + i] = 1
            new_strides[pad_new + i] = 0
        else:
            new_shape[pad_new + i] = t.shape[pad_old + k]
            new_strides[pad_new + i] = t.strides[pad_old + k]
            k += 1
    return view_strided(t, new_shape, new_strides, new_rank, t.offset)


# aten::stack(Tensor[] tensors, int dim=0) -> Tensor
def op_stack(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var tensors = v_tensor_list(args[unsafe_offset=0])
    var dim_in = v_int_or(args[unsafe_offset=1], 0)
    if len(tensors) == 0:
        unsupported("aten::stack of an empty tensor list")
    var rank = tensors[0].rank
    var out_rank = rank + 1
    var dim = dim_in + out_rank if dim_in < 0 else dim_in
    if dim < 0 or dim >= out_rank:
        raise Error("stack: dim out of range")
    var owned_views = List[Owned]()
    var view_list = List[T]()
    for x in tensors:
        if x.rank != rank:
            unsupported("aten::stack with mismatched ranks")
        var ov = own(_unsqueeze_view(x, dim))
        view_list.append(ov.t.copy())
        owned_views.append(ov^)
    var out = _cat_impl(view_list, dim)
    ret_owned(rets, 0, out)


# ---------------------------------------------------------------------------
# repeat
# ---------------------------------------------------------------------------

# Mirrors data_movement_ops.mojo's `_REPEAT_MAX_EXTENT`: both RepeatTiled
# kernels advance a 32-bit grid-stride counter.
comptime _REPEAT_TILED_MAX_EXTENT = 0x7FFF_F000


@fieldwise_init
struct RepeatPlan(Copyable, Movable):
    var rows: Int
    var cols: Int
    var r1: Int
    var ncopies: Int


def _repeat_tile_plan(
    padded_shape: List[Int], reps: List[Int]
) -> Optional[RepeatPlan]:
    var n = len(padded_shape)
    if n == 0:
        return None
    for i in range(n - 2):
        if padded_shape[i] != 1:
            return None
    var rows = padded_shape[n - 2] if n >= 2 else 1
    var cols = padded_shape[n - 1]
    var r1 = reps[n - 1]
    var ncopies = 1
    for i in range(n - 1):
        ncopies *= reps[i]
    if (
        cols * r1 > _REPEAT_TILED_MAX_EXTENT
        or rows * ncopies > _REPEAT_TILED_MAX_EXTENT
    ):
        return None
    return RepeatPlan(rows, cols, r1, ncopies)


# aten::repeat(Tensor self, SymInt[] repeats) -> Tensor
def op_repeat(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var reps = IntList(args[unsafe_offset=1]).to_list()
    var rank = t.rank
    if len(reps) < rank:
        raise Error(
            "repeat: number of dims of repeat dims can not be smaller than "
            "number of dims of tensor"
        )
    var n_out = len(reps)
    if n_out > MAX_RANK:
        raise Error("repeat: result rank exceeds the mojo device limit")
    var pad = n_out - rank
    var padded_shape = List[Int](capacity=n_out)
    var out_shape_list = List[Int](capacity=n_out)
    for i in range(n_out):
        if reps[i] < 0:
            raise Error("repeats can not be negative")
        var extent = 1 if i < pad else t.dim(i - pad)
        padded_shape.append(extent)
        out_shape_list.append(extent * reps[i])
    var out_shape = _shape_from_values(out_shape_list)
    var out = own(new_tensor(out_shape, n_out, t.stype, t.device))
    if out.t.numel > 0:
        var src = contiguous(t)
        var ctx = ctx_for(t.device)
        var plan: Optional[RepeatPlan] = None
        if ctx.api() != "cpu":
            plan = _repeat_tile_plan(padded_shape, reps)
        var cp = ctx_ptr(ctx)
        if plan:
            var p = plan.value().copy()
            var call = KernelCall("data_movement_ops", "RepeatTiled")
            call.arg_dtype(0, t.dtype)
            call.out_dtype(t.dtype)
            call.int(out.t.ptr)
            call.int(src.ptr)
            call.int(p.rows)
            call.int(p.cols)
            call.int(p.r1)
            call.int(p.ncopies)
            call.int(t.itemsize)
            call.int(cp)
            call.run()
        else:
            var call = KernelCall("data_movement_ops", "TileCopy")
            call.arg_dtype(0, t.dtype)
            call.out_dtype(t.dtype)
            call.int(out.t.ptr)
            call.int(src.ptr)
            call.tuple(_pad8_list(out_shape_list, 1))
            call.tuple(_pad8_list(padded_shape, 1))
            call.tuple(_pad8_list(_row_major(padded_shape), 0))
            call.int(t.itemsize)
            call.int(cp)
            call.run()
        _ = ctx
        release_if_new(src, t)
    ret_owned(rets, 0, out)


# ---------------------------------------------------------------------------
# tril / triu
# ---------------------------------------------------------------------------


def _triangular(t: T, diagonal: Int, upper: Int) raises -> Owned:
    if t.rank < 2:
        unsupported("aten::tril/triu on a tensor with fewer than 2 dims")
    var out = own(new_like(t))
    if out.t.numel > 0:
        var rows = t.dim(-2)
        var cols = t.dim(-1)
        var batch = t.numel // (rows * cols)
        var src = contiguous(t)
        var ctx = ctx_for(t.device)
        var cp = ctx_ptr(ctx)
        var call = KernelCall("data_movement_ops", "TriangularCopy")
        call.arg_dtype(0, t.dtype)
        call.out_dtype(t.dtype)
        call.int(out.t.ptr)
        call.int(src.ptr)
        call.int(batch)
        call.int(rows)
        call.int(cols)
        call.int(diagonal)
        call.int(upper)
        call.int(t.itemsize)
        call.int(cp)
        call.run()
        _ = ctx
        release_if_new(src, t)
    return out^


# aten::tril(Tensor self, SymInt diagonal=0) -> Tensor
def op_tril(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var diagonal = v_int_or(args[unsafe_offset=1], 0)
    var out = _triangular(t, diagonal, 0)
    ret_owned(rets, 0, out)


# aten::triu(Tensor self, SymInt diagonal=0) -> Tensor
def op_triu(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var diagonal = v_int_or(args[unsafe_offset=1], 0)
    var out = _triangular(t, diagonal, 1)
    ret_owned(rets, 0, out)


# ---------------------------------------------------------------------------
# select_scatter
# ---------------------------------------------------------------------------


def _select_view(t: T, dim: Int, index: Int) raises -> T:
    var rank = t.rank
    var size = t.dim(dim)
    var idx = index
    if idx < 0:
        idx += size
    if idx < 0 or idx >= size:
        raise Error("select_scatter: index out of range")
    var new_rank = rank - 1
    var new_shape = IndexList[MAX_RANK](1)
    var new_strides = IndexList[MAX_RANK](0)
    var pad_new = MAX_RANK - new_rank
    var pad_old = MAX_RANK - rank
    var k = 0
    for i in range(rank):
        if i != dim:
            new_shape[pad_new + k] = t.shape[pad_old + i]
            new_strides[pad_new + k] = t.strides[pad_old + i]
            k += 1
    var offset = t.offset + idx * t.stride(dim)
    return view_strided(t, new_shape, new_strides, new_rank, offset)


# aten::select_scatter(Tensor self, Tensor src, int dim, SymInt index) -> Tensor
def op_select_scatter(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var src = v_tensor(args[unsafe_offset=1])
    var dim_in = v_int(args[unsafe_offset=2])
    var index_in = v_int(args[unsafe_offset=3])
    var rank = a.rank
    if rank == 0:
        unsupported("aten::select_scatter on a 0-d tensor")
    var dim = dim_in + rank if dim_in < 0 else dim_in
    if dim < 0 or dim >= rank:
        raise Error("select_scatter: dim out of range")
    if src.device != a.device:
        unsupported("aten::select_scatter with tensors on different devices")
    var out = own(_materialize_contiguous(a))
    var view = own(_select_view(out.t, dim, index_in))
    # `select_scatter_symint` checks `slice.sizes() == src.sizes()` and does
    # not broadcast: a mismatch is an error, never an expand.
    if not src.same_shape(view.t):
        raise Error(
            (
                "select_scatter: expected src to have a size equal to the"
                " slice of self. src rank/size = "
            ),
            src.rank,
            "/",
            src.numel,
            ", slice rank/size = ",
            view.t.rank,
            "/",
            view.t.numel,
        )
    var feed = src.copy()
    var casted: Optional[T] = None
    if src.stype != out.t.stype:
        var c = cast_to(src, out.t.stype)
        casted = c.copy()
        feed = c^
    copy_strided_into(view.t, feed)
    if casted:
        release(casted.value().h)
    ret_owned(rets, 0, out)


# ---------------------------------------------------------------------------
# scatter.src / scatter.value
# ---------------------------------------------------------------------------


def _scatter_common(
    a: T,
    dim_in: Int,
    index: T,
    src: Optional[T],
    value: Float64,
    is_value: Bool,
) raises -> Owned:
    var rank = a.rank
    if rank == 0 or rank > 4:
        unsupported("aten::scatter with rank 0 or greater than 4")
    var dim = dim_in + rank if dim_in < 0 else dim_in
    if dim < 0 or dim >= rank:
        raise Error("scatter: dim out of range")
    if not _is_scatter_dtype(a.dtype):
        unsupported("aten::scatter of dtype " + String(a.dtype))
    if (
        index.dtype != DType.int64
        or index.device != a.device
        or index.rank != rank
    ):
        unsupported("aten::scatter with an unsupported index tensor")
    if src and (
        src.value().dtype != a.dtype
        or src.value().device != a.device
        or src.value().rank != rank
    ):
        unsupported("aten::scatter with an unsupported src tensor")
    # ATen's `scatter_shape_check`, restated before any pointer is read: the
    # index space must fit inside self on every non-scattered axis, and
    # inside src on every axis.
    if index.numel > 0:
        for d in range(rank):
            if d != dim and index.dim(d) > a.dim(d):
                raise Error(
                    (
                        "Expected index to be smaller than self apart from"
                        " dimension "
                    ),
                    dim,
                )
            if src and index.dim(d) > src.value().dim(d):
                raise Error(
                    "Expected index to be smaller than src on dimension ", d
                )

    var ctx = ctx_for(a.device)
    if a.dtype == DType.float64 and ctx.api() == "metal":
        unsupported("aten::scatter of float64 on Apple GPU")

    var out = own(_materialize_contiguous(a))
    var idx_c = contiguous(index)

    var src_c: Optional[T] = None
    var src_ptr = out.t.ptr
    var src_dtype = a.dtype
    if src:
        var sc = contiguous(src.value())
        src_ptr = sc.ptr
        src_dtype = sc.dtype
        src_c = sc^

    var pad4 = 4 - rank
    var params = List[Int](capacity=17)
    for i in range(4):  # dims4 (index's own extents)
        params.append(1 if i < pad4 else idx_c.dim(i - pad4))
    for i in range(4):  # out_strides4
        params.append(0 if i < pad4 else out.t.stride(i - pad4))
    for i in range(4):  # src_strides4
        if i < pad4:
            params.append(0)
        elif src_c:
            params.append(src_c.value().stride(i - pad4))
        else:
            params.append(0)
    for i in range(4):  # idx_strides4
        params.append(0 if i < pad4 else idx_c.stride(i - pad4))
    params.append(dim + pad4)
    params.append(a.dim(dim))

    var bad_index = False
    if idx_c.numel > 0:
        # The kernel skips a write whose index falls outside [0, self.size(dim))
        # and raises this flag; the read back below is one 4-byte D2H, and
        # scatter has already cloned the whole of `self` above.
        var flag = own(
            new_tensor(IndexList[MAX_RANK](1), 1, ST_INT32, a.device)
        )
        fill_value(flag.t, 0.0)
        var cp = ctx_ptr(ctx)
        var call = KernelCall("data_movement_ops", "ScatterDim")
        call.arg_dtype(0, a.dtype)
        call.arg_dtype(1, idx_c.dtype)
        call.arg_dtype(2, src_dtype)
        call.out_dtype(out.t.dtype)
        call.int(out.t.ptr)
        call.int(idx_c.ptr)
        call.int(src_ptr)
        call.tuple(params)
        call.int(flag.t.ptr)
        call.int(1 if is_value else 0)
        call.f64(value)
        call.int(dtype_code(a.dtype))
        call.int(cp)
        call.run()
        var host_flag = own(cpu_empty(IndexList[MAX_RANK](1), 1, ST_INT32))
        copy_to_host(ctx, flag.t.ptr, host_flag.t.ptr, 4)
        bad_index = (
            Pointer[Int32, MutUntrackedOrigin](
                unsafe_from_address=host_flag.t.ptr
            )[]
            != 0
        )
        _ = host_flag
        _ = flag
    _ = ctx
    if src_c:
        release_if_new(src_c.value(), src.value())
    release_if_new(idx_c, index)
    if bad_index:
        raise Error(
            (
                "index out of range in aten::scatter: every index must be in"
                " [0, self.size(dim)) with self.size(dim) = "
            ),
            a.dim(dim),
        )
    return out^


# aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
def op_scatter_src(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var dim = v_int(args[unsafe_offset=1])
    var index = v_tensor(args[unsafe_offset=2])
    var src = v_tensor(args[unsafe_offset=3])
    var out = _scatter_common(a, dim, index, src^, 0.0, False)
    ret_owned(rets, 0, out)


# aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
def op_scatter_value(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var dim = v_int(args[unsafe_offset=1])
    var index = v_tensor(args[unsafe_offset=2])
    var value = v_f64(args[unsafe_offset=3])
    if a.dtype == DType.bool:
        value = 1.0 if value != 0.0 else 0.0
    var out = _scatter_common(a, dim, index, None, value, True)
    ret_owned(rets, 0, out)


# ---------------------------------------------------------------------------
# index.Tensor: a single non-None index at position 0. int32/int64 gathers
# whole rows along dim 0 (GatherRows); a bool mask is data-dependent (the
# output's leading extent is the true count) and round-trips through the
# host, matching whatever mask.rank <= self.rank the caller passes.
# ---------------------------------------------------------------------------


def _index_gather_rows(self_t: T, idx: T, rets: Values) raises:
    if self_t.rank < 1:
        unsupported("aten::index.Tensor on a 0-d tensor")
    var src = contiguous(self_t)
    var idx_c = contiguous(idx)
    var row_len = 1
    for i in range(1, src.rank):
        row_len *= src.dim(i)
    var out_rank = idx_c.rank + (src.rank - 1)
    if out_rank > MAX_RANK:
        release_if_new(idx_c, idx)
        release_if_new(src, self_t)
        raise Error(
            "aten::index.Tensor: result rank exceeds the mojo device limit"
        )
    var out_shape = IndexList[MAX_RANK](1)
    var pad = MAX_RANK - out_rank
    for i in range(idx_c.rank):
        out_shape[pad + i] = idx_c.dim(i)
    for i in range(1, src.rank):
        out_shape[pad + idx_c.rank + (i - 1)] = src.dim(i)
    var out = own(new_tensor(out_shape, out_rank, src.stype, src.device))
    if out.t.numel > 0:
        var ctx = ctx_for(src.device)
        var cp = ctx_ptr(ctx)
        var call = KernelCall("data_movement_ops", "GatherRows")
        call.arg_dtype(0, src.dtype)
        call.arg_dtype(1, idx_c.dtype)
        call.out_dtype(out.t.dtype)
        call.int(out.t.ptr)
        call.int(src.ptr)
        call.int(idx_c.ptr)
        call.int(dtype_code(idx_c.dtype))
        call.int(idx_c.numel)
        call.int(row_len)
        call.int(src.dim(0))
        call.int(src.itemsize)
        call.int(cp)
        call.run()
        _ = ctx
    release_if_new(idx_c, idx)
    release_if_new(src, self_t)
    ret_owned(rets, 0, out)


def _index_bool_mask(self_t: T, mask: T, rets: Values) raises:
    if mask.rank > self_t.rank:
        unsupported("aten::index.Tensor: boolean mask has more dims than self")
    for i in range(mask.rank):
        if mask.dim(i) != self_t.dim(i):
            raise Error(
                "aten::index.Tensor: the shape of the mask does not match the"
                " shape of the indexed tensor"
            )
    var row_len = 1
    for i in range(mask.rank, self_t.rank):
        row_len *= self_t.dim(i)
    var itemsize = self_t.itemsize
    var src = contiguous(self_t)
    var mask_c = contiguous(mask)
    var mask_numel = mask_c.numel
    var host_mask = own(cpu_empty(mask_c.shape, mask_c.rank, mask_c.stype))
    var host_src = own(cpu_empty(src.shape, src.rank, src.stype))
    if mask_numel > 0:
        var ctx = ctx_for(mask_c.device)
        copy_to_host(ctx, mask_c.ptr, host_mask.t.ptr, mask_numel)
        _ = ctx
    if src.numel > 0:
        var ctx2 = ctx_for(src.device)
        copy_to_host(ctx2, src.ptr, host_src.t.ptr, src.numel * itemsize)
        _ = ctx2
    release_if_new(mask_c, mask)
    release_if_new(src, self_t)

    var mask_ptr = Pointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=host_mask.t.ptr
    )
    var count = 0
    for i in range(mask_numel):
        if mask_ptr[unsafe_offset=i] != 0:
            count += 1
    var out_rank = 1 + (self_t.rank - mask.rank)
    if out_rank > MAX_RANK:
        raise Error(
            "aten::index.Tensor: result rank exceeds the mojo device limit"
        )
    var out_shape = IndexList[MAX_RANK](1)
    var pad = MAX_RANK - out_rank
    out_shape[pad] = count
    for i in range(mask.rank, self_t.rank):
        out_shape[pad + 1 + (i - mask.rank)] = self_t.dim(i)
    var out = own(new_tensor(out_shape, out_rank, self_t.stype, self_t.device))
    if count > 0:
        var block_bytes = row_len * itemsize
        var host_out = own(cpu_empty(out.t.shape, out.t.rank, out.t.stype))
        var w = 0
        for i in range(mask_numel):
            if mask_ptr[unsafe_offset=i] != 0:
                unsafe_memcpy(
                    dest=Pointer[UInt8, MutUntrackedOrigin](
                        unsafe_from_address=host_out.t.ptr + w * block_bytes
                    ),
                    src=Pointer[UInt8, MutUntrackedOrigin](
                        unsafe_from_address=host_src.t.ptr + i * block_bytes
                    ),
                    count=block_bytes,
                )
                w += 1
        var ctx3 = ctx_for(self_t.device)
        copy_from_host(
            self_t.device, ctx3, out.t.ptr, host_out.t.ptr, count * block_bytes
        )
        _ = ctx3
        # Plain CPU allocations: keep alive through the reads above (see the
        # comment in `_host_cast`) -- `mask_ptr` in particular caches a raw
        # address into `host_mask` across every loop in this function.
        _ = host_out
    _ = host_mask
    _ = host_src
    ret_owned(rets, 0, out)


# aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
def op_index_tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var self_t = v_tensor(args[unsafe_offset=0])
    var present = v_opt_tensor_list_present(args[unsafe_offset=1])
    var idx_tensors = v_tensor_list(args[unsafe_offset=1])
    var first_pos = -1
    var count_present = 0
    for i in range(len(present)):
        if present[i]:
            count_present += 1
            if first_pos < 0:
                first_pos = i
    if count_present != 1 or first_pos != 0:
        unsupported("aten::index.Tensor with multiple or non-leading indices")
    var idx = idx_tensors[0].copy()
    if idx.device != self_t.device:
        unsupported("aten::index.Tensor with the index on a different device")
    if idx.dtype == DType.int32 or idx.dtype == DType.int64:
        _index_gather_rows(self_t, idx, rets)
        return
    if idx.dtype == DType.bool:
        _index_bool_mask(self_t, idx, rets)
        return
    unsupported("aten::index.Tensor with a non-integer, non-bool index dtype")


# aten::nonzero(Tensor self) -> Tensor
def op_nonzero(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    # A 0-d tensor has one element and no coordinates: ATen scans it like a
    # one-element tensor but reports the result as (n, 0), one empty
    # coordinate row per non-zero element.
    var eff_rank = t.rank if t.rank > 0 else 1
    var out_cols = t.rank
    var eff_shape = List[Int](capacity=eff_rank)
    if t.rank > 0:
        for i in range(t.rank):
            eff_shape.append(t.dim(i))
    else:
        eff_shape.append(1)
    var c = contiguous(t)
    var numel = c.numel
    var c_dtype = c.dtype
    var host = own(cpu_empty(c.shape, c.rank, c.stype))
    if numel > 0:
        var ctx = ctx_for(c.device)
        copy_to_host(ctx, c.ptr, host.t.ptr, numel * c.itemsize)
        _ = ctx
    release_if_new(c, t)

    var flags = List[Bool](capacity=numel)
    for i in range(numel):
        flags.append(_read_f64_at(host.t.ptr, i, c_dtype) != 0.0)
    _ = host  # a plain CPU allocation: keep it alive through the reads above
    var count = 0
    for f in flags:
        if f:
            count += 1

    var out_shape = IndexList[MAX_RANK](1)
    var pad = MAX_RANK - 2
    out_shape[pad] = count
    out_shape[pad + 1] = out_cols
    var out = own(new_tensor(out_shape, 2, ST_INT64, t.device))
    if count > 0 and out_cols > 0:
        var host_out = own(cpu_empty(out.t.shape, 2, ST_INT64))
        var out_ptr = Pointer[Int64, MutUntrackedOrigin](
            unsafe_from_address=host_out.t.ptr
        )
        var w = 0
        for i in range(numel):
            if flags[i]:
                var rem = i
                for d in range(eff_rank - 1, -1, -1):
                    out_ptr[unsafe_offset=w * eff_rank + d] = Int64(
                        rem % eff_shape[d]
                    )
                    rem = rem // eff_shape[d]
                w += 1
        var ctx2 = ctx_for(t.device)
        copy_from_host(
            t.device, ctx2, out.t.ptr, host_out.t.ptr, count * eff_rank * 8
        )
        _ = ctx2
        _ = host_out
    ret_owned(rets, 0, out)


# aten::set_.source_Tensor(Tensor(a!) self, Tensor source) -> Tensor(a!)
def op_set_source_tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var self_t = v_tensor(args[unsafe_offset=0])
    var source = v_tensor(args[unsafe_offset=1])
    if (
        self_t.device != source.device
        or self_t.device_type != source.device_type
    ):
        raise Error(
            "aten::set_.source_Tensor requires both tensors on the same mojo"
            " device"
        )
    # `set_storage_keep_dtype` (upstream's own `set_tensor_`): self's own
    # dtype metadata is untouched, only the storage/sizes/strides move --
    # unlike the old eager path's `_rebind_payload_exact`, which copied
    # `_dtype` too because that wrapper had no independent TensorImpl to
    # keep it in. Real ATen tensors do, so this now matches upstream exactly.
    check(
        external_call["tmb_tensor_set_storage", Int32](self_t.h, source.h),
        "tmb_tensor_set_storage",
    )
    set_sizes_strides(
        self_t, source.shape, source.strides, source.rank, source.offset
    )
    ret_ref(rets, 0, self_t)


# aten::empty_permuted(SymInt[] size, int[] physical_layout, *, ScalarType?
#   dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None)
#   -> Tensor
def op_empty_permuted(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var sizes = IntList(args[unsafe_offset=0])
    var layout = IntList(args[unsafe_offset=1])
    var stype = v_dtype_or(args[unsafe_offset=2], default_dtype())
    var device = _resolve_device(args[unsafe_offset=4])
    var rank = len(sizes)
    if len(layout) != rank:
        raise Error(
            (
                "Number of dimensions in size does not match the length of the"
                " physical_layout; i.e. len(size) = "
            ),
            rank,
            " is not equal to len(physical_layout) = ",
            len(layout),
        )
    # `empty_permuted_symint`: allocate contiguously in the PHYSICAL order the
    # caller asked for, then hand back a logical view of `size` whose strides
    # put dim `physical_layout[i]` at physical position `i`. The result is
    # dense but not contiguous -- the whole point of the op, and the reason
    # a plain contiguous allocation is not a valid answer (`x.stride()` and
    # anything keyed off `is_contiguous` observe the difference).
    var seen = List[Bool](capacity=rank)
    for _ in range(rank):
        seen.append(False)
    var phys_sizes = List[Int](capacity=rank)
    for i in range(rank):
        var d = layout[i]
        if d < 0 or d >= rank:
            raise Error("Dimension out of range in physical_layout: ", d)
        if seen[d]:
            raise Error("Duplicate dim not allowed in physical_layout: ", d)
        seen[d] = True
        phys_sizes.append(sizes[d])
    var phys = own(
        new_tensor(_shape_from_values(phys_sizes), rank, stype, device)
    )
    var shape = _shape_from_values(sizes.to_list())
    var strides = IndexList[MAX_RANK](0)
    var pad = MAX_RANK - rank
    var run = 1
    for i in range(rank - 1, -1, -1):
        strides[pad + layout[i]] = run
        # `TensorImpl::empty_tensor_restride` steps by max(size, 1), so a
        # zero-extent dim leaves the outer strides meaningful.
        run *= max(phys_sizes[i], 1)
    var out = own(view_strided(phys.t, shape, strides, rank, 0))
    ret_owned(rets, 0, out)


def register_data_movement(site: Site) raises:
    impl[op_clone, "clone"](site)
    impl[op_to_copy, "_to_copy"](site)
    impl[op_cat, "cat"](site)
    impl[op_cat_out, "cat.out"](site)
    impl[op_stack, "stack"](site)
    impl[op_repeat, "repeat"](site)
    impl[op_tril, "tril"](site)
    impl[op_triu, "triu"](site)
    impl[op_select_scatter, "select_scatter"](site)
    impl[op_scatter_src, "scatter.src"](site)
    impl[op_scatter_value, "scatter.value"](site)
    impl[op_index_tensor, "index.Tensor"](site)
    impl[op_nonzero, "nonzero"](site)
    impl[op_set_source_tensor, "set_.source_Tensor"](site)
    impl[op_empty_permuted, "empty_permuted"](site)


@export
def tmb_op_address() abi("C") -> Int:
    """Entry of this file's one-op extension: the address of the op the
    TMB_OP define selected (registry.mojo)."""
    return op_address_of[register_data_movement]()
