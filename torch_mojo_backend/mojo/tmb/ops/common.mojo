"""Helpers every op group shares: materializing a contiguous copy, strided
copies and fills, dtype casts (through the memory / data_movement
families on the tensor's current stream), scalar embedding and binary type
promotion, `out=` resizing, and the Philox reservation more than one group
needs. Calling another aten op through the real dispatcher is `abi.call_op`,
re-exported here."""
from std.ffi import external_call
from std.utils import IndexList

from tmb.backend.abi import (
    ST_BFLOAT16,
    TAG_BOOL,
    TAG_COMPLEX,
    TAG_DOUBLE,
    TAG_SCALAR_DOUBLE,
    TAG_INT,
    TAG_SCALAR_BOOL,
    TAG_SCALAR_INT,
    ST_BOOL,
    ST_FLOAT16,
    ST_FLOAT32,
    ST_FLOAT64,
    ST_COMPLEX128,
    ST_INT16,
    ST_INT32,
    ST_INT64,
    ST_INT8,
    ST_UINT8,
    ST_UINT16,
    ST_UINT32,
    ST_UINT64,
    ST_COMPLEX64,
    Owned,
    T,
    Value,
    Values,
    _channels_last_strides,
    check,
    is_dense,
    Results,
    call_op,
    call_op_raw,
    contiguous_strides,
    dtype_code,
    dtype_itemsize,
    dtype_name,
    is_dense,
    f64_bits,
    is_complex_stype,
    max_dtype,
    real_of,
    cpu_empty,
    tensor_arg,
    bool_arg,
    new_like,
    new_like_dtype,
    new_strided,
    new_tensor,
    own,
    own_if_new,
    release,
    set_sizes_strides,
    torch_dtype,
    unsupported,
    v_f64,
    v_scalar_is_integral,
    v_int,
    v_complex,
)
from tmb.backend.device import (
    copy_from_host,
    copy_to_host,
    ctx_for,
    ctx_ptr,
    dev,
    memset_bytes,
    memset_typed,
)
from tmb.backend.kernel_call import KernelCall
from tmb.kernels.common.op_utils import MAX_RANK

# `call_op` / `call_op_raw` live in abi.mojo (one implementation, shared with
# the ops that import them from there); they are re-exported here because most
# op groups reach every shared helper through ops_common.


def _padded(shape: IndexList[MAX_RANK]) -> List[Int]:
    var out = List[Int](capacity=MAX_RANK)
    for i in range(MAX_RANK):
        out.append(shape[i])
    return out^


def shape_str(t: T) raises -> String:
    var s = String("(")
    for i in range(t.rank):
        if i:
            s += ", "
        s += String(t.dim(i))
    return s + ")"


def copy_strided_into(dst: T, src: T) raises:
    """dst[...] = src[...] for EQUAL logical shapes, any strides, same dtype
    (memory CopyStrided: element-size dispatch, rank <= MAX_RANK).

    The kernel walks ONE shape -- the destination's -- indexing both tensors
    with their own strides, so a source of any other shape is read with the
    destination's extents and runs past its storage. Equal element counts are
    not enough: (2,3) into (3,2) has the same numel and reads out of bounds.
    A caller with a differently shaped dense source views it as the
    destination's shape first.
    """
    if src.stype != dst.stype:
        raise Error("copy_strided_into: dtype mismatch")
    if not dst.same_shape(src):
        raise Error(
            "copy_strided_into: shape mismatch, destination ",
            shape_str(dst),
            " and source ",
            shape_str(src),
        )
    if dst.numel == 0:
        return
    if dst.itemsize == 16:
        # complex128: no 16-byte element kernel; move the component pairs.
        copy_strided_into(dst.pairs(), src.pairs())
        return
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("memory", "CopyStrided")
    call.int(dst.ptr)
    call.int(src.ptr)
    call.tuple(_padded(dst.shape))
    call.tuple(_padded(dst.strides))
    call.tuple(_padded(src.strides))
    call.int(dst.itemsize)
    call.int(cp)
    call.run()
    _ = ctx


def broadcast_shape(a: T, b: T) raises -> IndexList[MAX_RANK]:
    """The broadcast of two shapes (leading-padded, so rank max(a, b))."""
    var shape = IndexList[MAX_RANK](1)
    for i in range(MAX_RANK):
        var x = a.shape[i]
        var y = b.shape[i]
        if x == y or y == 1:
            shape[i] = x
        elif x == 1:
            shape[i] = y
        else:
            raise Error("shapes are not broadcastable")
    return shape


def _sizes_str(t: T) -> String:
    var s = String("[")
    for i in range(t.rank):
        if i > 0:
            s += ", "
        s += String(t.dim(i))
    return s + "]"


def check_expands_to(t: T, target: T) raises:
    """`t.expand(target.sizes())` would succeed (ATen's `expand_inplace`):
    no more dims than the target, and each dim equal to the target's or 1.
    Raises expand's own errors."""
    if t.rank > target.rank:
        raise Error(
            "expand(",
            scalar_type_name(t.stype),
            "Type{",
            _sizes_str(t),
            "}, size=",
            _sizes_str(target),
            "): the number of sizes provided (",
            target.rank,
            (
                ") must be greater or equal to the number of dimensions in the"
                " tensor ("
            ),
            t.rank,
            ")",
        )
    for i in range(1, t.rank + 1):
        var have = t.dim(t.rank - i)
        var want = target.dim(target.rank - i)
        if have != want and have != 1:
            raise Error(
                "The expanded size of the tensor (",
                want,
                ") must match the existing size (",
                have,
                ") at non-singleton dimension ",
                target.rank - i,
                ".  Target sizes: ",
                _sizes_str(target),
                ".  Tensor sizes: ",
                _sizes_str(t),
            )


def device_str(t: T) -> String:
    """`c10::Device::str()`: the lower-case device-type name plus an index
    when the tensor carries one. PrivateUse1 prints under the name torch was
    renamed to (`rename_privateuse1_backend("mojo")`)."""
    if t.on_mojo():
        return String("mojo:") + String(t.device)
    if t.on_cpu():
        return String("cpu")
    return String("device type ") + String(t.device_type)


def _repeats_elements(t: T) -> Bool:
    for i in range(t.rank):
        if t.stride(i) <= 0 and t.dim(i) > 1:
            return True
    return False


comptime OVERLAP_NO = 0
comptime OVERLAP_FULL = 1
comptime OVERLAP_PARTIAL = 2
comptime OVERLAP_TOO_HARD = 3

comptime _OVERLAP_MESSAGE = (
    "unsupported operation: some elements of the input tensor and the"
    " written-to tensor refer to a single memory location. Please clone()"
    " the tensor before performing the operation."
)


def _dense(t: T) -> Bool:
    """`TensorImpl::is_non_overlapping_and_dense_or_false`."""
    return t.contig or is_dense(t.shape, t.strides, t.rank)


def overlap_status(a: T, b: T) -> Int:
    """`at::get_overlap_status` (ATen/MemoryOverlap.cpp), in its order: the
    same TensorImpl is FULL; an empty side is NO; a side that is not
    non-overlapping-and-dense (a broadcast, a strided view such as `x[::2]`)
    is TOO_HARD; different storages are NO; otherwise the dense byte ranges
    decide FULL / PARTIAL / NO."""
    if a.impl() == b.impl():
        return OVERLAP_FULL
    if a.numel == 0 or b.numel == 0:
        return OVERLAP_NO
    if not _dense(a) or not _dense(b):
        return OVERLAP_TOO_HARD
    var storage = a.storage_ptr()
    if storage == 0 or storage != b.storage_ptr():
        return OVERLAP_NO
    var a_end = a.ptr + a.numel * a.itemsize
    var b_end = b.ptr + b.numel * b.itemsize
    if a.ptr == b.ptr and a_end == b_end:
        if a.rank == b.rank:
            var same = True
            for i in range(a.rank):
                if a.stride(i) != b.stride(i):
                    same = False
            if same:
                return OVERLAP_FULL
        return OVERLAP_PARTIAL
    if a.ptr < b_end and b.ptr < a_end:
        return OVERLAP_PARTIAL
    return OVERLAP_NO


def assert_no_overlap(written: T, other: T) raises:
    """`at::assert_no_overlap`: the written tensor may not share any memory
    with an input, the same tensor included (FULL or PARTIAL raises);
    TOO_HARD lets the call through, as in torch, so disjoint interleaved
    views (`x[::2]` written from `x[1::2]`) are accepted."""
    var status = overlap_status(written, other)
    if status == OVERLAP_FULL or status == OVERLAP_PARTIAL:
        raise Error(_OVERLAP_MESSAGE)


def assert_no_partial_overlap(written: T, other: T) raises:
    """`at::assert_no_partial_overlap`: only PARTIAL raises (the same view
    is fine for an in-place elementwise op)."""
    if overlap_status(written, other) == OVERLAP_PARTIAL:
        raise Error(_OVERLAP_MESSAGE)


# ---------------------------------------------------------------------------
# When each out= / in-place op checks overlap, from the ATen v2.14 sources
# (structured meta, then the CUDA impl). "after resize" checks run here on
# `resized_geometry(out, ...)` before anything is resized or written: the
# same verdict, with no input pointer left stale. Status names as in
# `overlap_status`; "partial" = assert_no_partial_overlap, "overlap" =
# assert_no_overlap, "internal" = assert_no_internal_overlap.
#
#  op (overload)                     | order and checks
#  ----------------------------------+------------------------------------
#  cat.out (TensorShape.cpp meta)    | resize, then internal + overlap vs
#                                    | every input
#  gather.out (meta)                 | resize, then internal, overlap vs
#                                    | self, partial vs index
#  index_select.out (Indexing.cu)    | internal, overlap vs self and index
#                                    | on the out as given, then resize
#  scatter.* / scatter_add /         | internal, overlap vs index and src
#  scatter_reduce out= and in-place  | on the out as given (meta, before
#  (scatter_meta_impl, scatter_impl) | set_output); then resize; then
#                                    | `copy_(self)` unless the out IS self:
#                                    | partial vs self
#  index_add / index_reduce /        | resize, then internal, overlap vs
#  index_copy out= and in-place      | index and source (index_func_meta_
#  (index_func_meta_impl)            | impl); then `copy_(self)` unless the
#                                    | out IS self: partial vs self
#  put_ / index_put_                 | overlap of self vs values (self is
#                                    | never resized)
#  upsample_nearest2d,               | resize (meta); the kernel's same-
#  _upsample_nearest_exact2d,        | size shortcut `output.copy_(input)`:
#  upsample_bilinear2d, fwd and bwd  | internal, then partial vs input
#  (UpSampleNearest2d.cu,            | (nearest fwd skips it for an empty
#  UpSampleBilinear2d.cu)            | input, bwd for an empty grad_input;
#                                    | bilinear bwd zeroes grad_input first)
#  other upsample modes and sizes    | resize; no input check; internal
#                                    | only where the kernel copies a
#                                    | non-contiguous out back (nearest 2-d,
#                                    | nearest 3-d fwd, bilinear 2-d bwd,
#                                    | trilinear bwd, antialiased fwd)
#  reflection / replication pad      | no overlap check at all
# ---------------------------------------------------------------------------


def resized_geometry(dest: T, shape: IndexList[MAX_RANK], rank: Int) -> T:
    """`dest` as `resize_out(dest, shape, rank)` would leave it -- the same
    tensor (handle, storage, offset), contiguous at the new shape unless it
    already has that shape -- for overlap checks that ATen runs after the
    resize, made before it. Kernel-only: never handed to the dispatcher."""
    if dest.rank == rank:
        var same = True
        for d in range(rank):
            if dest.dim(d) != shape[MAX_RANK - rank + d]:
                same = False
        if same:
            return dest.copy()
    var t = dest.copy()
    # An empty view reports no data pointer; the resize keeps its offset.
    t.ptr = dest.storage_ptr() + dest.offset * dest.itemsize
    t.rank = rank
    t.shape = shape
    t.strides = contiguous_strides(shape, rank)
    t.numel = 1
    for d in range(rank):
        t.numel *= shape[MAX_RANK - rank + d]
    t.contig = True
    return t^


def shares_storage(a: T, b: T) -> Bool:
    """Whether `a` and `b` view one storage, overlapping or not. An `out=`
    that shares storage with an input must not be resized before the kernel
    has read that input: `resize_out` may reallocate the storage and leave the
    input's cached data pointer dangling. Compute into a fresh tensor, then
    resize and copy (see `tmb/ops/resample.mojo`'s `_dest`)."""
    var s = a.storage_ptr()
    return s != 0 and s == b.storage_ptr()


def same_view(a: T, b: T) -> Bool:
    """Whether `a` and `b` address exactly the same elements in the same
    order (same data pointer, shape and strides)."""
    if a.ptr != b.ptr or a.stype != b.stype or not a.same_shape(b):
        return False
    for i in range(a.rank):
        if a.dim(i) > 1 and a.stride(i) != b.stride(i):
            return False
    return True


def assert_no_internal_overlap(t: T) raises:
    """`at::assert_no_internal_overlap`: an `out=` tensor may not alias
    itself (e.g. a size-1 storage `.expand()`ed to more than one logical
    element) -- distinct reduction results written to the same physical
    address would silently collapse into whichever write lands last.

    An EMPTY tensor is exempt regardless of its strides (matches ATen's
    `is_non_overlapping_and_dense_or_false`, which short-circuits before ever
    reading strides for a 0-element tensor): nothing is written, so
    degenerate strides from an `.expand()` over a zero extent -- a normal way
    to reduce e.g. a (0, 3, 1) tensor over its last dim into `out=
    torch.empty(0, 1).expand(0, 3)` -- can't actually collapse anything.
    Confirmed on stock CUDA torch that this exact case is accepted."""
    if t.numel == 0:
        return
    if _repeats_elements(t):
        raise Error(
            "unsupported operation: more than one element of the written-to"
            " tensor refers to a single memory location. Please clone() the"
            " tensor before performing the operation."
        )


def check_out(dest: T, like: T) raises:
    """The dtype and device half of torch's generated `resize_out`
    (torchgen/dest/register_dispatch_key.py, `gen_resize_out_helper`).

    An `out=` tensor must ALREADY have the result's dtype and live on the
    result's device; only its shape may differ, and `resize_out` below fixes
    that. `like` is the input the structured meta function takes its
    `TensorOptions` from. Run this before any kernel: a wrong `out` then
    costs no launch, and a float result can never be silently truncated into
    an integer buffer nor copied across devices without ordering.
    """
    check_out_as(dest, like.stype, like)


def check_out_as(dest: T, stype: Int32, like: T) raises:
    """`check_out` for an `out=` whose dtype is not `like`'s (the int64
    `indices` of the max pools): `dest` must hold `stype` and live on
    `like`'s device."""
    if dest.stype != stype:
        raise Error(
            "Expected out tensor to have dtype ",
            dtype_name(stype),
            ", but got ",
            dtype_name(dest.stype),
            " instead",
        )
    if dest.device_type != like.device_type or dest.device != like.device:
        raise Error(
            "Expected out tensor to have device ",
            device_str(like),
            ", but got ",
            device_str(dest),
            " instead",
        )


def resize_storage_bytes(t: T, nbytes: Int) raises:
    """`resize_bytes_cuda`: reallocate `t`'s storage to `nbytes`, keeping the
    bytes up to the smaller size -- refused for a storage that is not
    resizable (a DLPack import, a `from_blob` view), as CUDA refuses it."""
    if external_call["tmb_tensor_storage_resizable", Int32](t.h) == 0:
        raise Error("Trying to resize storage that is not resizable")
    check(
        external_call["tmb_storage_resize", Int32](t.h, Int64(nbytes)),
        "tmb_storage_resize",
    )


def resize_out(mut t: T, shape: IndexList[MAX_RANK], rank: Int) raises:
    """torch's `resize_output` for a caller's `out=` tensor, in place.

    Boxed kernels get no resize before dispatch, so every `out=` op here
    does this itself; `aten::resize_` (tmb/ops/data_movement.mojo) is this
    too, after its argument checks. Two halves, both of
    `at::native::resize_impl`:

    * **An `out` that already has this logical shape is left alone** --
      strides and storage offset included. `out=base[4:8]` or a transposed
      `out` is written where it lives; rewriting it to a fresh contiguous
      layout at offset 0 would scribble over the start of `base`.
    * Otherwise it is re-laid-out contiguously **at its existing storage
      offset**, growing the storage to `(offset + numel) * itemsize` first
      (`tmb_storage_resize` preserves bytes up to `min(old, new)` like
      torch's `resize_`; `tmb_tensor_set_sizes_strides` bounds-checks
      against the storage's CURRENT size, hence the order). The common case
      is a composite handing an `out=` op a fresh `at::empty({0}, ...)`.

    `t`'s cached view fields are refreshed afterwards: its shape, strides,
    numel and contiguity all changed.

    Not reproduced: the `TORCH_WARN` ATen emits when the resized `out` was
    non-empty (deprecated behaviour, advisory only) -- the shim has no way to
    raise a python warning from a Mojo op.
    """
    if t.rank == rank:
        var same = True
        for i in range(rank):
            if t.dim(i) != shape[MAX_RANK - rank + i]:
                same = False
                break
        if same:
            return
    var numel = 1
    for i in range(rank):
        numel *= shape[MAX_RANK - rank + i]
    var offset = t.offset
    var nbytes = (offset + numel) * t.itemsize
    # `maybe_resize_storage_cuda`: an empty result never touches the storage.
    if numel > 0 and nbytes > t.storage_nbytes():
        resize_storage_bytes(t, nbytes)
    set_sizes_strides(t, shape, contiguous_strides(shape, rank), rank, offset)
    t = T(t.h)


def contiguous(t: T) raises -> T:
    """`t` itself when already contiguous, else a fresh contiguous copy
    (an owned handle: release it or return it -- `own_if_new(contiguous(t), t)`
    does both)."""
    if t.contig:
        return t.copy()
    var out = own(new_like(t))
    copy_strided_into(out.t, t)
    return out.take()


struct FillScalar(Copyable, Movable):
    """One constant to fill with, in the forms the different destination
    dtypes store it in.

    An ATen `Scalar` carries a tag, and rounding it all the way down to a
    Float64 the way one number-typed argument would loses three things a
    fill must keep: the truth of a bool destination (`.fill_(0.5)` is True,
    not `Int(0.5) == 0`), every integer bit above 2**53, and the sign of
    `-0.0`.
    """

    var f: Float64  # what a floating destination (and the fill kernel) takes
    var i: Int  # the exact value, when `integral`
    var integral: Bool
    var truth: Bool  # nonzero truth, what a bool destination stores

    def __init__(out self, value: Float64):
        self.f = value
        self.i = Int(value)
        self.integral = False
        self.truth = value != 0.0

    def __init__(out self, v: Value) raises:
        """From an ATen Scalar record."""
        self.integral = v_scalar_is_integral(v)
        if self.integral:
            self.i = v_int(v)
            self.f = Float64(self.i)
            self.truth = self.i != 0
        else:
            self.f = v_f64(v)
            self.i = Int(self.f)
            self.truth = self.f != 0.0

    def as_int(self) -> Int:
        """torch's Scalar -> integer conversion (a float truncates)."""
        return self.i

    def is_zero_bits(self) -> Bool:
        """Whether every dtype stores this value as all-zero bytes: positive
        zero and integer zero, but not `-0.0` (sign bit set)."""
        if self.integral:
            return self.i == 0
        return f64_bits(self.f) == 0


def fill_value(t: T, value: Float64) raises:
    """Constant fill from a plain number (`zero_`, an op filling with a
    computed float). `fill_value(t, some_scalar_record)` keeps the ATen
    Scalar's tag instead -- prefer it wherever the caller has the record."""
    _fill(t, FillScalar(value))


def fill_value(t: T, value: Value) raises:
    """Constant fill from an ATen `Scalar` argument record, tag kept."""
    _fill(t, FillScalar(value))


def _fill(t: T, s_in: FillScalar) raises:
    """A memset when contiguous, else the strided fill kernel. A half /
    bfloat16 destination takes the value through float first, as c10's
    Half / BFloat16 conversions from double do (1 + 2**-11 + 2**-30 fills
    a half with 1.0, not 1.0009765625)."""
    if t.numel == 0:
        return
    var s = s_in.copy()
    if s.integral and (
        t.dtype == DType.float32
        or t.dtype == DType.float16
        or t.dtype == DType.bfloat16
    ):
        # an integer converts to float in one rounding (int64 -> float), not
        # through a double
        s.f = Float64(Int64(s.i).cast[DType.float32]())
    elif t.dtype == DType.float16 or t.dtype == DType.bfloat16:
        s.f = Float64(Float32(s.f))
    if t.contig:
        _fill_contiguous(t, s)
        return
    if s.integral and t.itemsize == 8 and not t.dtype.is_floating_point():
        # The StridedFill kernel narrows a Float64 into the destination
        # dtype, which cannot carry a 64-bit integer past 2**53. Fill a dense
        # buffer exactly (memset) and lay that out instead.
        # (not abs(): it overflows for INT64_MIN, a uint64 2**63's bits)
        if s.i > _MAX_EXACT_INT or s.i < -_MAX_EXACT_INT:
            var dense = own(new_like(t))
            _fill_contiguous(dense.t, s)
            copy_strided_into(t, dense.t)
            _ = dense^  # alive past the launch
            return
    var ctx = ctx_for(t.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("memory", "StridedFill")
    call.int(t.ptr)
    call.f64(s.f)
    call.tuple(_padded(t.shape))
    call.tuple(_padded(t.strides))
    call.int(dtype_code(t.dtype))
    call.int(cp)
    call.run()
    _ = ctx


def _fill_contiguous(t: T, s: FillScalar) raises:
    var ctx = ctx_for(t.device)
    if t.dtype == DType.bool:
        # torch stores a bool as one byte holding exactly 0 or 1, and a
        # Scalar is true when it is nonzero.
        memset_bytes(ctx, t.ptr, UInt8(1) if s.truth else UInt8(0), t.numel)
    elif s.is_zero_bits():
        memset_bytes(ctx, t.ptr, 0, t.numel * t.itemsize)
    elif t.dtype == DType.float32:
        memset_typed[DType.float32](ctx, t.ptr, Float32(s.f), t.numel)
    elif t.dtype == DType.bfloat16:
        memset_typed[DType.bfloat16](ctx, t.ptr, BFloat16(s.f), t.numel)
    elif t.dtype == DType.float16:
        memset_typed[DType.float16](ctx, t.ptr, Float16(s.f), t.numel)
    elif t.dtype == DType.float64:
        memset_typed[DType.float64](ctx, t.ptr, s.f, t.numel)
    elif t.dtype == DType.int64:
        memset_typed[DType.int64](ctx, t.ptr, Int64(s.as_int()), t.numel)
    elif t.dtype == DType.int32:
        memset_typed[DType.int32](ctx, t.ptr, Int32(s.as_int()), t.numel)
    elif t.dtype == DType.int16:
        memset_typed[DType.int16](ctx, t.ptr, Int16(s.as_int()), t.numel)
    elif t.dtype == DType.int8:
        memset_typed[DType.int8](ctx, t.ptr, Int8(s.as_int()), t.numel)
    elif t.dtype == DType.uint8:
        memset_bytes(ctx, t.ptr, UInt8(s.as_int()), t.numel)
    elif t.dtype == DType.uint16:
        memset_typed[DType.uint16](ctx, t.ptr, UInt16(s.as_int()), t.numel)
    elif t.dtype == DType.uint32:
        memset_typed[DType.uint32](ctx, t.ptr, UInt32(s.as_int()), t.numel)
    elif t.dtype == DType.uint64:
        memset_typed[DType.uint64](ctx, t.ptr, UInt64(s.as_int()), t.numel)
    else:
        unsupported("fill of dtype " + String(t.dtype))
    _ = ctx


def is_cast_dtype(dt: DType) -> Bool:
    """Mirrors tmb/kernels/data_movement/entry.mojo's `CAST_DTYPES`, dtype
    only: float64 is in the list here even though `_cast` itself declines it
    on Apple GPUs. A caller choosing between the device cast kernel and a
    host-round-trip fallback must use `is_cast_dtype_on` instead, or it will
    route float64 into `_cast` on Metal and hit that raw decline."""
    return (
        dt == DType.float32
        or dt == DType.float16
        or dt == DType.bfloat16
        or dt == DType.float64
        or dt == DType.int64
        or dt == DType.int32
        or dt == DType.int16
        or dt == DType.int8
        or dt == DType.uint8
        or dt == DType.bool
    )


def is_cast_dtype_on(dt: DType, t: T) raises -> Bool:
    """`is_cast_dtype`, narrowed by `t`'s device: false for float64 on an
    Apple GPU, where the device cast kernel can't run it (`_cast` raises
    there). Every caller that picks between the device cast kernel and a
    host round-trip fallback (aten.to/copy_'s hot paths in core.mojo and
    data_movement.mojo, the bool-mixed promotions below) must check this
    instead of the bare dtype predicate, so float64 keeps working on Metal
    through the fallback exactly as it did before float64 joined
    `CAST_DTYPES`."""
    if dt == DType.float64 and t.on_mojo() and dev(t.device)[].api == "metal":
        return False
    return is_cast_dtype(dt)


def cast_into(dst: T, src: T) raises:
    """dst = src.to(dst.dtype) for a contiguous dst (data_movement CastSpec)."""
    if src.numel == 0:
        return
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("data_movement", "CastSpec")
    call.arg_dtype(0, src.dtype)
    call.out_dtype(dst.dtype)
    call.spec(src.spec(cp))
    call.int(dtype_code(dst.dtype))
    call.spec(dst.spec(cp))
    call.run()
    _ = ctx


def cast_to(t: T, stype: Int32) raises -> T:
    """A contiguous copy of `t` in dtype `stype` (t itself when unchanged).
    Both the output and the intermediate are released if the cast raises."""
    if t.stype == stype:
        return t.copy()
    var out = own(new_like_dtype(t, stype))
    var src = own_if_new(contiguous(t), t)
    cast_into(out.t, src.t)
    _ = src^  # alive past the launch (its last use above is the pointer read)
    return out.take()


def one_device(a: T, b: T) raises:
    """Both operands of a raw-pointer launch on the same mojo device. A
    pointer belonging to another device -- or to no mojo device at all --
    would be dereferenced against the wrong context."""
    if not a.on_mojo() or not b.on_mojo() or a.device != b.device:
        raise Error("expected every operand on the same mojo device")


def elementwise_direct(
    family: String,
    op: String,
    src_c: T,
    mut dst: T,
    out_dtype: DType,
) raises:
    """dst[...] = f(src_c[...]); src_c must already be contiguous, dst must
    already be the right shape/dtype/contiguity. The one-TensorSpec-in/
    one-out shape every direct unary op (abs, neg, sign, ...) and the
    vector-norm size-one-reduce fast path (reductions.mojo's
    `_vector_norm_abs*`, also a true 1:1 elementwise op) share."""
    one_device(src_c, dst)
    if src_c.numel == 0:
        return
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall(family, op)
    call.arg_dtype(0, src_c.dtype)
    call.out_dtype(out_dtype)
    call.spec(src_c.spec(cp))
    call.spec(dst.spec(cp))
    call.run()
    _ = ctx


def philox_reserve(
    generator: Int, device: Int, increment: Int
) raises -> Tuple[UInt64, UInt64]:
    """Atomically reserve `increment` counters of a device's (generator=0)
    or an explicit generator's Philox stream: returns `(seed, base_offset)`
    as they stood *before* the reservation (tmb_philox_reserve,
    agents_docs/native_backend.md). Generic device-runtime plumbing any RNG op of
    any group needs, not specific to one op group."""
    var seed: UInt64 = 0
    var offset: UInt64 = 0
    check(
        external_call["tmb_philox_reserve", Int32](
            generator,
            Int32(device),
            UInt64(increment),
            Pointer(to=seed),
            Pointer(to=offset),
        ),
        "tmb_philox_reserve",
    )
    return (seed, offset)


# ---------------------------------------------------------------------------
# Scalar embedding and binary dtype promotion, shared by the compare and
# binary op groups (ported from `_scalar_embed` / `_binary_promotion` /
# `_promoted_pair` in the old eager_kernels/aten_fast.py).
# ---------------------------------------------------------------------------

# int64 scalars round-trip through a Float64 fill argument exactly up to
# this magnitude.
comptime _MAX_EXACT_INT = 9007199254740992  # 2**53


def _is_embeddable_dtype(dtype: DType) -> Bool:
    """Dtypes `scalar_embed`'s destination fill can target (matches the old
    `_FILL_DTYPES`; a strict subset of what `fill_value` itself supports,
    kept for fidelity with the old eager path)."""
    return (
        dtype == DType.float32
        or dtype == DType.float16
        or dtype == DType.bfloat16
        or dtype == DType.float64
        or dtype == DType.int8
        or dtype == DType.int16
        or dtype == DType.int32
        or dtype == DType.int64
        or dtype == DType.uint8
        or dtype == DType.bool
    )


def scalar_embed(v: Value, dtype: DType) raises -> Float64:
    """`v` (an ATen Scalar record) validated for lossless embedding into
    `dtype`, as a Float64 ready for `fill_value` / a 0-d fill tensor.

    Ported from `_scalar_embed`: an int/bool magnitude above 2**53 would
    lose precision through the Float64 round-trip and is declined, a bool
    destination only accepts 0/1, and a float scalar against a
    non-floating destination is declined (no implicit promotion here --
    callers that want promotion cast the tensor operand first).
    """
    if not _is_embeddable_dtype(dtype):
        unsupported("scalar embedding into dtype " + String(dtype))
    if v_scalar_is_integral(v):
        var i = v_int(v)
        if abs(i) > _MAX_EXACT_INT:
            unsupported("scalar magnitude exceeds the exact float64 range")
        if dtype == DType.bool and i != 0 and i != 1:
            unsupported("a bool tensor's scalar must be 0 or 1")
        return Float64(i)
    if (
        dtype != DType.float16
        and dtype != DType.bfloat16
        and dtype != DType.float32
        and dtype != DType.float64
    ):
        unsupported("a float scalar against a non-floating tensor")
    return v_f64(v)


# ---------------------------------------------------------------------------
# type promotion (c10/core/ScalarType.cpp + ATen's ResultTypeState)
# ---------------------------------------------------------------------------

comptime ST_UNDEFINED = Int32(-1)


def is_float_stype(st: Int32) -> Bool:
    return (
        st == ST_FLOAT32
        or st == ST_FLOAT16
        or st == ST_BFLOAT16
        or st == ST_FLOAT64
    )


def is_int_stype(st: Int32) -> Bool:
    return (
        st == ST_UINT8
        or st == ST_INT8
        or st == ST_INT16
        or st == ST_INT32
        or st == ST_INT64
    )


def scalar_to_int(v: Value, st: Int32) raises -> Int:
    """An ATen Scalar as integral dtype `st`'s value, never through a float
    when it is an integer: `Scalar::to<scalar_t>()` (c10's checked_convert).
    Out of range raises as `c10::overflows` says: an integer may be negative
    down to minus an unsigned type's maximum (two's complement wrap), a float
    must lie in [lowest, max] and then truncates. The result is the exact
    integer, which the caller narrows into `st`."""
    var lo = Int.MIN
    var hi = Int.MAX
    var name = String("int64_t")
    if st == ST_INT8:
        lo = -128
        hi = 127
        name = "int8_t"
    elif st == ST_INT16:
        lo = -32768
        hi = 32767
        name = "int16_t"
    elif st == ST_INT32:
        lo = -2147483648
        hi = 2147483647
        name = "int"
    elif st == ST_UINT8:
        lo = -255
        hi = 255
        name = "uint8_t"
    if v_scalar_is_integral(v):
        var i = v_int(v)
        if i < lo or i > hi:
            raise Error(
                "value cannot be converted to type ", name, " without overflow"
            )
        return i
    # c10::overflows<To, double> (torch 2.11): no wrap for a float, a
    # value outside [lowest, max] -- or NaN / inf -- is refused, and one
    # inside truncates. uint8's lowest is 0; int64's max is 2**63 - 1.
    var f = v_f64(v)
    var lo_f = 0.0 if st == ST_UINT8 else Float64(lo)
    var ok = f >= lo_f and f <= Float64(hi)
    if st == ST_INT64:
        # c10::overflows compares with double(INT64_MAX), which is 2**63:
        # 2**63 itself passes, and the host static_cast gives INT64_MIN
        ok = f >= -9223372036854775808.0 and f <= 9223372036854775808.0
        if ok and f == 9223372036854775808.0:
            return Int.MIN
    if not ok:
        raise Error(
            "value cannot be converted to type ", name, " without overflow"
        )
    return Int(f)


def scalar_to_float(v: Value, st: Int32) raises -> Float64:
    """An ATen Scalar as floating dtype `st`'s value, `scalar_to_int`'s twin
    for a floating target (`Scalar::to<scalar_t>()`, c10's
    checked_convert): a finite value beyond the type's range raises as
    `c10::overflows` does, while inf and NaN pass. The result is the float64
    the caller narrows into `st` (a float64 target never overflows)."""
    var f = v_f64(v)
    if v.tag == TAG_SCALAR_INT or v.tag == TAG_INT:
        # an integer: one rounding into the target (int64 -> float for the
        # single-precision ones), not through a double
        f = int_to_component(v.a, st)
    if st != ST_FLOAT16 and st != ST_BFLOAT16 and st != ST_FLOAT32:
        return f
    var top = 3.4028234663852886e38  # float32
    var name = String("float")
    if st == ST_FLOAT16:
        top = 65504.0
        name = "c10::Half"
    elif st == ST_BFLOAT16:
        top = 3.3895313892515355e38
        name = "c10::BFloat16"
    var finite = f - f == 0.0  # inf - inf and NaN - NaN are NaN
    if finite and (f > top or f < -top):
        raise Error(
            "value cannot be converted to type ", name, " without overflow"
        )
    return f


def known_stype(st: Int32) -> Bool:
    return is_float_stype(st) or is_int_stype(st) or st == ST_BOOL


def _pw_int_rank(st: Int32) -> Int:
    if st == ST_INT8:
        return 1
    if st == ST_INT16:
        return 2
    if st == ST_INT32:
        return 3
    return 4


def _pw_float_rank(st: Int32) -> Int:
    if st == ST_FLOAT16 or st == ST_BFLOAT16:
        return 1
    if st == ST_FLOAT32:
        return 2
    return 3


def promote_types(a: Int32, b: Int32) raises -> Int32:
    """c10::promoteTypes over the dtypes the mojo device computes on."""
    if a == ST_UNDEFINED:
        return b
    if b == ST_UNDEFINED or a == b:
        return a
    if not known_stype(a) or not known_stype(b):
        unsupported(
            "type promotion of dtypes " + String(a) + " and " + String(b)
        )
    if a == ST_BOOL:
        return b
    if b == ST_BOOL:
        return a
    if is_float_stype(a) and is_float_stype(b):
        if (a == ST_FLOAT16 and b == ST_BFLOAT16) or (
            a == ST_BFLOAT16 and b == ST_FLOAT16
        ):
            return ST_FLOAT32
        return a if _pw_float_rank(a) > _pw_float_rank(b) else b
    if is_float_stype(a):
        return a
    if is_float_stype(b):
        return b
    # both integral
    if a == ST_UINT8 or b == ST_UINT8:
        var other = b if a == ST_UINT8 else a
        if other == ST_INT8:
            return ST_INT16
        return other
    return a if _pw_int_rank(a) > _pw_int_rank(b) else b


def combine_categories(higher: Int32, lower: Int32) raises -> Int32:
    """c10's combine_categories (the ResultTypeState fold)."""
    if is_float_stype(higher):
        return higher
    if higher == ST_BOOL or is_float_stype(lower):
        return promote_types(higher, lower)
    if higher != ST_UNDEFINED:
        return higher
    return lower


def can_cast(src: Int32, dst: Int32) -> Bool:
    """`c10::canCast` over real dtypes: a cast that does not lower the
    category (bool < integral < floating). What TensorIterator's
    `enforce_safe_casting_to_output` asks of an `out=` / in-place output."""
    var s = 2 if is_float_stype(src) else (0 if src == ST_BOOL else 1)
    var d = 2 if is_float_stype(dst) else (0 if dst == ST_BOOL else 1)
    return s <= d


def result_type(a: T, b: T) raises -> Int32:
    """`torch.result_type(a, b)` of two tensors: a dimensioned tensor ranks
    above a 0-dim one, which wins only when it is of a higher category
    (bool < integral < floating) -- `float16[3] + float32[]` is float16,
    `int64[3] + float32[]` float32 (at::native::update_result_type_state).
    """
    var dim = ST_UNDEFINED
    var zero = ST_UNDEFINED
    if a.rank == 0:
        zero = promote_types(zero, a.stype)
    else:
        dim = promote_types(dim, a.stype)
    if b.rank == 0:
        zero = promote_types(zero, b.stype)
    else:
        dim = promote_types(dim, b.stype)
    return combine_categories(dim, zero)


def binary_promotion(a: T, b: T) raises -> DType:
    """torch's promotion for a binary pair, restricted to what the
    broadcast-strided spec kernels cover: equal dtypes; bool with any
    castable dtype; int32/int64; float32 with float16/bfloat16; and
    float16<->bfloat16 (widens both sides to float32). Declines
    (`unsupported`) any other pair.

    Takes the tensors, not just their dtypes, so the bool+castable branch
    can check `is_cast_dtype_on` (float64 declines cleanly on an Apple GPU,
    where there is no host-fallback route for this promotion the way
    aten.to/copy_ have one).

    A caller casts each operand into the returned dtype with
    `cast_to(operand, torch_dtype(result))`, which already no-ops when an
    operand is already that dtype -- so, unlike the old `_binary_promotion`,
    this returns just the common dtype rather than a (cast lhs?, cast rhs?,
    dtype) triple; there is no separate cast-skipping fast path to expose.
    Ported from `_binary_promotion`.
    """
    var a_dtype = a.dtype
    var b_dtype = b.dtype
    if a_dtype == b_dtype:
        return a_dtype
    if a_dtype == DType.bool and is_cast_dtype_on(b_dtype, b):
        return b_dtype
    if b_dtype == DType.bool and is_cast_dtype_on(a_dtype, a):
        return a_dtype
    if a_dtype == DType.int32 and b_dtype == DType.int64:
        return DType.int64
    if b_dtype == DType.int32 and a_dtype == DType.int64:
        return DType.int64
    if a_dtype == DType.float32 and (
        b_dtype == DType.float16 or b_dtype == DType.bfloat16
    ):
        return DType.float32
    if b_dtype == DType.float32 and (
        a_dtype == DType.float16 or a_dtype == DType.bfloat16
    ):
        return DType.float32
    if (a_dtype == DType.float16 and b_dtype == DType.bfloat16) or (
        a_dtype == DType.bfloat16 and b_dtype == DType.float16
    ):
        return DType.float32
    if (
        (a_dtype == DType.float64 or b_dtype == DType.float64)
        and a_dtype.is_floating_point()
        and b_dtype.is_floating_point()
        and is_cast_dtype_on(DType.float64, a)
    ):
        # Rank-aware, as torch: a 0-d float64 (linalg's tolerances) meets a
        # dimensioned float32 in float32.
        return max_dtype(result_type(a, b))
    unsupported(
        "no dtype promotion for " + String(a_dtype) + " and " + String(b_dtype)
    )
    return a_dtype


def promoted_pair(a: T, b: T) raises -> Tuple[T, T]:
    """Same-dtype tensor pair following torch's promotion, materializing a
    cast side through `cast_to`.

    A deliberate SUBSET of `binary_promotion` (bool+castable, int32/int64
    only): `where`/`masked_fill` must keep declining mixed float widths
    rather than silently widening them here. Ported from `_promoted_pair`.
    The caller releases whichever of the pair has a handle (`.h`) different
    from the corresponding input -- that side was freshly materialized.
    """
    if a.dtype == b.dtype:
        return (a.copy(), b.copy())
    if a.dtype == DType.bool and is_cast_dtype_on(b.dtype, b):
        return (cast_to(a, torch_dtype(b.dtype)), b.copy())
    if b.dtype == DType.bool and is_cast_dtype_on(a.dtype, a):
        return (a.copy(), cast_to(b, torch_dtype(a.dtype)))
    if a.dtype == DType.int32 and b.dtype == DType.int64:
        return (cast_to(a, torch_dtype(DType.int64)), b.copy())
    if b.dtype == DType.int32 and a.dtype == DType.int64:
        return (a.copy(), cast_to(b, torch_dtype(DType.int64)))
    unsupported("mixed dtypes " + String(a.dtype) + " and " + String(b.dtype))
    return (a.copy(), b.copy())


def release_if_new(result: T, original: T):
    """Release `result` only when it is a fresh allocation distinct from
    `original`. `contiguous()`/`cast_to()` alias their input (returning it
    unchanged, via `T.copy()`) instead of allocating whenever the input
    already has the requested layout/dtype -- so a caller that wraps their
    result in `own()` unconditionally would release a handle it never
    allocated (an argument the caller only borrowed, e.g. an op's `self`).

    `abi.own_if_new(contiguous(t), t)` is the same rule as a scope guard and
    covers the raising paths too; reach for that one in new code."""
    if result.h != original.h:
        release(result.h)


def forward_args(args: Values, n: Int) -> List[Value]:
    """The first `n` argument records, copied: what an `out=` overload hands
    its functional overload through `call_op` (same schema prefix)."""
    var out = List[Value](capacity=max(n, 1))
    for i in range(n):
        out.append(args[unsafe_offset=i].copy())
    return out^


def store_out(mut dst: T, var src: T) raises:
    """Copy a FRESH functional result (owned, released here) into a caller's
    `out=` tensor: torch's `resize_output` then a strided copy.

    The result is computed before `dst` is touched, so an `out` that shares
    storage with an input is only resized after every input was read. The
    caller ran `check_out_as` on `dst` before computing."""
    var held = own(src^)
    assert_no_internal_overlap(dst)
    resize_out(dst, held.t.shape, held.t.rank)
    copy_strided_into(dst, held.t)


def is_channels_last_layout(t: T) -> Bool:
    """`suggest_memory_format() != Contiguous` for the dense layouts that
    matter here: a rank-4/5 tensor laid out channels-last (NHWC / NDHWC)
    that is not also plain contiguous."""
    if t.contig or (t.rank != 4 and t.rank != 5):
        return False
    var want = _channels_last_strides(t.shape, t.rank)
    for i in range(t.rank):
        if t.dim(i) > 1 and t.stride(i) != want[MAX_RANK - t.rank + i]:
            return False
    return True


def like_layout(var r: Owned, like: T, any_dense: Bool = False) raises -> Owned:
    """A fresh contiguous result `r` re-laid-out like `like` (the op's
    input): channels-last when `like` is (`suggest_memory_format`), or --
    `any_dense` -- with `like`'s own strides whenever `like` is dense
    (`empty_like`). Same values; `r` is released. A no-op, no copy, for a
    contiguous `like`."""
    if like.contig or not r.t.same_shape(like):
        return r^
    var strides: IndexList[MAX_RANK]
    if any_dense and is_dense(like.shape, like.strides, like.rank):
        strides = like.strides
    elif is_channels_last_layout(like):
        strides = _channels_last_strides(like.shape, like.rank)
    else:
        return r^
    var same = True
    for i in range(like.rank):
        var k = MAX_RANK - like.rank + i
        if like.shape[k] > 1 and r.t.strides[k] != strides[k]:
            same = False
    if same:
        return r^  # already laid out so: nothing to copy
    var out = own(
        new_strided(like.shape, strides, like.rank, r.t.stype, r.t.device)
    )
    copy_strided_into(out.t, r.t)
    _ = r^
    return out^


# ---------------------------------------------------------------------------
# complex dtypes: casts to and from them, and the launch of the complex
# kernel family (tmb/kernels/complex), shared by the copy paths (core,
# data_movement) and the complex ops (tmb/ops/complex.mojo).
# ---------------------------------------------------------------------------

comptime CX_FULL = 0
comptime CX_DEVICE = 1
comptime CX_HOST = 2


def cx_launch(
    kind: StaticString,
    out_addr: Int,
    a_addr: Int,
    b_addr: Int,
    mode_a: Int,
    mode_b: Int,
    a_re: Float64,
    a_im: Float64,
    b_re: Float64,
    b_im: Float64,
    numel: Int,
    component: DType,
    device: Int,
    p_re: Float64 = 1.0,
    p_im: Float64 = 0.0,
) raises:
    """One elementwise kernel of the complex family over `numel` complex
    elements (header of tmb/kernels/complex/entry.mojo): flat operands only,
    each full (dense, the output's extent), one device element or a host
    value."""
    if numel == 0:
        return
    var ctx = ctx_for(device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("complex", kind)
    call.arg_dtype(0, component)
    call.int(out_addr)
    call.int(a_addr)
    call.int(b_addr)
    call.int(mode_a | (mode_b << 2))
    call.f64(a_re)
    call.f64(a_im)
    call.f64(b_re)
    call.f64(b_im)
    call.int(numel)
    call.f64(p_re)
    call.f64(p_im)
    call.int(cp)
    call.run()
    _ = ctx


def cx_unary_into(kind: StaticString, dst: T, src: T) raises:
    """dst = kind(src) for two DENSE buffers of `src.numel` elements (src
    complex; dst complex, real or bool by kind)."""
    cx_launch(
        kind,
        dst.ptr,
        src.ptr,
        0,
        CX_FULL,
        CX_FULL,
        0.0,
        0.0,
        0.0,
        0.0,
        src.numel,
        src.dtype,
        src.device,
    )


def warn_complex_to_real() raises:
    """torch's `TORCH_WARN_ONCE("Casting complex values to real discards the
    imaginary part")` (Copy.cpp's copy_impl), raised by having CPU torch run
    the one statement that emits it: a 1-element complex-to-float host copy.
    Its once-per-process state is torch's own, as on CUDA."""
    var shape = IndexList[MAX_RANK](1)
    var src = own(cpu_empty(shape, 1, ST_COMPLEX64))
    var dst = own(cpu_empty(shape, 1, ST_FLOAT32))
    _ = call_op(
        "aten::copy_",
        "",
        [
            tensor_arg(dst.t),
            tensor_arg(src.t),
            bool_arg(False),
        ],
        1,
    )
    _ = src^
    _ = dst^


def host_convert(t: T, stype: Int32) raises -> T:
    """`t.to(stype)` through CPU torch (a dense download, torch's CPU copy_,
    one upload): the conversions no device kernel has (the unsigned wide
    integers, float64 on Metal), with torch's exact semantics."""
    var dense = own_if_new(contiguous(t), t)
    var host_src = own(cpu_empty(t.shape, t.rank, t.stype))
    copy_to_host(
        ctx_for(t.device), dense.t.ptr, host_src.t.ptr, t.numel * t.itemsize
    )
    _ = dense^
    var host_dst = own(cpu_empty(t.shape, t.rank, stype))
    _ = call_op(
        "aten::copy_",
        "",
        [tensor_arg(host_dst.t), tensor_arg(host_src.t), bool_arg(False)],
        1,
    )
    _ = host_src^
    var out = own(new_like_dtype(t, stype))
    copy_from_host(
        t.device,
        ctx_for(t.device),
        out.t.ptr,
        host_dst.t.ptr,
        out.t.numel * out.t.itemsize,
    )
    _ = host_dst^
    return out.take()


def cast_complex(t: T, stype: Int32) raises -> T:
    """`t.to(stype)` on t's device where `t` or `stype` is complex: a
    contiguous result (t itself when the dtype is unchanged). torch's
    conversions: real -> complex sets the imaginary part to 0; complex ->
    real keeps the real part (and warns, as copy_ does); complex -> bool is
    `re != 0 || im != 0`; complex64 <-> complex128 converts both parts.
    `t` carries no conj bit (the callers resolve it)."""
    if t.stype == stype:
        return t.copy()
    var src = own_if_new(contiguous(t), t)
    if src.t.is_complex() and is_complex_stype(stype):
        if not is_cast_dtype_on(DType.float64, src.t):
            var converted = own(host_convert(src.t, stype))
            _ = src^
            return converted.take()
        var both = own(new_like_dtype(src.t, stype))
        cast_into(both.t.pairs(), src.t.pairs())
        _ = src^
        return both.take()
    if is_complex_stype(stype):
        # real -> complex: zero the pairs, then the real parts in place
        var part = real_of(stype)
        if src.t.stype != part and not (
            is_cast_dtype_on(src.t.dtype, src.t)
            and is_cast_dtype_on(max_dtype(part), src.t)
        ):
            var converted = own(host_convert(src.t, stype))
            _ = src^
            return converted.take()
        var made = own(new_like_dtype(src.t, stype))
        fill_value(made.t, 0.0)
        var re = own_if_new(cast_to(src.t, part), src.t)
        copy_strided_into(made.t.component(0), re.t)
        _ = re^
        _ = src^
        return made.take()
    # complex -> real
    if stype == ST_BOOL:
        var truth = own(new_like_dtype(src.t, ST_BOOL))
        cx_unary_into("nonzero", truth.t, src.t)
        _ = src^
        return truth.take()
    if stype != real_of(src.t.stype) and not (
        is_cast_dtype_on(src.t.dtype, src.t)
        and is_cast_dtype_on(max_dtype(stype), src.t)
    ):
        var converted = own(host_convert(src.t, stype))
        _ = src^
        return converted.take()
    warn_complex_to_real()
    var re = own(contiguous(src.t.component(0)))
    _ = src^
    if stype == re.t.stype:
        return re.take()
    var out = own(cast_to(re.t, stype))
    _ = re^
    return out.take()


def conj_copy(t: T) raises -> T:
    """A fresh contiguous complex tensor holding conj(t's stored values):
    what resolving a conj bit takes (torch's `conj_physical` of the raw
    data)."""
    var dense = own_if_new(contiguous(t), t)
    var out = own(new_like(dense.t))
    cx_unary_into("conj", out.t, dense.t)
    _ = dense^
    return out.take()


def scalar_type_token(st: Int32) -> String:
    """The C++ type a checked_convert overflow error names (c10's
    AT_FORALL_SCALAR_TYPES spelling)."""
    if st == ST_FLOAT32:
        return "float"
    if st == ST_FLOAT64:
        return "double"
    if st == ST_FLOAT16:
        return "c10::Half"
    if st == ST_BFLOAT16:
        return "c10::BFloat16"
    if st == ST_INT64:
        return "int64_t"
    if st == ST_INT32:
        return "int"
    if st == ST_INT16:
        return "int16_t"
    if st == ST_INT8:
        return "int8_t"
    if st == ST_UINT8:
        return "uint8_t"
    if st == ST_UINT16:
        return "uint16_t"
    if st == ST_UINT32:
        return "uint32_t"
    if st == ST_UINT64:
        return "uint64_t"
    if st == ST_COMPLEX64:
        return "c10::complex<float>"
    return "c10::complex<double>"


def check_real_scalar_fits(st: Int32, v: Value) raises:
    """c10::checked_convert of a Scalar record (`v`) to the REAL dtype `st`
    (c10::overflows): bool takes anything; a complex value with a nonzero
    imaginary part fits no other real dtype; a float dtype refuses a finite
    value past its range (inf / NaN pass); an integer dtype refuses NaN /
    inf and a double past its range, and an integer past it (an unsigned
    dtype also takes -x for x within it, as C++'s wrap)."""
    if st == ST_BOOL or is_complex_stype(st):
        return
    var known = (
        st == ST_FLOAT16
        or st == ST_BFLOAT16
        or st == ST_FLOAT32
        or st == ST_FLOAT64
        or st == ST_UINT8
        or st == ST_UINT16
        or st == ST_UINT32
        or st == ST_UINT64
        or st == ST_INT8
        or st == ST_INT16
        or st == ST_INT32
        or st == ST_INT64
    )
    if not known:
        return  # (float8 and the rest: the fill kernels decide)
    var bad = False
    var is_int_value = v.tag == TAG_SCALAR_INT or v.tag == TAG_INT
    if v.tag == TAG_SCALAR_BOOL or v.tag == TAG_BOOL:
        return
    var re: Float64
    if is_int_value:
        re = Float64(v.a)
    else:
        var c = v_complex(v)
        re = c[0]
        if c[1] != 0.0:
            bad = True
    var finite = re - re == 0.0
    if st == ST_FLOAT16 or st == ST_BFLOAT16 or st == ST_FLOAT32:
        var fmax = 65504.0 if st == ST_FLOAT16 else (
            3.3895313892515355e38 if st
            == ST_BFLOAT16 else 3.4028234663852886e38
        )
        if finite and abs(re) > fmax:
            bad = True
    elif st != ST_FLOAT64:
        var lo: Float64
        var hi: Float64
        var unsigned = (
            st == ST_UINT8
            or st == ST_UINT16
            or st == ST_UINT32
            or st == ST_UINT64
        )
        if st == ST_UINT8:
            lo = 0.0
            hi = 255.0
        elif st == ST_UINT16:
            lo = 0.0
            hi = 65535.0
        elif st == ST_UINT32:
            lo = 0.0
            hi = 4294967295.0
        elif st == ST_UINT64:
            lo = 0.0
            hi = 18446744073709551615.0
        elif st == ST_INT8:
            lo = -128.0
            hi = 127.0
        elif st == ST_INT16:
            lo = -32768.0
            hi = 32767.0
        elif st == ST_INT32:
            lo = -2147483648.0
            hi = 2147483647.0
        else:
            lo = -9223372036854775808.0
            hi = 9223372036854775807.0
        if is_int_value:
            var iv = v.a
            if st == ST_UINT64:
                pass  # every int64 fits, a negative one by C++'s wrap
            elif unsigned:
                # -iv as an unsigned magnitude (INT64_MIN has no int64 -iv)
                var imax = UInt64(hi)
                var mag = UInt64(-(iv + 1)) + 1 if iv < 0 else UInt64(iv)
                bad = bad or mag > imax
            elif st != ST_INT64:
                bad = bad or re < lo or re > hi
        else:
            bad = bad or not finite or re < lo or re > hi
    if bad:
        raise Error(
            "value cannot be converted to type ",
            scalar_type_token(st),
            " without overflow",
        )


def uint64_cross_item(value: T, dst: Int32 = -1) raises -> Value:
    """`fill_item(value, dst, False)`: a uint64 value read for a fill on
    another device."""
    return fill_item(value, dst, False)


def fill_item(value: T, dst: Int32, same_device: Bool) raises -> Value:
    """`value.item()` as a fill / masked_fill / index_put value for a
    destination of dtype `dst`, the Scalar CUDA would convert. A uint64
    value is read unsigned (the call boundary has no unsigned Scalar):
    below 2**63 an int Scalar, which converts like CUDA's (an integer
    into a float rounds once); from 2**63 on, into a float or complex dtype
    the value itself, rounded once into float32 for the single-precision
    dtypes; into bool its truth; into uint64 the same 64 bits; into any
    other integer the overflow error. From another device (`same_device`
    False) a value of 2**63 or more is declined instead, except into uint64.
    """
    if value.stype != ST_UINT64:
        var r = call_op("aten::_local_scalar_dense", "", [tensor_arg(value)], 1)
        return r[0].copy()
    var bits: Int64
    if value.on_cpu():
        bits = Pointer[Int64, MutUntrackedOrigin](
            unsafe_from_address=value.ptr
        )[]
    else:
        var r = call_op("aten::_local_scalar_dense", "", [tensor_arg(value)], 1)
        bits = r[0].a
    if bits >= 0 or dst == ST_UINT64:
        return Value(TAG_SCALAR_INT, 0, bits, 0)
    if not same_device:
        unsupported("a uint64 value of 2**63 or more from another device")
    var u = UInt64(bits)
    if dst == ST_BOOL:
        return Value(TAG_SCALAR_BOOL, 0, 1, 0)
    if dst == ST_FLOAT64 or dst == ST_COMPLEX128:
        return Value(TAG_SCALAR_DOUBLE, 0, f64_bits(Float64(u)), 0)
    if (
        dst == ST_FLOAT32
        or dst == ST_FLOAT16
        or dst == ST_BFLOAT16
        or dst == ST_COMPLEX64
    ):
        var f = Float64(u.cast[DType.float32]())
        return Value(TAG_SCALAR_DOUBLE, 0, f64_bits(f), 0)
    raise Error(
        "value cannot be converted to type ",
        scalar_type_token(dst),
        " without overflow",
    )


def real_fill_value(st: Int32, v: Value) raises -> Value:
    """The Scalar record `v`, already range-checked for the real dtype `st`
    (`check_real_scalar_fits`), as the record the fill kernels take: the one
    conversion every Scalar entry point into a real tensor shares (fill_,
    scalar_tensor, full, masked_fill, item assignment).
    - A complex Scalar (its imaginary part is zero, or the check raised)
      becomes its real part as a double, then converts as that double does.
      A bool destination keeps the record: its truth reads either part.
    - Into uint64, a double of 2**63 or more becomes an int record holding
      the 64 bits of C++'s host `static_cast<uint64_t>` (the fill kernels go
      through int64): exact below 2**64, and 0 for 2**64 itself, which
      c10::overflows lets through (it compares with double(UINT64_MAX)).
    Anything else is returned unchanged."""
    if st == ST_BOOL or is_complex_stype(st):
        return v.copy()
    var r = v.copy()
    if r.tag == TAG_COMPLEX:
        r = Value(TAG_SCALAR_DOUBLE, 0, f64_bits(v_complex(v)[0]), 0)
    if st == ST_UINT64 and (r.tag == TAG_SCALAR_DOUBLE or r.tag == TAG_DOUBLE):
        var d = v_complex(r)[0]
        if d >= 9223372036854775808.0:
            var bits = Int64(0)
            if d < 18446744073709551616.0:
                bits = Int64(
                    Scalar[DType.uint64](UInt64(d)).cast[DType.int64]()
                )
            return Value(TAG_SCALAR_INT, 0, bits, 0)
    return r^


def scalar_type_name(st: Int32) -> String:
    """c10::toString(ScalarType): the name ATen's dispatch errors print
    (`"index_put" not implemented for 'UInt64'`)."""
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
    if st == ST_UINT16:
        return "UInt16"
    if st == ST_UINT32:
        return "UInt32"
    if st == ST_UINT64:
        return "UInt64"
    if st == ST_COMPLEX64:
        return "ComplexFloat"
    if st == ST_COMPLEX128:
        return "ComplexDouble"
    return String(st)


def is_wide_unsigned(st: Int32) -> Bool:
    """uint16 / uint32 / uint64: the dtypes CUDA's masked_fill, index_fill
    and index_put kernels are not instantiated for."""
    return st == ST_UINT16 or st == ST_UINT32 or st == ST_UINT64


def check_scalar_fits(st: Int32, re: Float64, im: Float64) raises:
    """c10::checked_convert of a Scalar to complex64: a finite part out of
    float's range raises, as CUDA's Scalar conversions do (complex128 holds
    any double)."""
    if st != ST_COMPLEX64:
        return
    comptime FMAX = 3.4028234663852886e38
    # (x - x == 0: finite; an inf or NaN converts as is)
    var bad_re = re - re == 0.0 and abs(re) > FMAX
    var bad_im = im - im == 0.0 and abs(im) > FMAX
    if bad_re or bad_im:
        raise Error(
            "value cannot be converted to type c10::complex<float> without"
            " overflow"
        )


def int_to_component(i: Int64, st: Int32) -> Float64:
    """An integer as dtype `st`'s real component in ONE rounding: int64 ->
    float for complex64 / float32 / half / bfloat16 (the latter two round
    on from that float, as c10 does), int64 -> double otherwise (never a
    double first for a float32 target)."""
    if (
        st == ST_COMPLEX64
        or st == ST_FLOAT32
        or st == ST_FLOAT16
        or st == ST_BFLOAT16
    ):
        return Float64(Scalar[DType.int64](i).cast[DType.float32]())
    return Float64(i)


def uint_to_component(u: UInt64, st: Int32) -> Float64:
    """`int_to_component` of an unsigned 64-bit integer."""
    if (
        st == ST_COMPLEX64
        or st == ST_FLOAT32
        or st == ST_FLOAT16
        or st == ST_BFLOAT16
    ):
        return Float64(Scalar[DType.uint64](u).cast[DType.float32]())
    return Float64(u)


def scalar_parts(v: Value, st: Int32) raises -> Tuple[Float64, Float64]:
    """A Scalar record's (re, im) as dtype `st` will hold them: an integer
    rounded once into the component type (`int_to_component`), any other
    record as `v_complex` reads it."""
    if v.tag == TAG_SCALAR_INT or v.tag == TAG_INT:
        return (int_to_component(v.a, st), 0.0)
    return v_complex(v)


def fill_complex(t: T, v: Value) raises:
    """fill_ of a complex tensor: the two parts of the Scalar (a real one has
    a zero imaginary part) into the two strided component views; one memset
    for an all-zero value."""
    var c = scalar_parts(v, t.stype)  # an integer: one rounding
    var re = c[0]
    var im = c[1]
    if t.is_conj():
        im = -im
    if t.numel == 0:
        return
    if f64_bits(re) == 0 and f64_bits(im) == 0:
        if t.contig:
            fill_value(t, 0.0)
        else:
            fill_value(t.pairs(), 0.0)
        return
    fill_value(t.component(0), re)
    fill_value(t.component(1), im)
