"""Helpers every op group shares: materializing a contiguous copy, strided
copies and fills, dtype casts (through the memory_ops / data_movement_ops
families on the tensor's current stream), scalar embedding and binary type
promotion, `out=` resizing, and the Philox reservation more than one group
needs. Calling another aten op through the real dispatcher is `abi.call_op`,
re-exported here."""
from std.ffi import external_call
from std.utils import IndexList

from abi import (
    T,
    Value,
    check,
    Results,
    call_op,
    call_op_raw,
    contiguous_strides,
    dtype_code,
    dtype_itemsize,
    dtype_name,
    f64_bits,
    max_dtype,
    new_like,
    new_like_dtype,
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
)
from device import ctx_for, ctx_ptr, dev, memset_bytes, memset_typed
from kernels import KernelCall
from op_utils import MAX_RANK

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
    (memory_ops CopyStrided: element-size dispatch, rank <= MAX_RANK).

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
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("memory_ops", "CopyStrided")
    call.int(dst.ptr)
    call.int(src.ptr)
    call.tuple(_padded(dst.shape))
    call.tuple(_padded(dst.strides))
    call.tuple(_padded(src.strides))
    call.int(dst.itemsize)
    call.int(cp)
    call.run()
    _ = ctx


def device_str(t: T) -> String:
    """`c10::Device::str()`: the lower-case device-type name plus an index
    when the tensor carries one. PrivateUse1 prints under the name torch was
    renamed to (`rename_privateuse1_backend("mojo")`)."""
    if t.on_mojo():
        return String("mojo:") + String(t.device)
    if t.on_cpu():
        return String("cpu")
    return String("device type ") + String(t.device_type)


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
    if dest.stype != like.stype:
        raise Error(
            "Expected out tensor to have dtype ",
            dtype_name(like.stype),
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


def resize_out(mut t: T, shape: IndexList[MAX_RANK], rank: Int) raises:
    """torch's `resize_output` for a caller's `out=` tensor, in place.

    A backend with no `aten::resize_` kernel of its own gets no resize
    before dispatch, so every `out=` op here does this itself. Two halves,
    both of `at::native::resize_impl`:

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
    if nbytes > t.storage_nbytes():
        check(
            external_call["tmb_storage_resize", Int32](t.h, Int64(nbytes)),
            "tmb_storage_resize",
        )
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


def _fill(t: T, s: FillScalar) raises:
    """A memset when contiguous on an accelerator, else the strided fill
    kernel. On the MAX CPU device a memset is not ordered against kernel
    launches (measured: a kernel reading a just-filled buffer saw stale
    memory 4 times in 50), so that device always fills with the kernel."""
    if t.numel == 0:
        return
    if t.contig and not dev(t.device)[].is_cpu:
        _fill_contiguous(t, s)
        return
    if s.integral and t.itemsize == 8 and not t.dtype.is_floating_point():
        # The StridedFill kernel narrows a Float64 into the destination
        # dtype, which cannot carry a 64-bit integer past 2**53. Fill a dense
        # buffer exactly (memset) and lay that out instead -- except on the
        # MAX CPU device, where a memset is not ordered against the strided
        # copy that would read it.
        if abs(s.i) > _MAX_EXACT_INT:
            if dev(t.device)[].is_cpu:
                unsupported(
                    "filling a strided 64-bit integer tensor on the MAX CPU"
                    " device with a magnitude above 2**53"
                )
            var dense = own(new_like(t))
            _fill_contiguous(dense.t, s)
            copy_strided_into(t, dense.t)
            return
    var ctx = ctx_for(t.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("memory_ops", "StridedFill")
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


def cast_into(dst: T, src: T) raises:
    """dst = src.to(dst.dtype) for a contiguous dst (data_movement_ops CastSpec).
    """
    if src.numel == 0:
        return
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("data_movement_ops", "CastSpec")
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


def philox_reserve(
    generator: Int, device: Int, increment: Int
) raises -> Tuple[UInt64, UInt64]:
    """Atomically reserve `increment` counters of a device's (generator=0)
    or an explicit generator's Philox stream: returns `(seed, base_offset)`
    as they stood *before* the reservation (tmb_philox_reserve,
    docs/native_backend.md). Generic device-runtime plumbing any RNG op of
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


def _is_cast_dtype(dtype: DType) -> Bool:
    """Dtypes `binary_promotion`/`promoted_pair` can materialize a cast into
    (matches the old `_CAST_DTYPES`)."""
    return (
        dtype == DType.float32
        or dtype == DType.float16
        or dtype == DType.bfloat16
        or dtype == DType.int64
        or dtype == DType.int32
        or dtype == DType.uint8
        or dtype == DType.bool
    )


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


def binary_promotion(a_dtype: DType, b_dtype: DType) raises -> DType:
    """torch's promotion for a binary pair, restricted to what the
    broadcast-strided spec kernels cover: equal dtypes; bool with any
    castable dtype; int32/int64; float32 with float16/bfloat16; and
    float16<->bfloat16 (widens both sides to float32). Declines
    (`unsupported`) any other pair.

    A caller casts each operand into the returned dtype with
    `cast_to(operand, torch_dtype(result))`, which already no-ops when an
    operand is already that dtype -- so, unlike the old `_binary_promotion`,
    this returns just the common dtype rather than a (cast lhs?, cast rhs?,
    dtype) triple; there is no separate cast-skipping fast path to expose.
    Ported from `_binary_promotion`.
    """
    if a_dtype == b_dtype:
        return a_dtype
    if a_dtype == DType.bool and _is_cast_dtype(b_dtype):
        return b_dtype
    if b_dtype == DType.bool and _is_cast_dtype(a_dtype):
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
    if a.dtype == DType.bool and _is_cast_dtype(b.dtype):
        return (cast_to(a, torch_dtype(b.dtype)), b.copy())
    if b.dtype == DType.bool and _is_cast_dtype(a.dtype):
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
