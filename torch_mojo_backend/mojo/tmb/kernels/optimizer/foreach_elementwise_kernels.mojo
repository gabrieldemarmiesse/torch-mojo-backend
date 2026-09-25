"""Fixed-arity batched FP32 foreach elementwise kernels: the APPLE half.

Every op here is also served by the single descriptor-batched body in
`foreach_batched_kernels`, which is the route on CUDA and ROCm and the one to
change when the element math changes. These exist because Metal cannot take
that route: Metal only translates pointer-typed kernel *arguments* into valid
GPU addresses, and a raw address smuggled as data (a descriptor struct or an
Int) reads back zeros — while per-tensor launches are dispatch bound for
optimizer-sized tensor lists.

So each kernel here takes a fixed arity of FOREACH_EW_SLOTS real pointer
arguments per tensor list plus per-slot chunk offsets/lengths as plain data
(non-pointer data passes fine, in fixed-width form — see `_SlotInts`), and one
launch covers up to FOREACH_EW_SLOTS tensors, grid-strided over the
concatenation of their fixed-size chunks.
Unused slots repeat a valid pointer with length zero and are never
dereferenced. `foreach_ew_enqueue` does the regrouping from the shared
descriptors.

Element math deliberately mirrors the scalar eager kernels these ops fall
back to (`elementwise`/`logic`), keeping the batched path
bit-compatible with the sequential per-tensor decomposition.
"""

from std.collections import InlineArray
from std.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.math import min
from std.sys.info import has_accelerator
from std.utils.numerics import isfinite

from tmb.kernels.common.op_utils import _enqueue_cached, ieee_sqrt


comptime FOREACH_EW_SLOTS = 8
comptime FOREACH_EW_CHUNK = 65_536  # elements per chunk; a multiple of _VEC
comptime FOREACH_EW_THREADS = 256
comptime _VEC = 4

comptime FES_MUL = 0
comptime FES_ADD = 1
comptime FES_DIV = 2

comptime FEA_ADDCMUL = 0
comptime FEA_ADDCDIV = 1

comptime _MutPtr = Pointer[Scalar[DType.float32], MutAnyOrigin]
comptime _ImmutPtr = Pointer[Scalar[DType.float32], ImmutAnyOrigin]

# Per-slot chunk offsets and lengths as they cross the launch ABI. `Int` is
# not device-passable (its width differs between host and device, and Metal's
# device `Int` is not the host's), and an `InlineArray` encodes element-wise,
# so the array element type has to be fixed-width too. Kernels convert back
# to `Int` at the use site and all index math stays in `Int`.
comptime _SlotInts = InlineArray[Int64, FOREACH_EW_SLOTS]


@always_inline
def _addr_ptr(addr: Int) -> _MutPtr:
    return Pointer[Scalar[DType.float32], MutUntrackedOrigin](
        unsafe_from_address=addr
    ).as_unsafe_any_origin()


@always_inline
def _slot_ints(values: InlineArray[Int, FOREACH_EW_SLOTS]) -> _SlotInts:
    """Widen a host-side per-slot `Int` array to the launch ABI type."""
    var widened = _SlotInts(fill=Int64(0))
    for slot in range(FOREACH_EW_SLOTS):
        widened[slot] = Int64(values[slot])
    return widened^


@always_inline
def _no_fuse[
    width: Int
](x: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """Round `x` exactly once: a volatile stack round-trip the Metal
    compiler cannot contract into a following add/sub as an fma.

    The lerp fallback composes separate mul and add kernels, so its product
    is rounded before the add; without this barrier Metal fuses
    `a + w * d` into fma and the batched path drifts by one ulp.
    (`llvm.arithmetic.fence` aborts Metal pipeline creation, so the barrier
    is spelled as volatile memory traffic instead.)
    """
    var tmp = x
    var pointer = Pointer(to=tmp).unsafe_bitcast[Scalar[DType.float32]]()
    pointer.unsafe_store[volatile=True](0, x)
    return pointer.unsafe_load[width=width, volatile=True](0)


@always_inline
def _slot_begin(chunk_ends: _SlotInts, chunk: Int) -> Tuple[Int, Int]:
    """(slot, element offset in that slot's tensor) for a global chunk id.

    Empty slots occupy zero chunks (their chunk_end repeats the previous
    one), so the scan naturally steps over them.
    """
    var slot = 0
    while slot + 1 < FOREACH_EW_SLOTS and chunk >= Int(chunk_ends[slot]):
        slot += 1
    var first_chunk = 0
    if slot != 0:
        first_chunk = Int(chunk_ends[slot - 1])
    return slot, (chunk - first_chunk) * FOREACH_EW_CHUNK


# Metal constraint (verified like the descriptor case in
# foreach_clip_kernels.mojo): copying pointer-typed kernel *arguments* into an
# InlineArray and indexing it dynamically miscompiles — stores through the
# selected pointer are silently dropped. A branch chain over the original
# arguments keeps every access on the translated argument values.


@always_inline
def _pick_mut(
    slot: Int,
    p0: _MutPtr,
    p1: _MutPtr,
    p2: _MutPtr,
    p3: _MutPtr,
    p4: _MutPtr,
    p5: _MutPtr,
    p6: _MutPtr,
    p7: _MutPtr,
) -> _MutPtr:
    var selected = p0
    if slot == 1:
        selected = p1
    elif slot == 2:
        selected = p2
    elif slot == 3:
        selected = p3
    elif slot == 4:
        selected = p4
    elif slot == 5:
        selected = p5
    elif slot == 6:
        selected = p6
    elif slot == 7:
        selected = p7
    return selected


@always_inline
def _pick_immut(
    slot: Int,
    p0: _ImmutPtr,
    p1: _ImmutPtr,
    p2: _ImmutPtr,
    p3: _ImmutPtr,
    p4: _ImmutPtr,
    p5: _ImmutPtr,
    p6: _ImmutPtr,
    p7: _ImmutPtr,
) -> _ImmutPtr:
    var selected = p0
    if slot == 1:
        selected = p1
    elif slot == 2:
        selected = p2
    elif slot == 3:
        selected = p3
    elif slot == 4:
        selected = p4
    elif slot == 5:
        selected = p5
    elif slot == 6:
        selected = p6
    elif slot == 7:
        selected = p7
    return selected


def _foreach_scalar_kernel[
    op_code: Int
](
    p0: _MutPtr,
    p1: _MutPtr,
    p2: _MutPtr,
    p3: _MutPtr,
    p4: _MutPtr,
    p5: _MutPtr,
    p6: _MutPtr,
    p7: _MutPtr,
    chunk_ends: _SlotInts,
    numels: _SlotInts,
    scalars: InlineArray[Float32, FOREACH_EW_SLOTS],
):
    """In-place x = x (op) scalar[slot], one scalar per slot.

    Matches `elementwise._scalar_elementwise` (add/mul) and the DivSpec
    binary kernel (`a / b`) element math exactly.
    """
    var slot, begin = _slot_begin(chunk_ends, Int(block_idx.x))
    var end = min(begin + FOREACH_EW_CHUNK, Int(numels[slot]))
    var values = _pick_mut(slot, p0, p1, p2, p3, p4, p5, p6, p7)
    var scalar = scalars[slot]

    var index = begin + Int(thread_idx.x) * _VEC
    var stride = FOREACH_EW_THREADS * _VEC
    while index + _VEC <= end:
        var value = values.unsafe_load[width=_VEC, alignment=4](index)
        comptime if op_code == FES_MUL:
            value = value * scalar
        comptime if op_code == FES_ADD:
            value = value + scalar
        comptime if op_code == FES_DIV:
            value = value / scalar
        values.unsafe_store[width=_VEC, alignment=4](index, value)
        index += stride

    # The chunk size is divisible by _VEC, so only a tensor's last chunk can
    # need this scalar tail.
    index = begin + ((end - begin) // _VEC) * _VEC + Int(thread_idx.x)
    while index < end:
        var value = values[unsafe_offset=index]
        comptime if op_code == FES_MUL:
            value = value * scalar
        comptime if op_code == FES_ADD:
            value = value + scalar
        comptime if op_code == FES_DIV:
            value = value / scalar
        values[unsafe_offset=index] = value
        index += FOREACH_EW_THREADS


def _foreach_mul_tensor_kernel(
    p0: _MutPtr,
    p1: _MutPtr,
    p2: _MutPtr,
    p3: _MutPtr,
    p4: _MutPtr,
    p5: _MutPtr,
    p6: _MutPtr,
    p7: _MutPtr,
    scalar_ptr: _ImmutPtr,
    chunk_ends: _SlotInts,
    numels: _SlotInts,
):
    """In-place x = x * scalar[0], the scalar a 0-d device tensor.

    `aten::_foreach_mul_.Tensor`; element math matches the `mul_.Tensor`
    sequential fallback.
    """
    var scalar = scalar_ptr[unsafe_offset=0]
    var slot, begin = _slot_begin(chunk_ends, Int(block_idx.x))
    var end = min(begin + FOREACH_EW_CHUNK, Int(numels[slot]))
    var values = _pick_mut(slot, p0, p1, p2, p3, p4, p5, p6, p7)

    var index = begin + Int(thread_idx.x) * _VEC
    var stride = FOREACH_EW_THREADS * _VEC
    while index + _VEC <= end:
        var value = values.unsafe_load[width=_VEC, alignment=4](index)
        values.unsafe_store[width=_VEC, alignment=4](index, value * scalar)
        index += stride

    # The chunk size is divisible by _VEC, so only a tensor's last chunk can
    # need this scalar tail.
    index = begin + ((end - begin) // _VEC) * _VEC + Int(thread_idx.x)
    while index < end:
        values[unsafe_offset=index] = values[unsafe_offset=index] * scalar
        index += FOREACH_EW_THREADS


def _foreach_nonfinite_unscale_kernel(
    p0: _MutPtr,
    p1: _MutPtr,
    p2: _MutPtr,
    p3: _MutPtr,
    p4: _MutPtr,
    p5: _MutPtr,
    p6: _MutPtr,
    p7: _MutPtr,
    inv_scale_ptr: _ImmutPtr,
    found_inf_ptr: _MutPtr,
    chunk_ends: _SlotInts,
    numels: _SlotInts,
):
    """`aten::_amp_foreach_non_finite_check_and_unscale_`: in-place
    x = x * inv_scale[0] (skipped when it is exactly 1), found_inf[0] = 1 if
    any x is non-finite. Every writer stores the same 1: no atomic."""
    var inv_scale = inv_scale_ptr[unsafe_offset=0]
    var slot, begin = _slot_begin(chunk_ends, Int(block_idx.x))
    var end = min(begin + FOREACH_EW_CHUNK, Int(numels[slot]))
    var values = _pick_mut(slot, p0, p1, p2, p3, p4, p5, p6, p7)

    var index = begin + Int(thread_idx.x) * _VEC
    var stride = FOREACH_EW_THREADS * _VEC
    while index + _VEC <= end:
        var value = values.unsafe_load[width=_VEC, alignment=4](index)
        if not isfinite(value).reduce_and():
            found_inf_ptr[unsafe_offset=0] = 1.0
        if inv_scale != 1.0:
            value = value * inv_scale
        values.unsafe_store[width=_VEC, alignment=4](index, value)
        index += stride

    # The chunk size is divisible by _VEC, so only a tensor's last chunk can
    # need this scalar tail.
    index = begin + ((end - begin) // _VEC) * _VEC + Int(thread_idx.x)
    while index < end:
        var value = values[unsafe_offset=index]
        if not isfinite(value):
            found_inf_ptr[unsafe_offset=0] = 1.0
        if inv_scale != 1.0:
            value = value * inv_scale
        values[unsafe_offset=index] = value
        index += FOREACH_EW_THREADS


def _foreach_lerp_kernel(
    a0: _MutPtr,
    a1: _MutPtr,
    a2: _MutPtr,
    a3: _MutPtr,
    a4: _MutPtr,
    a5: _MutPtr,
    a6: _MutPtr,
    a7: _MutPtr,
    e0: _ImmutPtr,
    e1: _ImmutPtr,
    e2: _ImmutPtr,
    e3: _ImmutPtr,
    e4: _ImmutPtr,
    e5: _ImmutPtr,
    e6: _ImmutPtr,
    e7: _ImmutPtr,
    chunk_ends: _SlotInts,
    numels: _SlotInts,
    weight: Float32,
    one_minus_weight: Float32,
    low_branch_arg: Int64,
):
    """In-place scalar lerp: self = self + weight * (end - self).

    Both operands of ATen's numerically stable branch pair are computed the
    way the sequential fallback composes them (sub, scale, add/sub), with
    the branch selected on the host exactly like `fast_aten_lerp`.
    """
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var low_branch = Int(low_branch_arg)
    var slot, begin = _slot_begin(chunk_ends, Int(block_idx.x))
    var end = min(begin + FOREACH_EW_CHUNK, Int(numels[slot]))
    var self_values = _pick_mut(slot, a0, a1, a2, a3, a4, a5, a6, a7)
    var end_values = _pick_immut(slot, e0, e1, e2, e3, e4, e5, e6, e7)

    var index = begin + Int(thread_idx.x) * _VEC
    var stride = FOREACH_EW_THREADS * _VEC
    while index + _VEC <= end:
        var start = self_values.unsafe_load[width=_VEC, alignment=4](index)
        var finish = end_values.unsafe_load[width=_VEC, alignment=4](index)
        var difference = finish - start
        var result = start + _no_fuse[_VEC](weight * difference)
        if low_branch == 0:
            result = finish - _no_fuse[_VEC](one_minus_weight * difference)
        self_values.unsafe_store[width=_VEC, alignment=4](index, result)
        index += stride

    index = begin + ((end - begin) // _VEC) * _VEC + Int(thread_idx.x)
    while index < end:
        var start = self_values[unsafe_offset=index]
        var finish = end_values[unsafe_offset=index]
        var difference = finish - start
        var result = start + _no_fuse[1](weight * difference)
        if low_branch == 0:
            result = finish - _no_fuse[1](one_minus_weight * difference)
        self_values[unsafe_offset=index] = result
        index += FOREACH_EW_THREADS


def _foreach_addc_kernel[
    op_code: Int
](
    a0: _MutPtr,
    a1: _MutPtr,
    a2: _MutPtr,
    a3: _MutPtr,
    a4: _MutPtr,
    a5: _MutPtr,
    a6: _MutPtr,
    a7: _MutPtr,
    b0: _ImmutPtr,
    b1: _ImmutPtr,
    b2: _ImmutPtr,
    b3: _ImmutPtr,
    b4: _ImmutPtr,
    b5: _ImmutPtr,
    b6: _ImmutPtr,
    b7: _ImmutPtr,
    c0: _ImmutPtr,
    c1: _ImmutPtr,
    c2: _ImmutPtr,
    c3: _ImmutPtr,
    c4: _ImmutPtr,
    c5: _ImmutPtr,
    c6: _ImmutPtr,
    c7: _ImmutPtr,
    chunk_ends: _SlotInts,
    numels: _SlotInts,
    scalars: InlineArray[Float32, FOREACH_EW_SLOTS],
):
    """In-place self = self + value[slot] * (t1 * t2) or (t1 / t2).

    Matches `logic._ternary_bcast` element math exactly.
    """
    var slot, begin = _slot_begin(chunk_ends, Int(block_idx.x))
    var end = min(begin + FOREACH_EW_CHUNK, Int(numels[slot]))
    var self_values = _pick_mut(slot, a0, a1, a2, a3, a4, a5, a6, a7)
    var first_values = _pick_immut(slot, b0, b1, b2, b3, b4, b5, b6, b7)
    var second_values = _pick_immut(slot, c0, c1, c2, c3, c4, c5, c6, c7)
    var scalar = scalars[slot]

    var index = begin + Int(thread_idx.x) * _VEC
    var stride = FOREACH_EW_THREADS * _VEC
    while index + _VEC <= end:
        var a = self_values.unsafe_load[width=_VEC, alignment=4](index)
        var b = first_values.unsafe_load[width=_VEC, alignment=4](index)
        var c = second_values.unsafe_load[width=_VEC, alignment=4](index)
        comptime if op_code == FEA_ADDCMUL:
            a = a + scalar * (b * c)
        else:
            a = a + scalar * (b / c)
        self_values.unsafe_store[width=_VEC, alignment=4](index, a)
        index += stride

    index = begin + ((end - begin) // _VEC) * _VEC + Int(thread_idx.x)
    while index < end:
        var a = self_values[unsafe_offset=index]
        var b = first_values[unsafe_offset=index]
        var c = second_values[unsafe_offset=index]
        comptime if op_code == FEA_ADDCMUL:
            a = a + scalar * (b * c)
        else:
            a = a + scalar * (b / c)
        self_values[unsafe_offset=index] = a
        index += FOREACH_EW_THREADS


def _foreach_sqrt_kernel(
    i0: _ImmutPtr,
    i1: _ImmutPtr,
    i2: _ImmutPtr,
    i3: _ImmutPtr,
    i4: _ImmutPtr,
    i5: _ImmutPtr,
    i6: _ImmutPtr,
    i7: _ImmutPtr,
    o0: _MutPtr,
    o1: _MutPtr,
    o2: _MutPtr,
    o3: _MutPtr,
    o4: _MutPtr,
    o5: _MutPtr,
    o6: _MutPtr,
    o7: _MutPtr,
    chunk_ends: _SlotInts,
    numels: _SlotInts,
):
    """Out-of-place out = sqrt(in), matching `elementwise` UOP_SQRT."""
    var slot, begin = _slot_begin(chunk_ends, Int(block_idx.x))
    var end = min(begin + FOREACH_EW_CHUNK, Int(numels[slot]))
    var in_values = _pick_immut(slot, i0, i1, i2, i3, i4, i5, i6, i7)
    var out_values = _pick_mut(slot, o0, o1, o2, o3, o4, o5, o6, o7)

    var index = begin + Int(thread_idx.x) * _VEC
    var stride = FOREACH_EW_THREADS * _VEC
    while index + _VEC <= end:
        var value = in_values.unsafe_load[width=_VEC, alignment=4](index)
        out_values.unsafe_store[width=_VEC, alignment=4](
            index, ieee_sqrt(value)
        )
        index += stride

    index = begin + ((end - begin) // _VEC) * _VEC + Int(thread_idx.x)
    while index < end:
        out_values[unsafe_offset=index] = ieee_sqrt(
            in_values[unsafe_offset=index]
        )
        index += FOREACH_EW_THREADS


def _gather_scalars_kernel(
    out_ptr: _MutPtr,
    s0: _ImmutPtr,
    s1: _ImmutPtr,
    s2: _ImmutPtr,
    s3: _ImmutPtr,
    s4: _ImmutPtr,
    s5: _ImmutPtr,
    s6: _ImmutPtr,
    s7: _ImmutPtr,
    base_arg: Int64,
    count_arg: Int64,
):
    """out[base + i] = s_i[0]: one batched launch replaces per-scalar copies
    (the `stack` of foreach-norm outputs in gradient clipping)."""
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var base = Int(base_arg)
    var count = Int(count_arg)
    var i = Int(thread_idx.x)
    if i < count:
        out_ptr[unsafe_offset=base + i] = _pick_immut(
            i, s0, s1, s2, s3, s4, s5, s6, s7
        )[unsafe_offset=0]


def enqueue_foreach_gather_scalars_f32(
    out_addr: Int,
    in_addrs: InlineArray[Int, FOREACH_EW_SLOTS],
    base: Int,
    count: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        _enqueue_cached[_gather_scalars_kernel](
            ctx,
            1,
            1,
            1,
            FOREACH_EW_SLOTS,
            _addr_ptr(out_addr),
            _addr_ptr(in_addrs[0]).as_imm(),
            _addr_ptr(in_addrs[1]).as_imm(),
            _addr_ptr(in_addrs[2]).as_imm(),
            _addr_ptr(in_addrs[3]).as_imm(),
            _addr_ptr(in_addrs[4]).as_imm(),
            _addr_ptr(in_addrs[5]).as_imm(),
            _addr_ptr(in_addrs[6]).as_imm(),
            _addr_ptr(in_addrs[7]).as_imm(),
            Int64(base),
            Int64(count),
        )
    else:
        raise Error("no GPU accelerator available at compile time")


def enqueue_foreach_scalar_f32[
    op_code: Int
](
    addrs: InlineArray[Int, FOREACH_EW_SLOTS],
    chunk_ends: InlineArray[Int, FOREACH_EW_SLOTS],
    numels: InlineArray[Int, FOREACH_EW_SLOTS],
    scalars: InlineArray[Float32, FOREACH_EW_SLOTS],
    total_chunks: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        _enqueue_cached[_foreach_scalar_kernel[op_code]](
            ctx,
            total_chunks,
            1,
            1,
            FOREACH_EW_THREADS,
            _addr_ptr(addrs[0]),
            _addr_ptr(addrs[1]),
            _addr_ptr(addrs[2]),
            _addr_ptr(addrs[3]),
            _addr_ptr(addrs[4]),
            _addr_ptr(addrs[5]),
            _addr_ptr(addrs[6]),
            _addr_ptr(addrs[7]),
            _slot_ints(chunk_ends),
            _slot_ints(numels),
            scalars,
        )
    else:
        raise Error("no GPU accelerator available at compile time")


def enqueue_foreach_mul_tensor_f32(
    addrs: InlineArray[Int, FOREACH_EW_SLOTS],
    chunk_ends: InlineArray[Int, FOREACH_EW_SLOTS],
    numels: InlineArray[Int, FOREACH_EW_SLOTS],
    scalar_addr: Int,
    total_chunks: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        _enqueue_cached[_foreach_mul_tensor_kernel](
            ctx,
            total_chunks,
            1,
            1,
            FOREACH_EW_THREADS,
            _addr_ptr(addrs[0]),
            _addr_ptr(addrs[1]),
            _addr_ptr(addrs[2]),
            _addr_ptr(addrs[3]),
            _addr_ptr(addrs[4]),
            _addr_ptr(addrs[5]),
            _addr_ptr(addrs[6]),
            _addr_ptr(addrs[7]),
            _addr_ptr(scalar_addr).as_imm(),
            _slot_ints(chunk_ends),
            _slot_ints(numels),
        )
    else:
        raise Error("no GPU accelerator available at compile time")


def enqueue_foreach_nonfinite_unscale_f32(
    addrs: InlineArray[Int, FOREACH_EW_SLOTS],
    chunk_ends: InlineArray[Int, FOREACH_EW_SLOTS],
    numels: InlineArray[Int, FOREACH_EW_SLOTS],
    inv_scale_addr: Int,
    found_inf_addr: Int,
    total_chunks: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        _enqueue_cached[_foreach_nonfinite_unscale_kernel](
            ctx,
            total_chunks,
            1,
            1,
            FOREACH_EW_THREADS,
            _addr_ptr(addrs[0]),
            _addr_ptr(addrs[1]),
            _addr_ptr(addrs[2]),
            _addr_ptr(addrs[3]),
            _addr_ptr(addrs[4]),
            _addr_ptr(addrs[5]),
            _addr_ptr(addrs[6]),
            _addr_ptr(addrs[7]),
            _addr_ptr(inv_scale_addr).as_imm(),
            _addr_ptr(found_inf_addr),
            _slot_ints(chunk_ends),
            _slot_ints(numels),
        )
    else:
        raise Error("no GPU accelerator available at compile time")


def enqueue_foreach_lerp_f32(
    self_addrs: InlineArray[Int, FOREACH_EW_SLOTS],
    end_addrs: InlineArray[Int, FOREACH_EW_SLOTS],
    chunk_ends: InlineArray[Int, FOREACH_EW_SLOTS],
    numels: InlineArray[Int, FOREACH_EW_SLOTS],
    weight: Float32,
    one_minus_weight: Float32,
    low_branch: Int,
    total_chunks: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        _enqueue_cached[_foreach_lerp_kernel](
            ctx,
            total_chunks,
            1,
            1,
            FOREACH_EW_THREADS,
            _addr_ptr(self_addrs[0]),
            _addr_ptr(self_addrs[1]),
            _addr_ptr(self_addrs[2]),
            _addr_ptr(self_addrs[3]),
            _addr_ptr(self_addrs[4]),
            _addr_ptr(self_addrs[5]),
            _addr_ptr(self_addrs[6]),
            _addr_ptr(self_addrs[7]),
            _addr_ptr(end_addrs[0]).as_imm(),
            _addr_ptr(end_addrs[1]).as_imm(),
            _addr_ptr(end_addrs[2]).as_imm(),
            _addr_ptr(end_addrs[3]).as_imm(),
            _addr_ptr(end_addrs[4]).as_imm(),
            _addr_ptr(end_addrs[5]).as_imm(),
            _addr_ptr(end_addrs[6]).as_imm(),
            _addr_ptr(end_addrs[7]).as_imm(),
            _slot_ints(chunk_ends),
            _slot_ints(numels),
            weight,
            one_minus_weight,
            Int64(low_branch),
        )
    else:
        raise Error("no GPU accelerator available at compile time")


def enqueue_foreach_addc_f32[
    op_code: Int
](
    self_addrs: InlineArray[Int, FOREACH_EW_SLOTS],
    first_addrs: InlineArray[Int, FOREACH_EW_SLOTS],
    second_addrs: InlineArray[Int, FOREACH_EW_SLOTS],
    chunk_ends: InlineArray[Int, FOREACH_EW_SLOTS],
    numels: InlineArray[Int, FOREACH_EW_SLOTS],
    scalars: InlineArray[Float32, FOREACH_EW_SLOTS],
    total_chunks: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        _enqueue_cached[_foreach_addc_kernel[op_code]](
            ctx,
            total_chunks,
            1,
            1,
            FOREACH_EW_THREADS,
            _addr_ptr(self_addrs[0]),
            _addr_ptr(self_addrs[1]),
            _addr_ptr(self_addrs[2]),
            _addr_ptr(self_addrs[3]),
            _addr_ptr(self_addrs[4]),
            _addr_ptr(self_addrs[5]),
            _addr_ptr(self_addrs[6]),
            _addr_ptr(self_addrs[7]),
            _addr_ptr(first_addrs[0]).as_imm(),
            _addr_ptr(first_addrs[1]).as_imm(),
            _addr_ptr(first_addrs[2]).as_imm(),
            _addr_ptr(first_addrs[3]).as_imm(),
            _addr_ptr(first_addrs[4]).as_imm(),
            _addr_ptr(first_addrs[5]).as_imm(),
            _addr_ptr(first_addrs[6]).as_imm(),
            _addr_ptr(first_addrs[7]).as_imm(),
            _addr_ptr(second_addrs[0]).as_imm(),
            _addr_ptr(second_addrs[1]).as_imm(),
            _addr_ptr(second_addrs[2]).as_imm(),
            _addr_ptr(second_addrs[3]).as_imm(),
            _addr_ptr(second_addrs[4]).as_imm(),
            _addr_ptr(second_addrs[5]).as_imm(),
            _addr_ptr(second_addrs[6]).as_imm(),
            _addr_ptr(second_addrs[7]).as_imm(),
            _slot_ints(chunk_ends),
            _slot_ints(numels),
            scalars,
        )
    else:
        raise Error("no GPU accelerator available at compile time")


def enqueue_foreach_sqrt_f32(
    in_addrs: InlineArray[Int, FOREACH_EW_SLOTS],
    out_addrs: InlineArray[Int, FOREACH_EW_SLOTS],
    chunk_ends: InlineArray[Int, FOREACH_EW_SLOTS],
    numels: InlineArray[Int, FOREACH_EW_SLOTS],
    total_chunks: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        _enqueue_cached[_foreach_sqrt_kernel](
            ctx,
            total_chunks,
            1,
            1,
            FOREACH_EW_THREADS,
            _addr_ptr(in_addrs[0]).as_imm(),
            _addr_ptr(in_addrs[1]).as_imm(),
            _addr_ptr(in_addrs[2]).as_imm(),
            _addr_ptr(in_addrs[3]).as_imm(),
            _addr_ptr(in_addrs[4]).as_imm(),
            _addr_ptr(in_addrs[5]).as_imm(),
            _addr_ptr(in_addrs[6]).as_imm(),
            _addr_ptr(in_addrs[7]).as_imm(),
            _addr_ptr(out_addrs[0]),
            _addr_ptr(out_addrs[1]),
            _addr_ptr(out_addrs[2]),
            _addr_ptr(out_addrs[3]),
            _addr_ptr(out_addrs[4]),
            _addr_ptr(out_addrs[5]),
            _addr_ptr(out_addrs[6]),
            _addr_ptr(out_addrs[7]),
            _slot_ints(chunk_ends),
            _slot_ints(numels),
        )
    else:
        raise Error("no GPU accelerator available at compile time")
