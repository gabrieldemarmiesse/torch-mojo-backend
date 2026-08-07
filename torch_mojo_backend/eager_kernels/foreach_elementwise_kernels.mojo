"""Fixed-arity batched FP32 foreach elementwise kernels.

Metal only translates pointer-typed kernel *arguments* into valid GPU
addresses; a raw address smuggled as data (a descriptor struct or an Int)
reads back zeros. The CUDA-style descriptor-batch multi-tensor kernel is
therefore impossible on Apple GPUs, and per-tensor launches are dispatch
bound for optimizer-sized tensor lists. These kernels take a fixed arity of
FOREACH_EW_SLOTS real pointer arguments per tensor list plus plain-Int
per-slot chunk offsets/lengths (non-pointer data passes fine), so one
launch covers up to FOREACH_EW_SLOTS tensors, grid-strided over the
concatenation of their fixed-size chunks. Unused slots repeat a valid
pointer with length zero and are never dereferenced.

Element math deliberately mirrors the scalar eager kernels these ops fall
back to (`elementwise_ops`/`logic_ops`), keeping the batched path
bit-compatible with the sequential per-tensor decomposition.
"""

from std.collections import InlineArray
from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.math import min
from std.sys.info import has_accelerator

from op_utils import _enqueue_cached, ieee_sqrt


comptime FOREACH_EW_SLOTS = 8
comptime FOREACH_EW_CHUNK = 65_536  # elements per chunk; a multiple of _VEC
comptime FOREACH_EW_THREADS = 256
comptime _VEC = 4

comptime FES_MUL = 0
comptime FES_ADD = 1
comptime FES_DIV = 2

comptime FEA_ADDCMUL = 0
comptime FEA_ADDCDIV = 1

comptime _MutPtr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
comptime _ImmutPtr = UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]


@always_inline
def _addr_ptr(addr: Int) -> _MutPtr:
    return UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin](
        unsafe_from_address=addr
    ).as_unsafe_any_origin()


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
    var pointer = UnsafePointer(to=tmp).bitcast[Scalar[DType.float32]]()
    pointer.store[volatile=True](0, x)
    return pointer.load[width=width, volatile=True](0)


@always_inline
def _slot_begin(
    chunk_ends: InlineArray[Int, FOREACH_EW_SLOTS], chunk: Int
) -> Tuple[Int, Int]:
    """(slot, element offset in that slot's tensor) for a global chunk id.

    Empty slots occupy zero chunks (their chunk_end repeats the previous
    one), so the scan naturally steps over them.
    """
    var slot = 0
    while slot + 1 < FOREACH_EW_SLOTS and chunk >= chunk_ends[slot]:
        slot += 1
    var first_chunk = 0
    if slot != 0:
        first_chunk = chunk_ends[slot - 1]
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
    chunk_ends: InlineArray[Int, FOREACH_EW_SLOTS],
    numels: InlineArray[Int, FOREACH_EW_SLOTS],
    scalars: InlineArray[Float32, FOREACH_EW_SLOTS],
):
    """In-place x = x (op) scalar[slot], one scalar per slot.

    Matches `elementwise_ops._scalar_elementwise` (add/mul) and the DivSpec
    binary kernel (`a / b`) element math exactly.
    """
    var slot, begin = _slot_begin(chunk_ends, Int(block_idx.x))
    var end = min(begin + FOREACH_EW_CHUNK, numels[slot])
    var values = _pick_mut(slot, p0, p1, p2, p3, p4, p5, p6, p7)
    var scalar = scalars[slot]

    var index = begin + Int(thread_idx.x) * _VEC
    var stride = FOREACH_EW_THREADS * _VEC
    while index + _VEC <= end:
        var value = values.load[width=_VEC, alignment=4](index)
        comptime if op_code == FES_MUL:
            value = value * scalar
        comptime if op_code == FES_ADD:
            value = value + scalar
        comptime if op_code == FES_DIV:
            value = value / scalar
        values.store[width=_VEC, alignment=4](index, value)
        index += stride

    # The chunk size is divisible by _VEC, so only a tensor's last chunk can
    # need this scalar tail.
    index = begin + ((end - begin) // _VEC) * _VEC + Int(thread_idx.x)
    while index < end:
        var value = values[index]
        comptime if op_code == FES_MUL:
            value = value * scalar
        comptime if op_code == FES_ADD:
            value = value + scalar
        comptime if op_code == FES_DIV:
            value = value / scalar
        values[index] = value
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
    chunk_ends: InlineArray[Int, FOREACH_EW_SLOTS],
    numels: InlineArray[Int, FOREACH_EW_SLOTS],
    weight: Float32,
    one_minus_weight: Float32,
    low_branch: Int,
):
    """In-place scalar lerp: self = self + weight * (end - self).

    Both operands of ATen's numerically stable branch pair are computed the
    way the sequential fallback composes them (sub, scale, add/sub), with
    the branch selected on the host exactly like `fast_aten_lerp`.
    """
    var slot, begin = _slot_begin(chunk_ends, Int(block_idx.x))
    var end = min(begin + FOREACH_EW_CHUNK, numels[slot])
    var self_values = _pick_mut(slot, a0, a1, a2, a3, a4, a5, a6, a7)
    var end_values = _pick_immut(slot, e0, e1, e2, e3, e4, e5, e6, e7)

    var index = begin + Int(thread_idx.x) * _VEC
    var stride = FOREACH_EW_THREADS * _VEC
    while index + _VEC <= end:
        var start = self_values.load[width=_VEC, alignment=4](index)
        var finish = end_values.load[width=_VEC, alignment=4](index)
        var difference = finish - start
        var result = start + _no_fuse[_VEC](weight * difference)
        if low_branch == 0:
            result = finish - _no_fuse[_VEC](one_minus_weight * difference)
        self_values.store[width=_VEC, alignment=4](index, result)
        index += stride

    index = begin + ((end - begin) // _VEC) * _VEC + Int(thread_idx.x)
    while index < end:
        var start = self_values[index]
        var finish = end_values[index]
        var difference = finish - start
        var result = start + _no_fuse[1](weight * difference)
        if low_branch == 0:
            result = finish - _no_fuse[1](one_minus_weight * difference)
        self_values[index] = result
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
    chunk_ends: InlineArray[Int, FOREACH_EW_SLOTS],
    numels: InlineArray[Int, FOREACH_EW_SLOTS],
    scalars: InlineArray[Float32, FOREACH_EW_SLOTS],
):
    """In-place self = self + value[slot] * (t1 * t2) or (t1 / t2).

    Matches `logic_ops._ternary_bcast` element math exactly.
    """
    var slot, begin = _slot_begin(chunk_ends, Int(block_idx.x))
    var end = min(begin + FOREACH_EW_CHUNK, numels[slot])
    var self_values = _pick_mut(slot, a0, a1, a2, a3, a4, a5, a6, a7)
    var first_values = _pick_immut(slot, b0, b1, b2, b3, b4, b5, b6, b7)
    var second_values = _pick_immut(slot, c0, c1, c2, c3, c4, c5, c6, c7)
    var scalar = scalars[slot]

    var index = begin + Int(thread_idx.x) * _VEC
    var stride = FOREACH_EW_THREADS * _VEC
    while index + _VEC <= end:
        var a = self_values.load[width=_VEC, alignment=4](index)
        var b = first_values.load[width=_VEC, alignment=4](index)
        var c = second_values.load[width=_VEC, alignment=4](index)
        comptime if op_code == FEA_ADDCMUL:
            a = a + scalar * (b * c)
        else:
            a = a + scalar * (b / c)
        self_values.store[width=_VEC, alignment=4](index, a)
        index += stride

    index = begin + ((end - begin) // _VEC) * _VEC + Int(thread_idx.x)
    while index < end:
        var a = self_values[index]
        var b = first_values[index]
        var c = second_values[index]
        comptime if op_code == FEA_ADDCMUL:
            a = a + scalar * (b * c)
        else:
            a = a + scalar * (b / c)
        self_values[index] = a
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
    chunk_ends: InlineArray[Int, FOREACH_EW_SLOTS],
    numels: InlineArray[Int, FOREACH_EW_SLOTS],
):
    """Out-of-place out = sqrt(in), matching `elementwise_ops` UOP_SQRT."""
    var slot, begin = _slot_begin(chunk_ends, Int(block_idx.x))
    var end = min(begin + FOREACH_EW_CHUNK, numels[slot])
    var in_values = _pick_immut(slot, i0, i1, i2, i3, i4, i5, i6, i7)
    var out_values = _pick_mut(slot, o0, o1, o2, o3, o4, o5, o6, o7)

    var index = begin + Int(thread_idx.x) * _VEC
    var stride = FOREACH_EW_THREADS * _VEC
    while index + _VEC <= end:
        var value = in_values.load[width=_VEC, alignment=4](index)
        out_values.store[width=_VEC, alignment=4](index, ieee_sqrt(value))
        index += stride

    index = begin + ((end - begin) // _VEC) * _VEC + Int(thread_idx.x)
    while index < end:
        out_values[index] = ieee_sqrt(in_values[index])
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
    base: Int,
    count: Int,
):
    """out[base + i] = s_i[0]: one batched launch replaces per-scalar copies
    (the `stack` of foreach-norm outputs in gradient clipping)."""
    var i = Int(thread_idx.x)
    if i < count:
        out_ptr[base + i] = _pick_immut(i, s0, s1, s2, s3, s4, s5, s6, s7)[0]


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
            String("FOREACH_EW_GATHER_SCALARS_F32_V1"),
            1,
            1,
            1,
            FOREACH_EW_SLOTS,
            _addr_ptr(out_addr),
            _addr_ptr(in_addrs[0]).as_immutable(),
            _addr_ptr(in_addrs[1]).as_immutable(),
            _addr_ptr(in_addrs[2]).as_immutable(),
            _addr_ptr(in_addrs[3]).as_immutable(),
            _addr_ptr(in_addrs[4]).as_immutable(),
            _addr_ptr(in_addrs[5]).as_immutable(),
            _addr_ptr(in_addrs[6]).as_immutable(),
            _addr_ptr(in_addrs[7]).as_immutable(),
            base,
            count,
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
            String(t"FOREACH_EW_SCALAR_{op_code}_F32_V1"),
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
            chunk_ends,
            numels,
            scalars,
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
            String("FOREACH_EW_LERP_F32_V1"),
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
            _addr_ptr(end_addrs[0]).as_immutable(),
            _addr_ptr(end_addrs[1]).as_immutable(),
            _addr_ptr(end_addrs[2]).as_immutable(),
            _addr_ptr(end_addrs[3]).as_immutable(),
            _addr_ptr(end_addrs[4]).as_immutable(),
            _addr_ptr(end_addrs[5]).as_immutable(),
            _addr_ptr(end_addrs[6]).as_immutable(),
            _addr_ptr(end_addrs[7]).as_immutable(),
            chunk_ends,
            numels,
            weight,
            one_minus_weight,
            low_branch,
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
            String(t"FOREACH_EW_ADDC_{op_code}_F32_V1"),
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
            _addr_ptr(first_addrs[0]).as_immutable(),
            _addr_ptr(first_addrs[1]).as_immutable(),
            _addr_ptr(first_addrs[2]).as_immutable(),
            _addr_ptr(first_addrs[3]).as_immutable(),
            _addr_ptr(first_addrs[4]).as_immutable(),
            _addr_ptr(first_addrs[5]).as_immutable(),
            _addr_ptr(first_addrs[6]).as_immutable(),
            _addr_ptr(first_addrs[7]).as_immutable(),
            _addr_ptr(second_addrs[0]).as_immutable(),
            _addr_ptr(second_addrs[1]).as_immutable(),
            _addr_ptr(second_addrs[2]).as_immutable(),
            _addr_ptr(second_addrs[3]).as_immutable(),
            _addr_ptr(second_addrs[4]).as_immutable(),
            _addr_ptr(second_addrs[5]).as_immutable(),
            _addr_ptr(second_addrs[6]).as_immutable(),
            _addr_ptr(second_addrs[7]).as_immutable(),
            chunk_ends,
            numels,
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
            String("FOREACH_EW_SQRT_F32_V1"),
            total_chunks,
            1,
            1,
            FOREACH_EW_THREADS,
            _addr_ptr(in_addrs[0]).as_immutable(),
            _addr_ptr(in_addrs[1]).as_immutable(),
            _addr_ptr(in_addrs[2]).as_immutable(),
            _addr_ptr(in_addrs[3]).as_immutable(),
            _addr_ptr(in_addrs[4]).as_immutable(),
            _addr_ptr(in_addrs[5]).as_immutable(),
            _addr_ptr(in_addrs[6]).as_immutable(),
            _addr_ptr(in_addrs[7]).as_immutable(),
            _addr_ptr(out_addrs[0]),
            _addr_ptr(out_addrs[1]),
            _addr_ptr(out_addrs[2]),
            _addr_ptr(out_addrs[3]),
            _addr_ptr(out_addrs[4]),
            _addr_ptr(out_addrs[5]),
            _addr_ptr(out_addrs[6]),
            _addr_ptr(out_addrs[7]),
            chunk_ends,
            numels,
        )
    else:
        raise Error("no GPU accelerator available at compile time")
