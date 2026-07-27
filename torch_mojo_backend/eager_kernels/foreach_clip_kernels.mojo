"""Runtime-dynamic pure-Mojo FP32 foreach clipping kernels.

Norms use a two-stage reduction. A fixed-size chunk is reduced by one block
into caller-owned scratch, then one block per tensor reduces its chunk
partials and writes the ordered scalar result. Multiplication uses the same
runtime chunk descriptors and applies the device-resident scalar in place.
"""

from std.collections import InlineArray
from std.ffi import _get_global_or_null, external_call
from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.primitives import block
from std.math import min, sqrt
from std.memory import alloc
from std.sys.info import has_apple_gpu_accelerator

from foreach_clip_contract import (
    FOREACH_CHUNK_ELEMENTS,
    FOREACH_DESC_CAP,
    FOREACH_THREADS,
    ForeachDesc,
)
from foreach_elementwise_kernels import (
    FOREACH_EW_CHUNK,
    FOREACH_EW_SLOTS,
    _pick_immut,
    _pick_mut,
    _slot_begin,
)
from op_utils import _enqueue_cached


comptime _VEC = 4
comptime _MUL_THREADS = 128
comptime _MUL_ILP = 8


@always_inline
def _ptr(addr: Int) -> UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin]:
    return UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin](
        unsafe_from_address=addr
    )


@always_inline
def _chunk_bounds(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    chunk: Int,
) -> Tuple[ForeachDesc, Int, Int]:
    var desc_index = 0
    while desc_index + 1 < desc_count and chunk >= descs[desc_index].chunk_end:
        desc_index += 1
    var desc = descs[desc_index]
    var first_chunk = 0
    if desc_index != 0:
        first_chunk = descs[desc_index - 1].chunk_end
    var begin = (chunk - first_chunk) * FOREACH_CHUNK_ELEMENTS
    return desc, begin, min(begin + FOREACH_CHUNK_ELEMENTS, desc.numel)


@__name("foreach_l2_norm_f32_chunk_partials_v1")
def _norm_chunk_partials(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    partials_addr: Int,
):
    var chunk = Int(block_idx.x)
    var desc, begin, end = _chunk_bounds(descs, desc_count, chunk)
    var values = _ptr(desc.tensor_addr)
    var accum = SIMD[DType.float32, _VEC](0.0)
    var index = begin + Int(thread_idx.x) * _VEC
    var stride = FOREACH_THREADS * _VEC
    while index + _VEC <= end:
        var value = values.load[width=_VEC, alignment=4](index)
        accum = value.fma(value, accum)
        index += stride

    var scalar_accum = accum.reduce_add()
    # The contract's chunk size is divisible by four, so only the last chunk
    # of a tensor can need this scalar tail.
    index = begin + ((end - begin) // _VEC) * _VEC + Int(thread_idx.x)
    while index < end:
        var value = values[index]
        scalar_accum += value * value
        index += FOREACH_THREADS

    var total = block.sum[block_size=FOREACH_THREADS, broadcast=False](
        scalar_accum
    )
    if thread_idx.x == 0:
        _ptr(partials_addr)[chunk] = total


@__name("foreach_l2_norm_f32_finalize_v1")
def _norm_finalize(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    partials_addr: Int,
):
    var desc_index = Int(block_idx.x)
    if desc_index >= desc_count:
        return
    var desc = descs[desc_index]
    var begin = 0
    if desc_index != 0:
        begin = descs[desc_index - 1].chunk_end
    var accum = Float32(0.0)
    for chunk in range(
        begin + Int(thread_idx.x), desc.chunk_end, FOREACH_THREADS
    ):
        accum += _ptr(partials_addr)[chunk]
    var total = block.sum[block_size=FOREACH_THREADS, broadcast=False](accum)
    if thread_idx.x == 0:
        _ptr(desc.output_addr)[0] = sqrt(total)


# ---------------------------------------------------------------------------
# Apple/Metal fixed-arity batched variants. The batched kernels above receive
# tensor addresses as plain `Int` fields inside `ForeachDesc` and dereference
# them through pointers built in the kernel. Metal only translates
# pointer-typed *kernel arguments* into valid GPU addresses/bindings; a raw
# address smuggled as data reads back zeros. So on Apple every tensor pointer
# is a real kernel argument: FOREACH_EW_SLOTS of them per launch, with
# per-slot chunk offsets as plain-Int data (see foreach_elementwise_kernels).
# Chunking and the two-stage reduction are unchanged, so results stay
# bitwise identical to the former per-tensor-launch variants.
# ---------------------------------------------------------------------------

comptime _ImmutPtr = UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]
comptime _MutPtr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]


def _norm_partials_batched_apple(
    partials_ptr: _MutPtr,
    v0: _ImmutPtr,
    v1: _ImmutPtr,
    v2: _ImmutPtr,
    v3: _ImmutPtr,
    v4: _ImmutPtr,
    v5: _ImmutPtr,
    v6: _ImmutPtr,
    v7: _ImmutPtr,
    chunk_ends: InlineArray[Int, FOREACH_EW_SLOTS],
    numels: InlineArray[Int, FOREACH_EW_SLOTS],
):
    comptime assert FOREACH_CHUNK_ELEMENTS == FOREACH_EW_CHUNK
    var slot, begin = _slot_begin(chunk_ends, Int(block_idx.x))
    var end = min(begin + FOREACH_CHUNK_ELEMENTS, numels[slot])
    var values = _pick_immut(slot, v0, v1, v2, v3, v4, v5, v6, v7)
    var accum = SIMD[DType.float32, _VEC](0.0)
    var index = begin + Int(thread_idx.x) * _VEC
    var stride = FOREACH_THREADS * _VEC
    while index + _VEC <= end:
        var value = values.load[width=_VEC, alignment=4](index)
        accum = value.fma(value, accum)
        index += stride

    var scalar_accum = accum.reduce_add()
    # The chunk size is divisible by four, so only the last chunk of a
    # tensor can need this scalar tail.
    index = begin + ((end - begin) // _VEC) * _VEC + Int(thread_idx.x)
    while index < end:
        var value = values[index]
        scalar_accum += value * value
        index += FOREACH_THREADS

    var total = block.sum[block_size=FOREACH_THREADS, broadcast=False](
        scalar_accum
    )
    if thread_idx.x == 0:
        partials_ptr[Int(block_idx.x)] = total


def _norm_finalize_batched_apple(
    o0: _MutPtr,
    o1: _MutPtr,
    o2: _MutPtr,
    o3: _MutPtr,
    o4: _MutPtr,
    o5: _MutPtr,
    o6: _MutPtr,
    o7: _MutPtr,
    partials_ptr: _ImmutPtr,
    chunk_ends: InlineArray[Int, FOREACH_EW_SLOTS],
    used_slots: Int,
):
    var slot = Int(block_idx.x)
    if slot >= used_slots:
        return
    var first_chunk = 0
    if slot != 0:
        first_chunk = chunk_ends[slot - 1]
    var accum = Float32(0.0)
    for chunk in range(
        first_chunk + Int(thread_idx.x), chunk_ends[slot], FOREACH_THREADS
    ):
        accum += partials_ptr[chunk]
    var total = block.sum[block_size=FOREACH_THREADS, broadcast=False](accum)
    if thread_idx.x == 0:
        var out_ptr = _pick_mut(slot, o0, o1, o2, o3, o4, o5, o6, o7)
        out_ptr[0] = sqrt(total)


def _mul_tensor_batched_apple(
    p0: _MutPtr,
    p1: _MutPtr,
    p2: _MutPtr,
    p3: _MutPtr,
    p4: _MutPtr,
    p5: _MutPtr,
    p6: _MutPtr,
    p7: _MutPtr,
    scalar_ptr: _ImmutPtr,
    chunk_ends: InlineArray[Int, FOREACH_EW_SLOTS],
    numels: InlineArray[Int, FOREACH_EW_SLOTS],
):
    var scalar = scalar_ptr[0]
    var slot, begin = _slot_begin(chunk_ends, Int(block_idx.x))
    var end = min(begin + FOREACH_CHUNK_ELEMENTS, numels[slot])
    var values = _pick_mut(slot, p0, p1, p2, p3, p4, p5, p6, p7)
    var index = begin + Int(thread_idx.x) * _VEC
    var stride = FOREACH_THREADS * _VEC
    while index + _VEC <= end:
        var value = values.load[width=_VEC, alignment=4](index)
        values.store[width=_VEC, alignment=4](index, value * scalar)
        index += stride

    index = begin + ((end - begin) // _VEC) * _VEC + Int(thread_idx.x)
    while index < end:
        values[index] = values[index] * scalar
        index += FOREACH_THREADS


@__name("foreach_mul_tensor_f32_aligned_streaming_ilp8_t128_v8")
def _mul_aligned_streaming_ilp8_t128(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    scalar_addr: Int,
):
    var chunk = Int(block_idx.x)
    var desc, begin, end = _chunk_bounds(descs, desc_count, chunk)
    var values = _ptr(desc.tensor_addr)
    var scalar = _ptr(scalar_addr)[0]
    var tid = Int(thread_idx.x)

    # Each descriptor may start at any four-byte alignment. Peeling at most
    # 31 values makes the first full warp access begin on a 128-byte boundary;
    # all following warp accesses then use exactly four 32-byte sectors instead
    # of five. Since the fixed chunk is a multiple of 128 bytes, the same
    # bounded peel applies independently to every chunk.
    var address = desc.tensor_addr + begin * 4
    var prefix = ((128 - address % 128) % 128) // 4
    var body_begin = min(begin + prefix, end)
    var prefix_index = begin + tid
    if prefix_index < body_begin:
        var prefix_value = values.load[width=1, alignment=4, non_temporal=True](
            prefix_index
        )[0]
        values.store[width=1, alignment=4, non_temporal=True](
            prefix_index, SIMD[DType.float32, 1](prefix_value * scalar)
        )

    var index = body_begin + tid
    var stride = _MUL_THREADS * _MUL_ILP
    while index + (_MUL_ILP - 1) * _MUL_THREADS < end:
        var loaded = SIMD[DType.float32, _MUL_ILP]()
        comptime for u in range(_MUL_ILP):
            loaded[u] = values.load[
                width=1,
                alignment=4,
                non_temporal=True,
            ](
                index + u * _MUL_THREADS
            )[0]
        loaded *= scalar
        comptime for u in range(_MUL_ILP):
            values.store[width=1, alignment=4, non_temporal=True](
                index + u * _MUL_THREADS,
                SIMD[DType.float32, 1](loaded[u]),
            )
        index += stride

    while index < end:
        var value = values.load[
            width=1,
            alignment=4,
            non_temporal=True,
        ](
            index
        )[0]
        values.store[width=1, alignment=4, non_temporal=True](
            index, SIMD[DType.float32, 1](value * scalar)
        )
        index += _MUL_THREADS


def _enqueue_norm_partials_cached(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    total_chunks: Int,
    partials_addr: Int,
    ctx: DeviceContext,
) raises:
    var cache_name = String(t"FOREACH_NORM_PARTIALS_F32_V1_{ctx.id()}")
    comptime FuncT = type_of(ctx.compile_function[_norm_chunk_partials]())
    if global_ptr := _get_global_or_null(cache_name):
        var cached = global_ptr.value().bitcast[FuncT]()
        ctx.enqueue_function(
            cached[],
            descs,
            desc_count,
            partials_addr,
            grid_dim=(total_chunks,),
            block_dim=(FOREACH_THREADS,),
        )
        return
    var compiled = ctx.compile_function[_norm_chunk_partials]()
    var cached = alloc[FuncT](1)
    cached.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_name), cached.bitcast[NoneType]()
    )
    ctx.enqueue_function(
        cached[],
        descs,
        desc_count,
        partials_addr,
        grid_dim=(total_chunks,),
        block_dim=(FOREACH_THREADS,),
    )


def _enqueue_norm_finalize_cached(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    partials_addr: Int,
    ctx: DeviceContext,
) raises:
    var cache_name = String(t"FOREACH_NORM_FINALIZE_F32_V1_{ctx.id()}")
    comptime FuncT = type_of(ctx.compile_function[_norm_finalize]())
    if global_ptr := _get_global_or_null(cache_name):
        var cached = global_ptr.value().bitcast[FuncT]()
        ctx.enqueue_function(
            cached[],
            descs,
            desc_count,
            partials_addr,
            grid_dim=(desc_count,),
            block_dim=(FOREACH_THREADS,),
        )
        return
    var compiled = ctx.compile_function[_norm_finalize]()
    var cached = alloc[FuncT](1)
    cached.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_name), cached.bitcast[NoneType]()
    )
    ctx.enqueue_function(
        cached[],
        descs,
        desc_count,
        partials_addr,
        grid_dim=(desc_count,),
        block_dim=(FOREACH_THREADS,),
    )


def _enqueue_mul_cached(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    total_chunks: Int,
    scalar_addr: Int,
    ctx: DeviceContext,
) raises:
    var cache_name = String(t"FOREACH_MUL_TENSOR_F32_V1_{ctx.id()}")
    comptime FuncT = type_of(
        ctx.compile_function[_mul_aligned_streaming_ilp8_t128]()
    )
    if global_ptr := _get_global_or_null(cache_name):
        var cached = global_ptr.value().bitcast[FuncT]()
        ctx.enqueue_function(
            cached[],
            descs,
            desc_count,
            scalar_addr,
            grid_dim=(total_chunks,),
            block_dim=(_MUL_THREADS,),
        )
        return
    var compiled = ctx.compile_function[_mul_aligned_streaming_ilp8_t128]()
    var cached = alloc[FuncT](1)
    cached.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_name), cached.bitcast[NoneType]()
    )
    ctx.enqueue_function(
        cached[],
        descs,
        desc_count,
        scalar_addr,
        grid_dim=(total_chunks,),
        block_dim=(_MUL_THREADS,),
    )


def enqueue_foreach_l2_norm_f32(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    total_chunks: Int,
    partials_addr: Int,
    ctx: DeviceContext,
) raises:
    if desc_count <= 0:
        return
    comptime if has_apple_gpu_accelerator():
        var desc_index = 0
        while desc_index < desc_count:
            var group_first_chunk = 0
            if desc_index != 0:
                group_first_chunk = descs[desc_index - 1].chunk_end
            var in_addrs = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
            var out_addrs = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
            var chunk_ends = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
            var numels = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
            var slot = 0
            while desc_index < desc_count and slot < FOREACH_EW_SLOTS:
                var desc = descs[desc_index]
                in_addrs[slot] = desc.tensor_addr
                out_addrs[slot] = desc.output_addr
                numels[slot] = desc.numel
                chunk_ends[slot] = desc.chunk_end - group_first_chunk
                desc_index += 1
                slot += 1
            var used_slots = slot
            var group_chunks = chunk_ends[used_slots - 1]
            while slot < FOREACH_EW_SLOTS:
                chunk_ends[slot] = group_chunks
                slot += 1
            # Padding/empty slots still need translatable pointer arguments;
            # they own zero chunks (or return early), so any real device
            # pointer works. The scratch buffer is always allocated.
            for backfill in range(FOREACH_EW_SLOTS):
                if in_addrs[backfill] == 0:
                    in_addrs[backfill] = partials_addr
                if out_addrs[backfill] == 0:
                    out_addrs[backfill] = partials_addr
            var partials = _ptr(
                partials_addr + group_first_chunk * 4
            ).as_unsafe_any_origin()
            if group_chunks > 0:
                _enqueue_cached[_norm_partials_batched_apple](
                    ctx,
                    String("FOREACH_NORM_PARTIALS_APPLE_B8_F32_V1"),
                    group_chunks,
                    1,
                    1,
                    FOREACH_THREADS,
                    partials,
                    _ptr(in_addrs[0]).as_unsafe_any_origin().as_immutable(),
                    _ptr(in_addrs[1]).as_unsafe_any_origin().as_immutable(),
                    _ptr(in_addrs[2]).as_unsafe_any_origin().as_immutable(),
                    _ptr(in_addrs[3]).as_unsafe_any_origin().as_immutable(),
                    _ptr(in_addrs[4]).as_unsafe_any_origin().as_immutable(),
                    _ptr(in_addrs[5]).as_unsafe_any_origin().as_immutable(),
                    _ptr(in_addrs[6]).as_unsafe_any_origin().as_immutable(),
                    _ptr(in_addrs[7]).as_unsafe_any_origin().as_immutable(),
                    chunk_ends,
                    numels,
                )
            _enqueue_cached[_norm_finalize_batched_apple](
                ctx,
                String("FOREACH_NORM_FINALIZE_APPLE_B8_F32_V1"),
                used_slots,
                1,
                1,
                FOREACH_THREADS,
                _ptr(out_addrs[0]).as_unsafe_any_origin(),
                _ptr(out_addrs[1]).as_unsafe_any_origin(),
                _ptr(out_addrs[2]).as_unsafe_any_origin(),
                _ptr(out_addrs[3]).as_unsafe_any_origin(),
                _ptr(out_addrs[4]).as_unsafe_any_origin(),
                _ptr(out_addrs[5]).as_unsafe_any_origin(),
                _ptr(out_addrs[6]).as_unsafe_any_origin(),
                _ptr(out_addrs[7]).as_unsafe_any_origin(),
                partials.as_immutable(),
                chunk_ends,
                used_slots,
            )
    else:
        if total_chunks > 0:
            _enqueue_norm_partials_cached(
                descs, desc_count, total_chunks, partials_addr, ctx
            )
        _enqueue_norm_finalize_cached(descs, desc_count, partials_addr, ctx)


def enqueue_foreach_mul_tensor_f32(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    total_chunks: Int,
    scalar_addr: Int,
    ctx: DeviceContext,
) raises:
    if desc_count <= 0 or total_chunks <= 0:
        return
    comptime if has_apple_gpu_accelerator():
        var scalar = _ptr(scalar_addr).as_unsafe_any_origin().as_immutable()
        var desc_index = 0
        while desc_index < desc_count:
            var group_first_chunk = 0
            if desc_index != 0:
                group_first_chunk = descs[desc_index - 1].chunk_end
            var addrs = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
            var chunk_ends = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
            var numels = InlineArray[Int, FOREACH_EW_SLOTS](fill=0)
            var slot = 0
            while desc_index < desc_count and slot < FOREACH_EW_SLOTS:
                var desc = descs[desc_index]
                addrs[slot] = desc.tensor_addr
                numels[slot] = desc.numel
                chunk_ends[slot] = desc.chunk_end - group_first_chunk
                desc_index += 1
                slot += 1
            var group_chunks = chunk_ends[slot - 1]
            while slot < FOREACH_EW_SLOTS:
                chunk_ends[slot] = group_chunks
                slot += 1
            if group_chunks == 0:
                continue
            # Padding/empty slots own zero chunks and are never dereferenced,
            # but Metal still needs a translatable pointer argument.
            for backfill in range(FOREACH_EW_SLOTS):
                if addrs[backfill] == 0:
                    addrs[backfill] = scalar_addr
            _enqueue_cached[_mul_tensor_batched_apple](
                ctx,
                String("FOREACH_MUL_TENSOR_APPLE_B8_F32_V1"),
                group_chunks,
                1,
                1,
                FOREACH_THREADS,
                _ptr(addrs[0]).as_unsafe_any_origin(),
                _ptr(addrs[1]).as_unsafe_any_origin(),
                _ptr(addrs[2]).as_unsafe_any_origin(),
                _ptr(addrs[3]).as_unsafe_any_origin(),
                _ptr(addrs[4]).as_unsafe_any_origin(),
                _ptr(addrs[5]).as_unsafe_any_origin(),
                _ptr(addrs[6]).as_unsafe_any_origin(),
                _ptr(addrs[7]).as_unsafe_any_origin(),
                scalar,
                chunk_ends,
                numels,
            )
    else:
        _enqueue_mul_cached(descs, desc_count, total_chunks, scalar_addr, ctx)
