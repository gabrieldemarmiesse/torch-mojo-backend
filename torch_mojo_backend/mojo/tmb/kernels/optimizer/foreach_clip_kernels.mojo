"""Runtime-dynamic pure-Mojo FP32 foreach L2-norm kernels.

Norms use a two-stage reduction. A fixed-size chunk is reduced by one block
into caller-owned scratch, then one block per tensor reduces its chunk
partials and writes the ordered scalar result. The rescaling multiply that
follows a norm is not here: it is `aten::_foreach_mul_.Tensor`, one member of
the elementwise family `foreach_batched_kernels` serves.
"""

from std.collections import InlineArray
from std.ffi import _get_global_or_null, external_call
from std.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives import block
from std.math import min
from std.memory.alloc import unsafe_alloc
from std.sys.info import has_apple_gpu_accelerator

from tmb.kernels.optimizer.foreach_clip_contract import (
    FOREACH_CHUNK_ELEMENTS,
    FOREACH_DESC_CAP,
    FOREACH_THREADS,
    ForeachDesc,
)
from tmb.kernels.optimizer.foreach_elementwise_kernels import (
    FOREACH_EW_CHUNK,
    FOREACH_EW_SLOTS,
    _SlotInts,
    _pick_immut,
    _pick_mut,
    _slot_begin,
    _slot_ints,
)
from tmb.kernels.common.op_utils import _enqueue_cached, ieee_sqrt


comptime _VEC = 4


@always_inline
def _ptr(addr: Int) -> Pointer[Scalar[DType.float32], MutUntrackedOrigin]:
    return Pointer[Scalar[DType.float32], MutUntrackedOrigin](
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
    desc_count_arg: Int64,
    partials_addr_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var desc_count = Int(desc_count_arg)
    var partials_addr = Int(partials_addr_arg)
    var chunk = Int(block_idx.x)
    var desc, begin, end = _chunk_bounds(descs, desc_count, chunk)
    var values = _ptr(desc.tensor_addr)
    var accum = SIMD[DType.float32, _VEC](0.0)
    var index = begin + Int(thread_idx.x) * _VEC
    var stride = FOREACH_THREADS * _VEC
    while index + _VEC <= end:
        var value = values.unsafe_load[width=_VEC, alignment=4](index)
        accum = value.fma(value, accum)
        index += stride

    var scalar_accum = accum.reduce_add()
    # The contract's chunk size is divisible by four, so only the last chunk
    # of a tensor can need this scalar tail.
    index = begin + ((end - begin) // _VEC) * _VEC + Int(thread_idx.x)
    while index < end:
        var value = values[unsafe_offset=index]
        scalar_accum += value * value
        index += FOREACH_THREADS

    var total = block.sum[block_size=FOREACH_THREADS, broadcast=False](
        scalar_accum
    )
    if thread_idx.x == 0:
        _ptr(partials_addr)[unsafe_offset=chunk] = total


@__name("foreach_l2_norm_f32_finalize_v1")
def _norm_finalize(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count_arg: Int64,
    partials_addr_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var desc_count = Int(desc_count_arg)
    var partials_addr = Int(partials_addr_arg)
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
        accum += _ptr(partials_addr)[unsafe_offset=chunk]
    var total = block.sum[block_size=FOREACH_THREADS, broadcast=False](accum)
    if thread_idx.x == 0:
        _ptr(desc.output_addr)[unsafe_offset=0] = ieee_sqrt(total)


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

comptime _ImmutPtr = Pointer[Scalar[DType.float32], ImmutAnyOrigin]
comptime _MutPtr = Pointer[Scalar[DType.float32], MutAnyOrigin]


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
    chunk_ends: _SlotInts,
    numels: _SlotInts,
):
    comptime assert FOREACH_CHUNK_ELEMENTS == FOREACH_EW_CHUNK
    var slot, begin = _slot_begin(chunk_ends, Int(block_idx.x))
    var end = min(begin + FOREACH_CHUNK_ELEMENTS, Int(numels[slot]))
    var values = _pick_immut(slot, v0, v1, v2, v3, v4, v5, v6, v7)
    var accum = SIMD[DType.float32, _VEC](0.0)
    var index = begin + Int(thread_idx.x) * _VEC
    var stride = FOREACH_THREADS * _VEC
    while index + _VEC <= end:
        var value = values.unsafe_load[width=_VEC, alignment=4](index)
        accum = value.fma(value, accum)
        index += stride

    var scalar_accum = accum.reduce_add()
    # The chunk size is divisible by four, so only the last chunk of a
    # tensor can need this scalar tail.
    index = begin + ((end - begin) // _VEC) * _VEC + Int(thread_idx.x)
    while index < end:
        var value = values[unsafe_offset=index]
        scalar_accum += value * value
        index += FOREACH_THREADS

    var total = block.sum[block_size=FOREACH_THREADS, broadcast=False](
        scalar_accum
    )
    if thread_idx.x == 0:
        partials_ptr[unsafe_offset=Int(block_idx.x)] = total


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
    chunk_ends: _SlotInts,
    used_slots_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var used_slots = Int(used_slots_arg)
    var slot = Int(block_idx.x)
    if slot >= used_slots:
        return
    var first_chunk = 0
    if slot != 0:
        first_chunk = Int(chunk_ends[slot - 1])
    var accum = Float32(0.0)
    for chunk in range(
        first_chunk + Int(thread_idx.x), Int(chunk_ends[slot]), FOREACH_THREADS
    ):
        accum += partials_ptr[unsafe_offset=chunk]
    var total = block.sum[block_size=FOREACH_THREADS, broadcast=False](accum)
    if thread_idx.x == 0:
        var out_ptr = _pick_mut(slot, o0, o1, o2, o3, o4, o5, o6, o7)
        out_ptr[unsafe_offset=0] = ieee_sqrt(total)


def _enqueue_norm_partials_cached(
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    total_chunks: Int,
    partials_addr: Int,
    ctx: DeviceContext,
) raises:
    var cache_name = String(t"FOREACH_NORM_PARTIALS_F32_V1_{ctx.id()}")
    comptime FuncT = type_of(ctx.compile_function[_norm_chunk_partials]())
    var global_ptr = _get_global_or_null(cache_name)
    if global_ptr:
        var cached = global_ptr.value().unsafe_bitcast[FuncT]()
        ctx.enqueue_function(
            cached[],
            descs,
            Int64(desc_count),
            Int64(partials_addr),
            grid_dim=(total_chunks,),
            block_dim=(FOREACH_THREADS,),
        )
        return
    var compiled = ctx.compile_function[_norm_chunk_partials]()
    var cached = unsafe_alloc[FuncT](1)
    cached.unsafe_write(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_name), cached.unsafe_bitcast[NoneType]()
    )
    ctx.enqueue_function(
        cached[],
        descs,
        Int64(desc_count),
        Int64(partials_addr),
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
    var global_ptr = _get_global_or_null(cache_name)
    if global_ptr:
        var cached = global_ptr.value().unsafe_bitcast[FuncT]()
        ctx.enqueue_function(
            cached[],
            descs,
            Int64(desc_count),
            Int64(partials_addr),
            grid_dim=(desc_count,),
            block_dim=(FOREACH_THREADS,),
        )
        return
    var compiled = ctx.compile_function[_norm_finalize]()
    var cached = unsafe_alloc[FuncT](1)
    cached.unsafe_write(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_name), cached.unsafe_bitcast[NoneType]()
    )
    ctx.enqueue_function(
        cached[],
        descs,
        Int64(desc_count),
        Int64(partials_addr),
        grid_dim=(desc_count,),
        block_dim=(FOREACH_THREADS,),
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
                    group_chunks,
                    1,
                    1,
                    FOREACH_THREADS,
                    partials,
                    _ptr(in_addrs[0]).as_unsafe_any_origin().as_imm(),
                    _ptr(in_addrs[1]).as_unsafe_any_origin().as_imm(),
                    _ptr(in_addrs[2]).as_unsafe_any_origin().as_imm(),
                    _ptr(in_addrs[3]).as_unsafe_any_origin().as_imm(),
                    _ptr(in_addrs[4]).as_unsafe_any_origin().as_imm(),
                    _ptr(in_addrs[5]).as_unsafe_any_origin().as_imm(),
                    _ptr(in_addrs[6]).as_unsafe_any_origin().as_imm(),
                    _ptr(in_addrs[7]).as_unsafe_any_origin().as_imm(),
                    _slot_ints(chunk_ends),
                    _slot_ints(numels),
                )
            _enqueue_cached[_norm_finalize_batched_apple](
                ctx,
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
                partials.as_imm(),
                _slot_ints(chunk_ends),
                Int64(used_slots),
            )
    else:
        if total_chunks > 0:
            _enqueue_norm_partials_cached(
                descs, desc_count, total_chunks, partials_addr, ctx
            )
        _enqueue_norm_finalize_cached(descs, desc_count, partials_addr, ctx)
