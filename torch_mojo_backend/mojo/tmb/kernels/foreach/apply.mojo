"""The generic multi-tensor apply: one kernel body for every `aten::_foreach_*`
elementwise overload (CUDA's multi_tensor_apply + ForeachFunctors.cuh).

A foreach op applies the same elementwise math to every tensor of a list, so
running it as one ATen call per tensor is dispatch bound (an optimizer step
touches dozens of parameters). Here the whole list is ONE launch, and the
grid describes the WORK rather than the list: every tensor is cut into
`chunk_elements`-sized chunks and one block takes one chunk, each descriptor's
`chunk_end` being a running prefix sum of chunk counts so that a flat block
index maps back to (tensor, element range). Descriptors travel by value,
FE_DESC_CAP at a time; a longer list is several launches of the same compiled
kernel.

`_mta_kernel` is comptime-parametrized by
  * `dtype` -- the lists' storage dtype (a bool list is its uint8 storage,
    computed in int32 and stored back as 0/1);
  * `kind` -- the element math, `functors.foreach_math` (exp, add, lerp, ...);
  * `V` -- the overload's operand variant (`fe_variant`): how many tensor
    lists it reads (1-3), whether it writes a fresh output list, where its
    per-tensor host scalar goes (alpha, splatted into the second or third
    operand, or nowhere), and whether its second operand is a 0-d device
    tensor.
So `_foreach_add_.List`, `_foreach_add.Scalar`, `_foreach_exp` and
`_foreach_lerp.ScalarList` are the same function with different parameters;
nothing here knows about any one aten op.

`chunk_elements` is a runtime launch argument derived from the element count
and the SM count (`foreach_chunk_elements`), so every shape reuses one
compiled kernel and a few big tensors still fill the device.

Apple GPUs cannot take this route: Metal only translates pointer-typed kernel
arguments into GPU addresses, so a descriptor carrying raw addresses reads
back zeros. There `foreach_apply_enqueue` regroups the same descriptors into
the fixed-arity float32 kernels of `apple_kernels.mojo`, for the overloads
the host routes to Apple at all.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.collections import Array
from std.memory import bitcast
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv, min
from std.sys.info import has_apple_gpu_accelerator, size_of
from std.utils.numerics import isfinite

from tmb.kernels.common.op_utils import (
    _device_sm_count,
    _enqueue_cached,
    _make_ptr,
)
from tmb.kernels.foreach.apple_kernels import (
    FEA_ADDCDIV,
    FEA_ADDCMUL,
    FES_ADD,
    FES_DIV,
    FES_MUL,
    FOREACH_EW_CHUNK,
    FOREACH_EW_SLOTS,
    enqueue_foreach_addc_f32,
    enqueue_foreach_lerp_f32,
    enqueue_foreach_mul_tensor_f32,
    enqueue_foreach_nonfinite_unscale_f32,
    enqueue_foreach_scalar_f32,
    enqueue_foreach_sqrt_f32,
)
from tmb.kernels.foreach.contract import (
    FE_ROLE_ALPHA,
    FE_ROLE_B,
    FE_ROLE_C,
    fe_device_b,
    fe_lists,
    fe_out_of_place,
    fe_role,
    foreach_apple_supported,
)
from tmb.kernels.foreach.functors import foreach_math, foreach_opmath


comptime FE_THREADS = 256

# Blocks per SM the chunk size aims for. Swept on an H100 PCIe (114 SMs, FP32),
# ours/stock device time on `benchmarks/test_foreach.py`:
#                        2       4       8
#   mul_      4x1M    0.880   0.842   0.884
#   mul_      16x64k  0.390   0.391   0.596
#   addcmul_  4x1M    0.856   0.816   0.762
#   addcmul_  16x64k  0.412   0.438   0.651
# 8 starves the many-small-tensors regime (its chunk hits FE_MIN_CHUNK, so each
# block does one vector step and the descriptor scan dominates); 2 gives up the
# big-tensor cases. Fitted on this card, but derived from the RUNTIME SM count,
# so a smaller or larger GPU scales with it rather than inheriting a grid.
comptime FE_BLOCKS_PER_SM = 4

# The chunk size is rounded down to a multiple of this many elements, which
# keeps every chunk boundary 16-byte aligned for every dtype here (512
# one-byte elements is the smallest case) so only a tensor's base pointer can
# be misaligned.
comptime FE_CHUNK_GRAIN = 512
comptime FE_MIN_CHUNK = 1024
# A ceiling only bounds per-block work; nothing depends on the exact value.
comptime FE_MAX_CHUNK = 65_536

# Descriptors per launch. This bounds ONE launch argument (48 descriptors of
# six ints plus their scalars: 2.7 KB of the 4 KB parameter space), not the
# list: a longer TensorList becomes several launches of the same kernel.
comptime FE_DESC_CAP = 48


struct ForeachDesc(
    DevicePassable,
    ImplicitlyCopyable,
    TrivialRegisterPassable,
):
    """One tensor of the list: its operand addresses, length, and chunk
    prefix sum. Addresses an overload does not use stay zero and are never
    dereferenced: which ones a kernel reads is a compile-time fact."""

    comptime device_type: AnyType = Self

    var a: Int
    var b: Int
    var c: Int
    var dst: Int
    var numel: Int
    var chunk_end: Int

    def __init__(
        out self, a: Int, b: Int, c: Int, dst: Int, numel: Int, chunk_end: Int
    ):
        self.a = a
        self.b = b
        self.c = c
        self.dst = dst
        self.numel = numel
        self.chunk_end = chunk_end

    def _to_device_type(
        self,
        mut encoder: Some[DeviceTypeEncoder],
        target: MutOpaquePointer[_],
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "ForeachDesc"


@always_inline
def empty_foreach_desc() -> ForeachDesc:
    return ForeachDesc(0, 0, 0, 0, 0, 0)


@always_inline
def _compute_dtype[dtype: DType, is_bool: Bool]() -> DType:
    """The dtype `foreach_math` sees: a bool list (uint8 storage) as int32,
    which is what C++ promotes bool arithmetic to."""
    comptime if is_bool:
        return DType.int32
    else:
        return dtype


@always_inline
def _host_scalar[op: DType](bits: Int64) -> Scalar[op]:
    """A per-tensor host scalar in opmath: the host sends a floating list's
    scalar as float64 bits (`Scalar.to<opmath_t>()` narrows it here) and an
    integer or bool list's as the int64 value."""
    comptime if op.is_floating_point():
        return bitcast[DType.float64](bits).cast[op]()
    else:
        return bits.cast[op]()


@always_inline
def _mta_step[
    dtype: DType,
    kind: StaticString,
    V: Int,
    is_bool: Bool,
    width: Int,
    alignment: Int,
](
    a_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    b_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    c_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    out_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    flag_ptr: Pointer[Float32, MutUntrackedOrigin],
    index: Int,
    scalar: Scalar[foreach_opmath[_compute_dtype[dtype, is_bool]()]()],
    device_b: Scalar[foreach_opmath[_compute_dtype[dtype, is_bool]()]()],
):
    """`width` elements of the overload at `index`: gather the operands from
    their sources, apply the functor, store. The vector body and the scalar
    peel/tail of `_mta_kernel` are all calls to this."""
    comptime cdt = _compute_dtype[dtype, is_bool]()
    comptime op = foreach_opmath[cdt]()
    comptime lists = fe_lists[V]()
    comptime role = fe_role[V]()
    var a = a_ptr.unsafe_load[width=width, alignment=alignment](index).cast[
        cdt
    ]()
    var b = SIMD[op, width](0)
    var c = SIMD[op, width](0)
    comptime if lists >= 2:
        b = b_ptr.unsafe_load[width=width, alignment=alignment](index).cast[
            op
        ]()
    elif fe_device_b[V]() or kind == "nonfinite_unscale":
        b = SIMD[op, width](device_b)
    elif role == FE_ROLE_B:
        b = SIMD[op, width](scalar)
    comptime if lists >= 3:
        c = c_ptr.unsafe_load[width=width, alignment=alignment](index).cast[
            op
        ]()
    elif role == FE_ROLE_C:
        c = SIMD[op, width](scalar)
    var alpha = Scalar[op](1)
    comptime if role == FE_ROLE_ALPHA:
        alpha = scalar
    comptime if kind == "nonfinite_unscale":
        # AmpKernels.cu: every writer stores the same 1, so the race is benign
        # and needs no atomic; an inverse scale of exactly 1 leaves the values
        # untouched.
        if not isfinite(a).reduce_and():
            flag_ptr[] = 1.0
        if device_b == 1:
            return
    var r = foreach_math[kind, cdt, width](a, b, c, alpha)
    comptime if is_bool:
        out_ptr.unsafe_store[width=width, alignment=alignment](
            index, r.ne(0).cast[dtype]()
        )
    else:
        out_ptr.unsafe_store[width=width, alignment=alignment](
            index, r.cast[dtype]()
        )


@__name(t"multi_tensor_apply_{kind}_v{V}_{dtype}_t{FE_THREADS}")
def _mta_kernel[
    dtype: DType, kind: StaticString, V: Int, is_bool: Bool
](
    descs: Array[ForeachDesc, FE_DESC_CAP],
    scalars: Array[Int64, FE_DESC_CAP],
    desc_count_arg: Int64,
    chunk_elements_arg: Int64,
    device_addr_arg: Int64,
    aux_addr_arg: Int64,
):
    """One block per chunk of the concatenation of the list."""
    comptime cdt = _compute_dtype[dtype, is_bool]()
    comptime op = foreach_opmath[cdt]()
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var desc_count = Int(desc_count_arg)
    var chunk_elements = Int(chunk_elements_arg)

    var chunk = Int(block_idx.x)
    var desc_index = 0
    while desc_index + 1 < desc_count and chunk >= descs[desc_index].chunk_end:
        desc_index += 1
    var desc = descs[desc_index]
    var first_chunk = 0
    if desc_index != 0:
        first_chunk = descs[desc_index - 1].chunk_end
    var begin = (chunk - first_chunk) * chunk_elements
    var end = min(begin + chunk_elements, desc.numel)

    var scalar = _host_scalar[op](scalars[desc_index])
    var device_b = Scalar[op](0)
    comptime if kind == "nonfinite_unscale":
        # GradScaler's inverse scale is a float32 0-d tensor for every list
        # dtype.
        device_b = _make_ptr[DType.float32](Int(device_addr_arg))[
            unsafe_offset=0
        ].cast[op]()
    elif fe_device_b[V]():
        device_b = (
            _make_ptr[dtype](Int(device_addr_arg))[unsafe_offset=0]
            .cast[cdt]()
            .cast[op]()
        )
    var flag_ptr = _make_ptr[DType.float32](Int(aux_addr_arg))

    var a_ptr = _make_ptr[dtype](desc.a)
    var b_ptr = _make_ptr[dtype](desc.b)
    var c_ptr = _make_ptr[dtype](desc.c)
    var out_ptr = a_ptr
    comptime if fe_out_of_place[V]():
        out_ptr = _make_ptr[dtype](desc.dst)

    # 16-byte accesses need a 16-byte-aligned address, and a contiguous
    # tensor may be an offset view of one. Every chunk boundary is 16-byte
    # aligned (FE_CHUNK_GRAIN), so one peel of at most 16 bytes per chunk
    # lands the vector body on a boundary -- as long as all of the operands
    # share the base residue, which co-allocated tensor lists do.
    comptime item = size_of[dtype]()
    comptime vector = 16 // item
    var residue = desc.a % 16
    var vectorizable = residue % item == 0
    comptime if fe_lists[V]() >= 2:
        if desc.b % 16 != residue:
            vectorizable = False
    comptime if fe_lists[V]() >= 3:
        if desc.c % 16 != residue:
            vectorizable = False
    comptime if fe_out_of_place[V]():
        if desc.dst % 16 != residue:
            vectorizable = False

    var lane = Int(thread_idx.x)
    if vectorizable:
        var body = begin + min(((16 - residue) % 16) // item, end - begin)
        var head = begin + lane
        while head < body:
            _mta_step[dtype, kind, V, is_bool, 1, item](
                a_ptr, b_ptr, c_ptr, out_ptr, flag_ptr, head, scalar, device_b
            )
            head += FE_THREADS
        var index = body + lane * vector
        while index + vector <= end:
            _mta_step[dtype, kind, V, is_bool, vector, 16](
                a_ptr, b_ptr, c_ptr, out_ptr, flag_ptr, index, scalar, device_b
            )
            index += FE_THREADS * vector
        var tail = body + ((end - body) // vector) * vector + lane
        while tail < end:
            _mta_step[dtype, kind, V, is_bool, 1, item](
                a_ptr, b_ptr, c_ptr, out_ptr, flag_ptr, tail, scalar, device_b
            )
            tail += FE_THREADS
    else:
        var index = begin + lane
        while index < end:
            _mta_step[dtype, kind, V, is_bool, 1, item](
                a_ptr, b_ptr, c_ptr, out_ptr, flag_ptr, index, scalar, device_b
            )
            index += FE_THREADS


def foreach_chunk_elements(total_elements: Int, ctx: DeviceContext) -> Int:
    """Elements per block: the whole list's work spread over the whole device
    (also the fused optimizers' chunk). Apple keeps the fixed chunk its
    fixed-arity kernels are written against."""
    comptime if has_apple_gpu_accelerator():
        return FOREACH_EW_CHUNK
    else:
        var blocks = FE_BLOCKS_PER_SM * _device_sm_count(ctx)
        var chunk = (
            ceildiv(total_elements, blocks) if blocks > 0 else total_elements
        )
        chunk = (chunk // FE_CHUNK_GRAIN) * FE_CHUNK_GRAIN
        if chunk < FE_MIN_CHUNK:
            chunk = FE_MIN_CHUNK
        return min(chunk, FE_MAX_CHUNK)


# --- Apple: the same descriptors, Metal's fixed-arity ABI ---------------------


@always_inline
def _apple_backfill(mut addrs: Array[Int, FOREACH_EW_SLOTS], slot_count: Int):
    """Give empty/padding slots a valid dummy address (never dereferenced:
    those slots own zero chunks). Metal still requires every pointer-typed
    argument to translate to a real buffer."""
    var first_addr = 0
    for slot in range(slot_count):
        if addrs[slot] != 0:
            first_addr = addrs[slot]
            break
    for slot in range(FOREACH_EW_SLOTS):
        if addrs[slot] == 0:
            addrs[slot] = first_addr


def _foreach_apply_apple[
    kind: StaticString, V: Int
](
    descs: Array[ForeachDesc, FE_DESC_CAP],
    scalars: Array[Int64, FE_DESC_CAP],
    desc_count: Int,
    device_addr: Int,
    aux_addr: Int,
    ctx: DeviceContext,
) raises:
    var desc_index = 0
    while desc_index < desc_count:
        var group_first_chunk = 0
        if desc_index != 0:
            group_first_chunk = descs[desc_index - 1].chunk_end
        var first_addrs = Array[Int, FOREACH_EW_SLOTS](fill=0)
        var second_addrs = Array[Int, FOREACH_EW_SLOTS](fill=0)
        var third_addrs = Array[Int, FOREACH_EW_SLOTS](fill=0)
        var chunk_ends = Array[Int, FOREACH_EW_SLOTS](fill=0)
        var numels = Array[Int, FOREACH_EW_SLOTS](fill=0)
        var group_scalars = Array[Float32, FOREACH_EW_SLOTS](fill=0.0)
        var slot = 0
        while desc_index < desc_count and slot < FOREACH_EW_SLOTS:
            var desc = descs[desc_index]
            first_addrs[slot] = desc.a
            # sqrt is the one out-of-place member: its second list is the
            # output.
            second_addrs[slot] = desc.dst if kind == "sqrt" else desc.b
            third_addrs[slot] = desc.c
            numels[slot] = desc.numel
            chunk_ends[slot] = desc.chunk_end - group_first_chunk
            group_scalars[slot] = _host_scalar[DType.float32](
                scalars[desc_index]
            )
            desc_index += 1
            slot += 1
        var used_slots = slot
        var group_chunks = chunk_ends[used_slots - 1]
        while slot < FOREACH_EW_SLOTS:
            chunk_ends[slot] = group_chunks
            slot += 1
        if group_chunks == 0:
            continue
        _apple_backfill(first_addrs, used_slots)
        _apple_backfill(second_addrs, used_slots)
        _apple_backfill(third_addrs, used_slots)
        comptime if kind == "mul" and fe_device_b[V]():
            enqueue_foreach_mul_tensor_f32(
                first_addrs, chunk_ends, numels, device_addr, group_chunks, ctx
            )
        elif kind == "mul" or kind == "add" or kind == "div":
            comptime code = FES_MUL if kind == "mul" else (
                FES_ADD if kind == "add" else FES_DIV
            )
            enqueue_foreach_scalar_f32[code](
                first_addrs,
                chunk_ends,
                numels,
                group_scalars,
                group_chunks,
                ctx,
            )
        elif kind == "addcmul" or kind == "addcdiv":
            comptime code = FEA_ADDCMUL if kind == "addcmul" else FEA_ADDCDIV
            enqueue_foreach_addc_f32[code](
                first_addrs,
                second_addrs,
                third_addrs,
                chunk_ends,
                numels,
                group_scalars,
                group_chunks,
                ctx,
            )
        elif kind == "lerp":
            # One weight per launch: the host routes only `.Scalar` here.
            var weight = group_scalars[0]
            var low_branch = 1 if abs(weight) < Float32(0.5) else 0
            enqueue_foreach_lerp_f32(
                first_addrs,
                second_addrs,
                chunk_ends,
                numels,
                weight,
                Float32(1.0) - weight,
                low_branch,
                group_chunks,
                ctx,
            )
        elif kind == "nonfinite_unscale":
            enqueue_foreach_nonfinite_unscale_f32(
                first_addrs,
                chunk_ends,
                numels,
                device_addr,
                aux_addr,
                group_chunks,
                ctx,
            )
        elif kind == "sqrt":
            enqueue_foreach_sqrt_f32(
                first_addrs, second_addrs, chunk_ends, numels, group_chunks, ctx
            )
        else:
            comptime assert False, "no Apple launch for this foreach overload"


def foreach_apply_enqueue[
    dtype: DType, kind: StaticString, V: Int, is_bool: Bool
](
    descs: Array[ForeachDesc, FE_DESC_CAP],
    scalars: Array[Int64, FE_DESC_CAP],
    desc_count: Int,
    total_chunks: Int,
    chunk_elements: Int,
    device_addr: Int,
    aux_addr: Int,
    ctx: DeviceContext,
) raises:
    if desc_count <= 0 or total_chunks <= 0:
        return
    comptime if has_apple_gpu_accelerator():
        comptime if dtype == DType.float32 and foreach_apple_supported(kind, V):
            _foreach_apply_apple[kind, V](
                descs, scalars, desc_count, device_addr, aux_addr, ctx
            )
        else:
            raise Error(
                "mojo foreach ", kind, ": no Apple launch for this overload"
            )
    else:
        _enqueue_cached[_mta_kernel[dtype, kind, V, is_bool]](
            ctx,
            total_chunks,
            1,
            1,
            FE_THREADS,
            descs,
            scalars,
            Int64(desc_count),
            Int64(chunk_elements),
            Int64(device_addr),
            Int64(aux_addr),
        )
