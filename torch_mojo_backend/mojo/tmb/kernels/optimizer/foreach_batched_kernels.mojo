"""One descriptor-batched kernel body for the whole foreach elementwise family.

`aten::_foreach_*` is a list-of-tensors op: the same elementwise math applied
to every tensor of a TensorList. Running it as one ATen call per tensor is
dispatch bound (an optimizer step touches dozens of parameters), so the list
has to become ONE launch, and the grid has to describe the WORK rather than
the list.

This file is that single implementation. `_foreach_ew_kernel` is
comptime-parametrized by

  * `dtype` -- any of `op_utils.FLOAT_DTYPES`; every op computes in FP32 and
    narrows back, which is exactly what the sequential `elementwise` /
    `logic` kernels these paths replace do, so the batched result stays
    bit-compatible with the per-tensor decomposition;
  * `op` -- which element math to apply: mul/add/div by a host scalar (one per
    tensor, so `.Scalar` and `.ScalarList` are the same code), multiply by a
    device-resident scalar tensor, scalar lerp, addcmul, addcdiv, sqrt, and
    GradScaler's non-finite check and unscale.

Every tensor is cut into `chunk_elements`-sized chunks and one block takes one
chunk, with the descriptor's `chunk_end` a running prefix sum of chunk counts
so a flat block index maps back to (tensor, element range).

`chunk_elements` is a runtime launch argument, and that is the whole
performance story of this file: a chunk fixed at 65_536 elements gives a
four-tensor, 1M-elements-each list 64 blocks, which leaves more than half of
an H100's 114 SMs idle for the entire launch, and gives a sixteen-tensor
64k-elements-each list 16 blocks. Sizing the chunk from the element count and
the SM count instead puts both on a few hundred blocks. Being a launch
argument and not a compile-time parameter, every shape reuses one compiled
kernel -- no shape is baked in anywhere.

Apple GPUs cannot take this route at all: Metal only translates pointer-typed
kernel *arguments* into GPU addresses, so a descriptor carrying raw addresses
reads back as zeros (measured; see `foreach_elementwise_kernels`).
`_foreach_ew_enqueue` therefore regroups the very same descriptors into that
file's fixed-arity Metal launches on Apple, and takes the descriptor route
everywhere else.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.collections import Array
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv, min
from std.sys.info import has_apple_gpu_accelerator, size_of
from std.utils.numerics import isfinite

from tmb.kernels.optimizer.foreach_elementwise_kernels import (
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
from tmb.kernels.common.op_utils import (
    _enqueue_cached,
    _make_ptr,
    _device_sm_count,
    ieee_sqrt,
)


# Element math selectors. `.Scalar` and `.ScalarList` share a code: the scalar
# is per tensor either way.
comptime FEW_MUL = 0  # self *= scalar[i]
comptime FEW_ADD = 1  # self += scalar[i]
comptime FEW_DIV = 2  # self /= scalar[i]
comptime FEW_MUL_TENSOR = 3  # self *= *scalar_addr (a 0-d device tensor)
comptime FEW_LERP = 4  # self = lerp(self, end, weight)
comptime FEW_ADDCMUL = 5  # self += scalar[i] * (t1 * t2)
comptime FEW_ADDCDIV = 6  # self += scalar[i] * (t1 / t2)
comptime FEW_SQRT = 7  # out = sqrt(in), the one out-of-place member
# *flag_addr = 1 if any self is non-finite; self *= *scalar_addr (an FP32 0-d
# device tensor, whatever the list dtype) unless that scalar is exactly 1.
comptime FEW_NONFINITE_UNSCALE = 8

comptime FEW_THREADS = 256

# Blocks per SM the chunk size aims for. Swept on an H100 PCIe (114 SMs, FP32),
# ours/stock device time on `benchmarks/test_foreach.py`:
#                        2       4       8
#   mul_      4x1M    0.880   0.842   0.884
#   mul_      16x64k  0.390   0.391   0.596
#   addcmul_  4x1M    0.856   0.816   0.762
#   addcmul_  16x64k  0.412   0.438   0.651
# 8 starves the many-small-tensors regime (its chunk hits FEW_MIN_CHUNK, so each
# block does one vector step and the descriptor scan dominates); 2 gives up the
# big-tensor cases. Fitted on this card, but derived from the RUNTIME SM count,
# so a smaller or larger GPU scales with it rather than inheriting a grid.
comptime FEW_BLOCKS_PER_SM = 4

# The chunk size is rounded down to a multiple of this many elements, which
# keeps every chunk boundary 16-byte aligned for every dtype here (512 * 2
# bytes is the smallest case) so only a tensor's base pointer can be
# misaligned.
comptime FEW_CHUNK_GRAIN = 512
comptime FEW_MIN_CHUNK = 1024
# A ceiling only bounds per-block work; nothing depends on the exact value.
comptime FEW_MAX_CHUNK = 65_536

# Descriptors per launch. This bounds ONE launch argument, not the list: a
# longer TensorList becomes several launches of the same compiled kernel.
comptime FEW_DESC_CAP = 64


# The family roster, and the two sub-families whose members reach the SAME
# entry point and differ only in the code they pass it: mul/add/div all go to
# `enqueue_foreach_scalar_f32`, addcmul/addcdiv both go to
# `enqueue_foreach_addc_f32`. `_FEW_SOLO` is everything with a signature of
# its own. Splitting the roster this way is what lets the Apple dispatch be a
# `comptime for` over each sub-family plus one arm per solo op, and lets
# `_few_metal_covered` be computed from the very same lists the arms are
# written from, so the two cannot drift apart.
comptime _FEW_SCALAR_FAMILY = [FEW_MUL, FEW_ADD, FEW_DIV]
comptime _FEW_ADDC_FAMILY = [FEW_ADDCMUL, FEW_ADDCDIV]
comptime _FEW_SOLO = [
    FEW_MUL_TENSOR,
    FEW_LERP,
    FEW_SQRT,
    FEW_NONFINITE_UNSCALE,
]


@always_inline
def _few_addrs[op: Int]() -> Int:
    """Device addresses one metadata record carries for `op`."""
    comptime if op == FEW_LERP or op == FEW_SQRT:
        return 2
    elif op == FEW_ADDCMUL or op == FEW_ADDCDIV:
        return 3
    elif (
        op == FEW_MUL
        or op == FEW_ADD
        or op == FEW_DIV
        or op == FEW_MUL_TENSOR
        or op == FEW_NONFINITE_UNSCALE
    ):
        return 1
    else:
        # Not a fallthrough default: an op with no entry here would otherwise
        # get a plausible-looking arity and mispack its metadata in silence.
        comptime assert False, "no metadata arity for this foreach op"


@always_inline
def _few_fields[op: Int]() -> Int:
    """Ints one metadata record carries: the addresses then the numel."""
    return _few_addrs[op]() + 1


@always_inline
def _few_has_scalar[op: Int]() -> Bool:
    """Whether `op` takes one host scalar per tensor."""
    comptime if (
        op == FEW_MUL
        or op == FEW_ADD
        or op == FEW_DIV
        or op == FEW_ADDCMUL
        or op == FEW_ADDCDIV
    ):
        return True
    else:
        return False


@always_inline
def _few_label[op: Int]() -> StaticString:
    """Algorithm-and-regime fragment of the kernel name a profiler prints."""
    comptime if op == FEW_MUL:
        return "mul_scalar"
    elif op == FEW_ADD:
        return "add_scalar"
    elif op == FEW_DIV:
        return "div_scalar"
    elif op == FEW_MUL_TENSOR:
        return "mul_device_scalar"
    elif op == FEW_LERP:
        return "lerp_scalar"
    elif op == FEW_ADDCMUL:
        return "addcmul"
    elif op == FEW_ADDCDIV:
        return "addcdiv"
    elif op == FEW_SQRT:
        return "sqrt"
    elif op == FEW_NONFINITE_UNSCALE:
        return "nonfinite_check_unscale"
    else:
        # A wrong name here would be printed to users by CUPTI/Nsight and would
        # name the wrong algorithm, so this is a build error, not a default.
        comptime assert False, "no kernel-name fragment for this foreach op"


@always_inline
def _few_metal_covered[op: Int]() -> Bool:
    """Whether `_foreach_ew_enqueue_apple` has an arm that launches `op`.

    Computed from the same three rosters the arms are generated from, so it
    answers the question that matters -- "will this op actually launch?" --
    rather than restating a list by hand.
    """
    comptime for candidate in _FEW_SCALAR_FAMILY:
        comptime if op == candidate:
            return True
    comptime for candidate in _FEW_ADDC_FAMILY:
        comptime if op == candidate:
            return True
    comptime for candidate in _FEW_SOLO:
        comptime if op == candidate:
            return True
    return False


@always_inline
def _few_scalar_family_code[op: Int]() -> Int:
    """The `FES_*` code the shared scalar entry point takes for `op`.

    `FEW_*` and `FES_*` are separate namespaces that happen to agree
    numerically today; map them rather than depending on that.
    """
    comptime if op == FEW_MUL:
        return FES_MUL
    elif op == FEW_ADD:
        return FES_ADD
    elif op == FEW_DIV:
        return FES_DIV
    else:
        comptime assert False, "not a foreach scalar-family op"


@always_inline
def _few_addc_family_code[op: Int]() -> Int:
    """The `FEA_*` code the shared addc entry point takes for `op`."""
    comptime if op == FEW_ADDCMUL:
        return FEA_ADDCMUL
    elif op == FEW_ADDCDIV:
        return FEA_ADDCDIV
    else:
        comptime assert False, "not a foreach addc-family op"


struct ForeachEwDesc(
    DevicePassable,
    ImplicitlyCopyable,
    TrivialRegisterPassable,
):
    """One tensor of the list: its addresses, length, and chunk prefix sum.

    `addr1` / `addr2` are the extra operands (unused ones stay zero and are
    never dereferenced: which addresses an op reads is a compile-time fact).
    """

    comptime device_type: AnyType = Self

    var addr0: Int
    var addr1: Int
    var addr2: Int
    var numel: Int
    var chunk_end: Int

    def __init__(
        out self, addr0: Int, addr1: Int, addr2: Int, numel: Int, chunk_end: Int
    ):
        self.addr0 = addr0
        self.addr1 = addr1
        self.addr2 = addr2
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
        return "ForeachEwDesc"


@always_inline
def empty_foreach_ew_desc() -> ForeachEwDesc:
    return ForeachEwDesc(0, 0, 0, 0, 0)


@always_inline
def _few_element[
    dtype: DType, op: Int, width: Int, alignment: Int
](
    a_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    b_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    c_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    out_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    flag_ptr: Pointer[Float32, MutUntrackedOrigin],
    index: Int,
    scalar: Float32,
    weight: Float32,
    one_minus_weight: Float32,
    low_branch: Int,
):
    """`width` elements of one op at `index`, computed in FP32.

    The single definition of every op's element math. Both the vectorized body
    and the scalar peel/tail of `_foreach_ew_kernel` are calls to this, and so
    is nothing else: an op's arithmetic exists once.
    """
    var a = a_ptr.unsafe_load[width=width, alignment=alignment](index).cast[
        DType.float32
    ]()
    var result: SIMD[DType.float32, width]
    comptime if op == FEW_MUL or op == FEW_MUL_TENSOR:
        result = a * scalar
    elif op == FEW_ADD:
        result = a + scalar
    elif op == FEW_DIV:
        result = a / scalar
    elif op == FEW_SQRT:
        result = ieee_sqrt(a)
    elif op == FEW_NONFINITE_UNSCALE:
        # ATen's AmpKernels.cu: every writer stores the same 1, so the race is
        # benign and needs no atomic.
        if not isfinite(a).reduce_and():
            flag_ptr[] = 1.0
        result = a if scalar == 1.0 else a * scalar
    elif op == FEW_LERP:
        # ATen's numerically stable branch pair, selected on the host exactly
        # like `fast_aten_lerp` does. The scale-and-add is written as one
        # expression, so the backend may contract it into an fma -- which is
        # what ATen's own CUDA lerp kernel compiles to, and at most one ulp
        # from the two-kernel sequential composition this replaces. (The
        # Metal path in `foreach_elementwise_kernels` blocks that contraction
        # with `_no_fuse`; the volatile round-trip it needs is not worth
        # paying here for one ulp.)
        var finish = b_ptr.unsafe_load[width=width, alignment=alignment](
            index
        ).cast[DType.float32]()
        var difference = finish - a
        if low_branch == 0:
            result = finish - one_minus_weight * difference
        else:
            result = a + weight * difference
    elif op == FEW_ADDCMUL or op == FEW_ADDCDIV:
        var b = b_ptr.unsafe_load[width=width, alignment=alignment](index).cast[
            DType.float32
        ]()
        var c = c_ptr.unsafe_load[width=width, alignment=alignment](index).cast[
            DType.float32
        ]()
        comptime if op == FEW_ADDCMUL:
            result = a + scalar * (b * c)
        else:
            result = a + scalar * (b / c)
    else:
        # No default arithmetic: an op with no arm here would otherwise write
        # back its own input, or another op's answer, without a diagnostic.
        comptime assert False, "no element math for this foreach op"
    out_ptr.unsafe_store[width=width, alignment=alignment](
        index, result.cast[dtype]()
    )


@__name(t"foreach_{_few_label[op]()}_desc_{dtype}_t{FEW_THREADS}")
def _foreach_ew_kernel[
    dtype: DType, op: Int
](
    descs: Array[ForeachEwDesc, FEW_DESC_CAP],
    scalars: Array[Float32, FEW_DESC_CAP],
    desc_count_arg: Int64,
    chunk_elements_arg: Int64,
    scalar_addr_arg: Int64,
    weight: Float32,
    one_minus_weight: Float32,
    low_branch_arg: Int64,
    flag_addr_arg: Int64,
):
    """One block per chunk of the concatenation of the list."""
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var desc_count = Int(desc_count_arg)
    var chunk_elements = Int(chunk_elements_arg)
    var low_branch = Int(low_branch_arg)

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

    var scalar = Float32(0.0)
    comptime if _few_has_scalar[op]():
        scalar = scalars[desc_index]
    comptime if op == FEW_MUL_TENSOR:
        scalar = _make_ptr[dtype](Int(scalar_addr_arg))[unsafe_offset=0].cast[
            DType.float32
        ]()
    comptime if op == FEW_NONFINITE_UNSCALE:
        scalar = _make_ptr[DType.float32](Int(scalar_addr_arg))[unsafe_offset=0]
    var flag_ptr = _make_ptr[DType.float32](Int(flag_addr_arg))

    var a_ptr = _make_ptr[dtype](desc.addr0)
    var b_ptr = _make_ptr[dtype](desc.addr1)
    var c_ptr = _make_ptr[dtype](desc.addr2)
    var out_ptr = a_ptr
    comptime if op == FEW_SQRT:
        out_ptr = b_ptr

    # 16-byte accesses need a 16-byte-aligned address, and a contiguous
    # tensor may be an offset view of one. Every chunk boundary is 16-byte
    # aligned (FEW_CHUNK_GRAIN), so one peel of at most 16 bytes per chunk
    # lands the vector body on a boundary -- as long as all of the op's
    # operands share the base residue, which co-allocated tensor lists do.
    comptime item = size_of[dtype]()
    comptime vector = 16 // item
    var residue = desc.addr0 % 16
    var vectorizable = residue % item == 0
    comptime if _few_addrs[op]() >= 2:
        if desc.addr1 % 16 != residue:
            vectorizable = False
    comptime if _few_addrs[op]() >= 3:
        if desc.addr2 % 16 != residue:
            vectorizable = False

    var lane = Int(thread_idx.x)
    if vectorizable:
        var body = begin + min(((16 - residue) % 16) // item, end - begin)
        var head = begin + lane
        while head < body:
            _few_element[dtype, op, 1, item](
                a_ptr,
                b_ptr,
                c_ptr,
                out_ptr,
                flag_ptr,
                head,
                scalar,
                weight,
                one_minus_weight,
                low_branch,
            )
            head += FEW_THREADS
        var index = body + lane * vector
        while index + vector <= end:
            _few_element[dtype, op, vector, 16](
                a_ptr,
                b_ptr,
                c_ptr,
                out_ptr,
                flag_ptr,
                index,
                scalar,
                weight,
                one_minus_weight,
                low_branch,
            )
            index += FEW_THREADS * vector
        var tail = body + ((end - body) // vector) * vector + lane
        while tail < end:
            _few_element[dtype, op, 1, item](
                a_ptr,
                b_ptr,
                c_ptr,
                out_ptr,
                flag_ptr,
                tail,
                scalar,
                weight,
                one_minus_weight,
                low_branch,
            )
            tail += FEW_THREADS
    else:
        var index = begin + lane
        while index < end:
            _few_element[dtype, op, 1, item](
                a_ptr,
                b_ptr,
                c_ptr,
                out_ptr,
                flag_ptr,
                index,
                scalar,
                weight,
                one_minus_weight,
                low_branch,
            )
            index += FEW_THREADS


def foreach_ew_chunk_elements(total_elements: Int, ctx: DeviceContext) -> Int:
    """Elements per block: the whole list's work spread over the whole device.

    Apple keeps the fixed chunk its fixed-arity kernels are written against.
    """
    comptime if has_apple_gpu_accelerator():
        return FOREACH_EW_CHUNK
    else:
        var blocks = FEW_BLOCKS_PER_SM * _device_sm_count(ctx)
        var chunk = (
            ceildiv(total_elements, blocks) if blocks > 0 else total_elements
        )
        chunk = (chunk // FEW_CHUNK_GRAIN) * FEW_CHUNK_GRAIN
        if chunk < FEW_MIN_CHUNK:
            chunk = FEW_MIN_CHUNK
        return min(chunk, FEW_MAX_CHUNK)


@always_inline
def _few_backfill(mut addrs: Array[Int, FOREACH_EW_SLOTS], slot_count: Int):
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


def _foreach_ew_enqueue_apple[
    dtype: DType, op: Int
](
    descs: Array[ForeachEwDesc, FEW_DESC_CAP],
    scalars: Array[Float32, FEW_DESC_CAP],
    desc_count: Int,
    scalar_addr: Int,
    weight: Float32,
    one_minus_weight: Float32,
    low_branch: Int,
    flag_addr: Int,
    ctx: DeviceContext,
) raises:
    """Same descriptors, Metal's fixed-arity ABI: FOREACH_EW_SLOTS tensors per
    launch, every pointer a real kernel argument."""
    comptime if dtype != DType.float32:
        raise Error("mojo foreach: the Apple batched path is float32 only")
    else:
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
                first_addrs[slot] = desc.addr0
                second_addrs[slot] = desc.addr1
                third_addrs[slot] = desc.addr2
                numels[slot] = desc.numel
                chunk_ends[slot] = desc.chunk_end - group_first_chunk
                group_scalars[slot] = scalars[desc_index]
                desc_index += 1
                slot += 1
            var used_slots = slot
            var group_chunks = chunk_ends[used_slots - 1]
            while slot < FOREACH_EW_SLOTS:
                chunk_ends[slot] = group_chunks
                slot += 1
            if group_chunks == 0:
                continue
            _few_backfill(first_addrs, used_slots)
            comptime if _few_addrs[op]() >= 2:
                _few_backfill(second_addrs, used_slots)
            comptime if _few_addrs[op]() >= 3:
                _few_backfill(third_addrs, used_slots)

            # One arm per sub-family, plus one per solo op. `launched` is a
            # compile-time constant (every assignment sits in a `comptime if`),
            # so for a covered op the guard below folds away entirely.
            var launched = False
            comptime for candidate in _FEW_SCALAR_FAMILY:
                comptime if op == candidate:
                    enqueue_foreach_scalar_f32[
                        _few_scalar_family_code[candidate]()
                    ](
                        first_addrs,
                        chunk_ends,
                        numels,
                        group_scalars,
                        group_chunks,
                        ctx,
                    )
                    launched = True
            comptime for candidate in _FEW_ADDC_FAMILY:
                comptime if op == candidate:
                    enqueue_foreach_addc_f32[
                        _few_addc_family_code[candidate]()
                    ](
                        first_addrs,
                        second_addrs,
                        third_addrs,
                        chunk_ends,
                        numels,
                        group_scalars,
                        group_chunks,
                        ctx,
                    )
                    launched = True
            comptime if op == FEW_MUL_TENSOR:
                enqueue_foreach_mul_tensor_f32(
                    first_addrs,
                    chunk_ends,
                    numels,
                    scalar_addr,
                    group_chunks,
                    ctx,
                )
                launched = True
            comptime if op == FEW_LERP:
                enqueue_foreach_lerp_f32(
                    first_addrs,
                    second_addrs,
                    chunk_ends,
                    numels,
                    weight,
                    one_minus_weight,
                    low_branch,
                    group_chunks,
                    ctx,
                )
                launched = True
            comptime if op == FEW_NONFINITE_UNSCALE:
                enqueue_foreach_nonfinite_unscale_f32(
                    first_addrs,
                    chunk_ends,
                    numels,
                    scalar_addr,
                    flag_addr,
                    group_chunks,
                    ctx,
                )
                launched = True
            comptime if op == FEW_SQRT:
                enqueue_foreach_sqrt_f32(
                    first_addrs,
                    second_addrs,
                    chunk_ends,
                    numels,
                    group_chunks,
                    ctx,
                )
                launched = True
            if not launched:
                # Unreachable once `_few_metal_covered` holds, and compiled
                # away with it. It stays because the failure it catches -- an
                # op whose arm went missing -- is otherwise a mutating op that
                # silently does nothing at all.
                raise Error(
                    "mojo foreach ",
                    _few_label[op](),
                    ": no Apple launch for this op",
                )


def foreach_ew_enqueue[
    dtype: DType, op: Int
](
    descs: Array[ForeachEwDesc, FEW_DESC_CAP],
    scalars: Array[Float32, FEW_DESC_CAP],
    desc_count: Int,
    total_chunks: Int,
    chunk_elements: Int,
    scalar_addr: Int,
    weight: Float32,
    one_minus_weight: Float32,
    low_branch: Int,
    flag_addr: Int,
    ctx: DeviceContext,
) raises:
    # Checked on EVERY target, not just Apple: the Apple arms are the half of
    # the family that no CI machine here can run, so an op added without one
    # has to fail the build wherever it is first compiled rather than turn into
    # a mutating op that silently does nothing on a Mac.
    comptime assert _few_metal_covered[
        op
    ](), "this foreach op has no arm in _foreach_ew_enqueue_apple"
    if desc_count <= 0 or total_chunks <= 0:
        return
    comptime if has_apple_gpu_accelerator():
        _foreach_ew_enqueue_apple[dtype, op](
            descs,
            scalars,
            desc_count,
            scalar_addr,
            weight,
            one_minus_weight,
            low_branch,
            flag_addr,
            ctx,
        )
    else:
        _enqueue_cached[_foreach_ew_kernel[dtype, op]](
            ctx,
            total_chunks,
            1,
            1,
            FEW_THREADS,
            descs,
            scalars,
            Int64(desc_count),
            Int64(chunk_elements),
            Int64(scalar_addr),
            weight,
            one_minus_weight,
            Int64(low_branch),
            Int64(flag_addr),
        )
