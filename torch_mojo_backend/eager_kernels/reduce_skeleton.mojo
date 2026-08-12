# ===----------------------------------------------------------------------=== #
# One reduction skeleton, parametrized over the ACCUMULATOR.
#
# Every scalar-payload reduction this backend implements -- sum, mean, amax,
# amin, max, min, the L2 vector norm, any and all -- is the same three moving
# parts over the same geometry, and they used to be nine hand-written launches
# with three different launch policies between them. What actually differs
# between two of them is four lines of algebra: the accumulator dtype, its
# identity, how an input element maps into it, how two accumulators combine,
# and how a finished accumulator becomes an output element. `ReduceOp` is that
# four-line interface and everything below is written once against it.
#
# GEOMETRY. The input is a contiguous buffer viewed as (outer, reduce, inner):
# element (o, r, i) sits at `(o * reduce + r) * inner + i` and output (o, i) at
# `o * inner + i`. `inner == 1` is a reduction over the trailing dims -- rows
# of a (rows, cols) buffer, a full reduction being rows == 1 -- and `inner > 1`
# is a reduction over an interior or leading interval, which the Python side
# used to pay a full permuted materialization for. This is the same geometry
# the variance moments and the arg-reductions already use, deliberately: one
# shape of reduction in this codebase, not three.
#
# SPLITTING. The number of outputs is not the amount of parallelism available;
# a full reduction has ONE output and would otherwise run in one block while
# every other SM idles (that is exactly what `_reduce_block_kernel` did, and it
# cost 12x stock torch on a 16.7M-element sum). So the reduce axis is split
# across blocks whenever the unsplit grid cannot fill the device, with the
# split count derived from the RUNTIME SM count. splits == 1 finalizes inside
# stage 1 and launches nothing else; splits > 1 writes one accumulator per
# (output, split) into a workspace that a merge kernel reduces. A workspace
# plus a separate merge is preferred to `Atomic.fetch_add` (sequentially
# consistent by default): it is deterministic and it is the house rule.
#
# IDENTITY, NOT ZERO. The merge cannot memset a workspace and add into it,
# because `amax`'s identity is -inf and `all`'s is true. Every accumulator
# supplies its own identity and every stage seeds with it, so a shard that
# receives an empty slice (the last shard when `splits` does not divide the
# extent) contributes a true no-op instead of a wrong zero.
#
# NaN. Torch's rules differ per op and so they live in the accumulator, not in
# the skeleton: `max`/`min`/`amax`/`amin` PROPAGATE NaN (verified against stock
# torch: `torch.amax([1, nan]) -> nan`), which an ordinary `max()` does not do,
# so `MaxOp.combine` takes the candidate when it is greater OR when it is NaN;
# `sum` and `mean` inherit NaN through the arithmetic; `any`/`all` treat NaN as
# truthy, which is a nonzero test and not a comparison.
# ===----------------------------------------------------------------------=== #

from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    thread_idx,
)
from std.gpu.primitives import warp
from std.gpu.primitives.warp import shuffle_down
from std.math import ceildiv, isnan, sqrt
from std.memory import stack_allocation
from std.sys.info import has_accelerator, size_of
from std.utils.coord import Coord
from std.utils.numerics import max_or_inf, min_or_neg_inf
from std.utils.static_tuple import StaticTuple

from std.python._cpython import PyObjectPtr
from std.utils.index import IndexList

from op_utils import (
    MAX_RANK,
    TensorSpec,
    _adjacent_reduce_geom,
    _check_into_sized,
    _device_sm_count,
    _enqueue_cached,
    _make_ptr,
    _parallel_for,
    _reduce_spec_geom,
    _spec_ptr,
    _vec16_phase,
)
from variant_gates import _dtype_arg_on, _dtype_supported


comptime RED_THREADS = 256

# Stage-1 blocks in flight targeted when the reduction has too few outputs to
# fill the device on its own. FITTED ON AN H100 PCIe (114 SMs) and nowhere
# else; scaling by the RUNTIME sm count keeps the SHAPE of the grid portable,
# but the position of the optimum below is this card's. Swept here for the
# ONE-slot scalar payload (device us of `torch.sum`, min of three 200-launch
# bursts under the GPU flock; the last row repeats the chosen value at the end
# of the sweep and reproduces the first to 0.2%, so nothing below is drift):
#
#   blocks/SM   f32 16.7M full   f32 4096^2 d0   bf16 16.7M full   bf16 4096^2 d0
#           1            47.59           71.64             11.50            30.14
#           2            40.77           48.70              9.51            20.44
#           4            40.26           43.17             10.02            16.05
#           8            39.66           43.93             10.82            19.70
#          16            39.86           51.06             15.34            26.36
#   4 (again)            40.33           43.18             10.03            16.04
#
# Below 4 the device is starved; above it the workspace the extra shards write
# (and the merge reads back) costs more than the parallelism buys. 4 is best or
# within 1.5% of best on three of the four, and the two that prefer a neighbour
# lose far more at the other end (bf16 full likes 2 by 5%, but bf16 dim=0 at 2
# is 27% worse). This is the same optimum, in the same place, that the variance
# moments found for a TWO-slot payload (MOMENT_BLOCKS_PER_SM): re-swept here
# because that is not something one payload size may assume of another, and it
# happened to agree.
comptime RED_BLOCKS_PER_SM = 4

# When the contiguous-axis kernel gives a whole 256-thread block to one row it
# pays a cross-warp fold (a shared-memory slot per warp plus a barrier) per row.
# That is free on a long row and ruinous on a short one, so below this many
# 16-byte vectors PER THREAD the row goes to a WARP instead: eight rows per
# block, no barrier and no shared memory at all, the fold collapsing to five
# shuffles. Swept on an H100 PCIe over 4096x4096 trailing-axis reductions
# (device us, the two regimes measured directly against each other):
#
#   case            vectors/thread   block-per-row   warp-per-row
#   bf16 sum d1                  2           17.10           8.49
#   bf16 amax d1                 2           19.57          10.66
#   bool all d1                  1           16.04          10.33
#   bool any d1                  1           17.81          12.53
#   f32  sum d1                  4           36.41          38.25
#   f32  amax d1                 4           37.23          37.95
#
# The warp form is worth 1.6-2.0x at one or two vectors per thread and loses
# 2-5% at four, so the crossover lies between; 4 puts every measured case on
# its own better side. Fitted on this card -- what travels is the shape of the
# rule (amortize the fold against the loads), not the number.
comptime RED_MIN_VECS_PER_THREAD = 4

# Smallest reduction, in ELEMENTS READ, worth splitting at all. Below it the
# whole thing is LAUNCH-bound -- stage 1 costs about what an empty launch costs
# whether it reads four thousand elements or none -- so the merge launch is
# pure loss, and this is the guard that makes the small end degrade cleanly to
# a single launch. It is a total, not a per-axis extent, because the two
# geometries measure their per-split floors in different units (elements per
# block on the contiguous axis, rows per lane on the strided one) and only the
# total says how long stage 1 will actually take.
#
# Swept on an H100 PCIe against the unsplit path, f32 full reductions (device
# us, min of three 400-launch bursts; "unsplit" is the same build with the
# guard set unreachably high):
#
#   elements    2048   4096   8192   16384   32768   65536   262144
#   unsplit     1.65   1.87   2.21    2.92    4.28    7.13     23.51
#   split       3.33   3.34   3.36    3.38    3.38    3.40      4.35
#
# The split path is a flat ~3.35us floor (two launches) and the unsplit one
# grows with the data, so they cross between 16384 and 32768; 32768 puts each
# measured size on its own better side. Fitted on f32: bf16 reads half the
# bytes, so its true crossover sits at a larger element count and this constant
# splits it very slightly early (a few hundred ns at one size). Fitted on this
# card.
comptime RED_MIN_SPLIT_ELEMS = 32768

# Floor on the reduce extent handed to one split of the STRIDED kernel, in rows
# of the reduce axis (every lane of a strided block walks the whole shard, so
# this is elements per lane). It is what keeps the fill target from shattering
# a short reduce axis into shards too small to pay for their workspace slot.
#
# The benchmark suite cannot see this constant at all: on 4096x4096 dim=0 the
# split count is `min(by_fill, by_work) = min(29, 4096/floor)` and by_fill wins
# for every floor from 4 to 32. These shapes can. Device us of `torch.sum(...,
# dim=0)`, f32, min of three 400-launch bursts under the GPU flock; the last
# row repeats the chosen value and reproduces the first to 0.01us:
#
#   rows/split   64x512   256x2048   512x8192   357x789
#            4     4.18       8.63       8.07      9.72
#            8     3.94       6.41       8.08      6.45
#           16     4.03       5.37       8.07      5.32
#           32     4.71       5.30       8.07      5.61
#   16 (again)     4.03       5.37       8.07      5.32
#
# 512x8192 is the by_fill-bound case and is flat, as predicted. Of the rest,
# the smallest reduce axis wants MORE splits and the mid ones want fewer, so
# the constant is a genuine trade: 16 costs 2% against 8 on 64x512 and buys
# 16-18% on the two mid shapes, while 32 costs 20% on 64x512. Fitted on this
# card; what travels is the shape of the rule, not the number.
comptime RED_MIN_ROWS_PER_SPLIT = 16


trait ReduceOp:
    """The payload of one reduction: identity, map, combine, finalize.

    Deliberately NOT parametrized by dtype -- a `ReduceOp` is the algebra, and
    the element dtype is a parameter of the skeleton that instantiates it. That
    is what lets ONE bridge function serve every scalar reduction spec op:
    `_rowred_spec_into_go[SumOp]` and `_rowred_spec_into_go[MaxOp]` are the
    same code.
    """

    comptime name: StaticString
    """Kernel-name token, e.g. "sum" -> `reduce_contig_sum_float32`."""

    comptime dtypes: List[DType]
    """Operand dtypes this reduction accepts, in dispatch order."""

    comptime errors_on_empty_axis: Bool
    """Does torch REFUSE this reduction when the reduce axis has length zero?

    Only the selection ops do -- "amax(): Expected reduction dim 1 to have
    non-zero size", because there is no element to select. Every other
    accumulator answers with its identity finalized over n == 0, which is
    torch's answer too: sum -> 0, the L2 norm -> 0, all -> true, any -> false,
    and mean -> nan (0/0). A reduction with no OUTPUTS is never an error for
    anyone; it writes nothing."""

    @staticmethod
    def acc_dtype[in_dt: DType]() -> DType:
        """The accumulator dtype for an operand of `in_dt`."""
        ...

    @staticmethod
    def out_dtype[in_dt: DType]() -> DType:
        """The output element dtype for an operand of `in_dt`."""
        ...

    @staticmethod
    def identity[acc: DType, width: SIMDLength]() -> SIMD[acc, width]:
        """The value an empty shard contributes: a true no-op under combine."""
        ...

    @staticmethod
    def map[
        in_dt: DType, width: SIMDLength, //, acc: DType
    ](x: SIMD[in_dt, width]) -> SIMD[acc, width]:
        """One input element as an accumulator (cast, |x|^2, nonzero test...).
        """
        ...

    @staticmethod
    def combine[
        dtype: DType, width: SIMDLength
    ](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
        """Associative, commutative merge of two accumulators.

        Generic over dtype (rather than pinned to `acc_dtype`) so it satisfies
        `warp.reduce`'s reduction-function signature directly.
        """
        ...

    @staticmethod
    def finish[
        acc: DType, //, out_dt: DType
    ](a: Scalar[acc], n: Int) -> Scalar[out_dt]:
        """The finished accumulator over `n` elements as an output element.

        `n` is the FULL reduce extent, not the shard's, so a scaling finalize
        (mean) is folded in here instead of costing a second launch.
        """
        ...


# ---------------------------------------------------------------------------
# The accumulators. Four lines of algebra each; everything else is shared.
# ---------------------------------------------------------------------------

# Floating-point reductions accumulate in float32 (matching torch); integer
# ones accumulate in their own dtype.
comptime SCALAR_DTYPES: List[DType] = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.int64,
    DType.int32,
]

comptime FLOAT_ONLY_DTYPES: List[DType] = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
]

comptime TRUTHY_DTYPES: List[DType] = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.int64,
    DType.int32,
    DType.int16,
    DType.int8,
    DType.uint8,
    DType.bool,
]


@always_inline
def _float_acc[in_dt: DType]() -> DType:
    """float rows accumulate in float32; int rows in their own dtype."""
    comptime if in_dt.is_floating_point():
        return DType.float32
    else:
        return in_dt


struct SumOp(ReduceOp):
    """sum: additive, identity 0, NaN inherited through the addition."""

    comptime name = "sum"
    comptime dtypes = SCALAR_DTYPES
    comptime errors_on_empty_axis = False

    @staticmethod
    def acc_dtype[in_dt: DType]() -> DType:
        return _float_acc[in_dt]()

    @staticmethod
    def out_dtype[in_dt: DType]() -> DType:
        return in_dt

    @staticmethod
    def identity[acc: DType, width: SIMDLength]() -> SIMD[acc, width]:
        return SIMD[acc, width](0)

    @staticmethod
    def map[
        in_dt: DType, width: SIMDLength, //, acc: DType
    ](x: SIMD[in_dt, width]) -> SIMD[acc, width]:
        return x.cast[acc]()

    @staticmethod
    def combine[
        dtype: DType, width: SIMDLength
    ](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return a + b

    @staticmethod
    def finish[
        acc: DType, //, out_dt: DType
    ](a: Scalar[acc], n: Int) -> Scalar[out_dt]:
        return a.cast[out_dt]()


struct MeanOp(ReduceOp):
    """mean: sum with the 1/n folded into the finalize.

    Deliberately NOT `sum` plus a scaling launch -- the scale is one multiply
    on a value already in a register at the end of the merge.
    """

    comptime name = "mean"
    comptime dtypes = FLOAT_ONLY_DTYPES
    # mean over an empty axis is nan in torch, which 0/0 produces here.
    comptime errors_on_empty_axis = False

    @staticmethod
    def acc_dtype[in_dt: DType]() -> DType:
        return DType.float32

    @staticmethod
    def out_dtype[in_dt: DType]() -> DType:
        return in_dt

    @staticmethod
    def identity[acc: DType, width: SIMDLength]() -> SIMD[acc, width]:
        return SIMD[acc, width](0)

    @staticmethod
    def map[
        in_dt: DType, width: SIMDLength, //, acc: DType
    ](x: SIMD[in_dt, width]) -> SIMD[acc, width]:
        return x.cast[acc]()

    @staticmethod
    def combine[
        dtype: DType, width: SIMDLength
    ](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return a + b

    @staticmethod
    def finish[
        acc: DType, //, out_dt: DType
    ](a: Scalar[acc], n: Int) -> Scalar[out_dt]:
        return (a / Scalar[acc](n)).cast[out_dt]()


struct NormL2Op(ReduceOp):
    """linalg_vector_norm(ord=2): sum of squares, square root at the end.

    Composed in Python until now (mul -> sum -> sqrt), which meant a full
    temporary the size of the input and three launches for what is one pass.
    No rescaling: torch's own f32 vector_norm overflows to inf on large
    operands and this matches it element for element.
    """

    comptime name = "norml2"
    comptime dtypes = FLOAT_ONLY_DTYPES
    comptime errors_on_empty_axis = False

    @staticmethod
    def acc_dtype[in_dt: DType]() -> DType:
        return DType.float32

    @staticmethod
    def out_dtype[in_dt: DType]() -> DType:
        return in_dt

    @staticmethod
    def identity[acc: DType, width: SIMDLength]() -> SIMD[acc, width]:
        return SIMD[acc, width](0)

    @staticmethod
    def map[
        in_dt: DType, width: SIMDLength, //, acc: DType
    ](x: SIMD[in_dt, width]) -> SIMD[acc, width]:
        var v = x.cast[acc]()
        return v * v

    @staticmethod
    def combine[
        dtype: DType, width: SIMDLength
    ](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return a + b

    @staticmethod
    def finish[
        acc: DType, //, out_dt: DType
    ](a: Scalar[acc], n: Int) -> Scalar[out_dt]:
        return sqrt(a).cast[out_dt]()


struct MaxOp(ReduceOp):
    """amax / max: selection, identity -inf, NaN PROPAGATED (torch's rule)."""

    comptime name = "max"
    comptime dtypes = SCALAR_DTYPES
    comptime errors_on_empty_axis = True

    @staticmethod
    def acc_dtype[in_dt: DType]() -> DType:
        return _float_acc[in_dt]()

    @staticmethod
    def out_dtype[in_dt: DType]() -> DType:
        return in_dt

    @staticmethod
    def identity[acc: DType, width: SIMDLength]() -> SIMD[acc, width]:
        return SIMD[acc, width](min_or_neg_inf[acc]())

    @staticmethod
    def map[
        in_dt: DType, width: SIMDLength, //, acc: DType
    ](x: SIMD[in_dt, width]) -> SIMD[acc, width]:
        return x.cast[acc]()

    @staticmethod
    def combine[
        dtype: DType, width: SIMDLength
    ](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime if dtype.is_floating_point():
            # `max(a, b)` lowers to maxnum, which SUPPRESSES NaN; torch
            # propagates it. Taking b when it is greater or when it is NaN
            # gives a combine that is still associative and commutative (any
            # NaN anywhere wins) at the cost of one extra compare per element,
            # which a bandwidth-bound reduction does not notice.
            return (b.gt(a) | isnan(b)).select(b, a)
        else:
            return max(a, b)

    @staticmethod
    def finish[
        acc: DType, //, out_dt: DType
    ](a: Scalar[acc], n: Int) -> Scalar[out_dt]:
        return a.cast[out_dt]()


struct MinOp(ReduceOp):
    """amin / min: the mirror of MaxOp, NaN equally propagated."""

    comptime name = "min"
    comptime dtypes = SCALAR_DTYPES
    comptime errors_on_empty_axis = True

    @staticmethod
    def acc_dtype[in_dt: DType]() -> DType:
        return _float_acc[in_dt]()

    @staticmethod
    def out_dtype[in_dt: DType]() -> DType:
        return in_dt

    @staticmethod
    def identity[acc: DType, width: SIMDLength]() -> SIMD[acc, width]:
        return SIMD[acc, width](max_or_inf[acc]())

    @staticmethod
    def map[
        in_dt: DType, width: SIMDLength, //, acc: DType
    ](x: SIMD[in_dt, width]) -> SIMD[acc, width]:
        return x.cast[acc]()

    @staticmethod
    def combine[
        dtype: DType, width: SIMDLength
    ](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime if dtype.is_floating_point():
            return (b.lt(a) | isnan(b)).select(b, a)
        else:
            return min(a, b)

    @staticmethod
    def finish[
        acc: DType, //, out_dt: DType
    ](a: Scalar[acc], n: Int) -> Scalar[out_dt]:
        return a.cast[out_dt]()


struct AnyOp(ReduceOp):
    """any: OR of a nonzero test. NaN is truthy (torch's rule).

    The accumulator is int32 rather than bool: the warp shuffle the block fold
    is built on has no 8-bit form (`std.gpu.primitives.warp._shuffle` refuses
    it with "unhandled shuffle dtype"), and a bool-width workspace would save
    a few KB on a reduction that reads megabytes. The values are only ever 0
    or 1, so OR and AND are exact.
    """

    comptime name = "any"
    comptime dtypes = TRUTHY_DTYPES
    comptime errors_on_empty_axis = False

    @staticmethod
    def acc_dtype[in_dt: DType]() -> DType:
        return DType.int32

    @staticmethod
    def out_dtype[in_dt: DType]() -> DType:
        return DType.bool

    @staticmethod
    def identity[acc: DType, width: SIMDLength]() -> SIMD[acc, width]:
        return SIMD[acc, width](0)

    @staticmethod
    def map[
        in_dt: DType, width: SIMDLength, //, acc: DType
    ](x: SIMD[in_dt, width]) -> SIMD[acc, width]:
        # `.eq` is an ORDERED equal (NaN == 0 -> False), so the inverse maps
        # NaN -> true, matching torch. `.cast[bool]()` would lower to an
        # ordered `ne` and wrongly map NaN -> false.
        return (~x.eq(SIMD[in_dt, width]())).cast[acc]()

    @staticmethod
    def combine[
        dtype: DType, width: SIMDLength
    ](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return a | b

    @staticmethod
    def finish[
        acc: DType, //, out_dt: DType
    ](a: Scalar[acc], n: Int) -> Scalar[out_dt]:
        return a.ne(Scalar[acc](0)).cast[out_dt]()


struct AllOp(ReduceOp):
    """all: AND of a nonzero test, identity true. int32 accumulator, as AnyOp.
    """

    comptime name = "all"
    comptime dtypes = TRUTHY_DTYPES
    comptime errors_on_empty_axis = False

    @staticmethod
    def acc_dtype[in_dt: DType]() -> DType:
        return DType.int32

    @staticmethod
    def out_dtype[in_dt: DType]() -> DType:
        return DType.bool

    @staticmethod
    def identity[acc: DType, width: SIMDLength]() -> SIMD[acc, width]:
        return SIMD[acc, width](1)

    @staticmethod
    def map[
        in_dt: DType, width: SIMDLength, //, acc: DType
    ](x: SIMD[in_dt, width]) -> SIMD[acc, width]:
        return (~x.eq(SIMD[in_dt, width]())).cast[acc]()

    @staticmethod
    def combine[
        dtype: DType, width: SIMDLength
    ](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return a & b

    @staticmethod
    def finish[
        acc: DType, //, out_dt: DType
    ](a: Scalar[acc], n: Int) -> Scalar[out_dt]:
        return a.ne(Scalar[acc](0)).cast[out_dt]()


# ---------------------------------------------------------------------------
# Shared folds
# ---------------------------------------------------------------------------


@always_inline
def _lane_fold[
    Op: ReduceOp, acc: DType, width: SIMDLength
](v: SIMD[acc, width]) -> Scalar[acc]:
    """Collapse a SIMD accumulator to one, halving with `Op.combine`."""
    comptime if width == 1:
        return v[0]
    else:
        comptime half = width // 2
        return _lane_fold[Op, acc, half](
            Op.combine(v.slice[half, offset=0](), v.slice[half, offset=half]())
        )


@always_inline
def _block_fold[
    Op: ReduceOp, acc: DType, threads: Int
](tid: Int, v: Scalar[acc]) -> Scalar[acc]:
    """One accumulator per thread -> the block's, VALID IN THREAD 0 ONLY.

    A warp shuffle reduction (`shuffle_down`, so lane 0 of each warp holds its
    warp's answer) plus one shared-memory slot per warp, folded by thread 0.
    Deliberately not broadcast: every caller here is a `if tid == 0:` store, so
    a broadcast would only add `threads / WARP_SIZE` shared loads on 255 of
    every 256 threads -- measurable on a short row, where the fold is a large
    share of the block's whole life.
    """

    @always_inline
    @parameter
    def cb[
        dtype: DType, width: SIMDLength
    ](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return Op.combine(x, y)

    var lane_total = warp.reduce[shuffle_down, cb](v)
    comptime n_warps = threads // WARP_SIZE
    comptime if n_warps == 1:
        return lane_total
    else:
        var smem = stack_allocation[
            n_warps, acc, address_space=AddressSpace.SHARED
        ]()
        if tid % WARP_SIZE == 0:
            smem[tid // WARP_SIZE] = lane_total
        barrier()
        var total = Op.identity[acc, 1]()[0]
        if tid == 0:

            @parameter
            for k in range(n_warps):
                total = Op.combine(total, smem[k])
        return total


@always_inline
def _scan_contig[
    Op: ReduceOp, dtype: DType, acc: DType, V: Int, vec_align: Int, lanes: Int
](
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    start: Int,
    n: Int,
    head: Int,
    n_vec: Int,
    vec_start: Int,
    tail_start: Int,
    lane: Int,
) -> Scalar[acc]:
    """One lane's share of a contiguous shard, over a `lanes`-wide group.

    16-byte vector body plus the scalar head and tail that the buffer's own
    alignment phase leaves over: a 128-bit access is legal only at an address
    that is a multiple of 16, so the partition is computed on the ADDRESS and
    not on the column index (a sliced operand can start at any phase).
    """
    var acc_vec = Op.identity[acc, V]()
    var v = lane
    while v < n_vec:
        acc_vec = Op.combine(
            acc_vec,
            Op.map[acc=acc](
                in_ptr.load[width=V, alignment=vec_align](vec_start + v * V)
            ),
        )
        v += lanes
    var total = _lane_fold[Op, acc, V](acc_vec)

    var jh = lane
    while jh < head:
        total = Op.combine(total, Op.map[acc=acc](in_ptr[start + jh]))
        jh += lanes
    var jt = tail_start + lane
    while jt < n:
        total = Op.combine(total, Op.map[acc=acc](in_ptr[start + jt]))
        jt += lanes
    return total


# ---------------------------------------------------------------------------
# Stage 1: contiguous reduce axis (inner == 1)
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(RED_THREADS))
)
@__name(t"reduce_contig_{Op.name}_{dtype}_l{lanes}")
def _reduce_contig_kernel[
    Op: ReduceOp, dtype: DType, lanes: Int
](
    out_ptr: UnsafePointer[Scalar[Op.out_dtype[dtype]()], MutAnyOrigin],
    ws_ptr: UnsafePointer[Scalar[Op.acc_dtype[dtype]()], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols_arg: Int64,
    outputs_arg: Int64,
    splits_arg: Int64,
):
    """A `lanes`-wide group walks its shard of one row with 16-byte vector
    loads and folds the accumulator; grid is (row group, split).

    `lanes` is the ONLY difference between the two contiguous-axis regimes and
    it is a comptime parameter, not a second kernel: `lanes == RED_THREADS` is
    a whole block per row (long rows, where the cross-warp fold is amortized
    over many loads), `lanes == WARP_SIZE` is a warp per row (short rows, where
    it is not -- and a warp needs no barrier and no shared memory at all, so
    the fold collapses to five shuffles). `_block_fold` covers both because a
    one-warp block IS the shuffle.

    `ws_ptr` is only touched when splits > 1; the launcher passes a null
    pointer for the fused single-split case, which writes the finished value
    straight to `out_ptr` and needs no second launch.
    """
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var cols = Int(cols_arg)
    var outputs = Int(outputs_arg)
    var splits = Int(splits_arg)
    comptime acc = Op.acc_dtype[dtype]()
    comptime V = 16 // size_of[dtype]()
    comptime vec_align = V * size_of[dtype]()  # 16 bytes
    comptime groups = RED_THREADS // lanes
    var tid = Int(thread_idx.x)
    var lane = tid % lanes  # lanes is a comptime power of two: a mask
    var row = Int(block_idx.x) * groups + tid // lanes
    comptime if groups > 1:
        # Safe: with more than one group per block there is no barrier below
        # (a one-warp fold is pure shuffle), so a partial block cannot
        # desynchronize anything.
        if row >= outputs:
            return
    var split = Int(block_idx.y)
    var base = row * cols

    # This split's shard of the row. The last shard may be empty (splits does
    # not divide cols); an empty shard contributes the identity. The fused case
    # skips the shard arithmetic entirely: `ceildiv` by a RUNTIME `splits` is an
    # integer division, and on a short row a handful of those is a real share of
    # the block's whole life (measured: 4096-column bf16 rows).
    var j0 = 0
    var n = cols
    if splits != 1:
        var chunk = ceildiv(cols, splits)
        j0 = split * chunk
        n = min(cols, j0 + chunk) - j0
    var start = base + j0

    var phase = _vec16_phase[dtype](Int(in_ptr))
    var head = n
    var n_vec = 0
    if phase >= 0:
        head = (V - (phase + start) % V) % V
        if head > n:
            head = n
        n_vec = (n - head) // V
    if n <= 0:
        head = 0
        n_vec = 0
    var total = _scan_contig[Op, dtype, acc, V, vec_align, lanes](
        in_ptr, start, n, head, n_vec, start + head, head + n_vec * V, lane
    )
    var group_total = _block_fold[Op, acc, lanes](lane, total)
    if lane == 0:
        if splits == 1:
            out_ptr[row] = Op.finish[out_dt=Op.out_dtype[dtype]()](
                group_total, cols
            )
        else:
            ws_ptr[split * outputs + row] = group_total


# ---------------------------------------------------------------------------
# Stage 1: strided reduce axis (inner > 1)
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(RED_THREADS))
)
@__name(t"reduce_strided_{Op.name}_{dtype}")
def _reduce_strided_kernel[
    Op: ReduceOp, dtype: DType
](
    out_ptr: UnsafePointer[Scalar[Op.out_dtype[dtype]()], MutAnyOrigin],
    ws_ptr: UnsafePointer[Scalar[Op.acc_dtype[dtype]()], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    reduce_arg: Int64,
    inner_arg: Int64,
    outputs_arg: Int64,
    splits_arg: Int64,
):
    """One thread per output column, walking down the reduced axis with stride
    `inner`.

    Consecutive threads read consecutive addresses, so every step of the walk
    is a fully coalesced load -- which is the whole point: reducing a leading
    dimension in place costs one pass over the input instead of a permuted
    copy plus a trailing-dim reduction. Block (x = outer * column-tile,
    y = split); no block fold is needed because a thread owns its output.
    """
    var reduce_n = Int(reduce_arg)
    var inner = Int(inner_arg)
    var outputs = Int(outputs_arg)
    var splits = Int(splits_arg)
    comptime acc = Op.acc_dtype[dtype]()

    var tiles = ceildiv(inner, RED_THREADS)
    var blk = Int(block_idx.x)
    var outer_index = blk // tiles
    var i = (blk % tiles) * RED_THREADS + Int(thread_idx.x)
    if i >= inner:
        return

    var chunk = ceildiv(reduce_n, splits)
    var split = Int(block_idx.y)
    var r0 = split * chunk
    var r1 = min(reduce_n, r0 + chunk)
    var base = outer_index * reduce_n * inner + i
    var out_index = outer_index * inner + i

    var total = Op.identity[acc, 1]()[0]
    for r in range(r0, r1):
        total = Op.combine(total, Op.map[acc=acc](in_ptr[base + r * inner]))

    if splits == 1:
        out_ptr[out_index] = Op.finish[out_dt=Op.out_dtype[dtype]()](
            total, reduce_n
        )
    else:
        ws_ptr[split * outputs + out_index] = total


# ---------------------------------------------------------------------------
# Stage 2: merge the per-(output, split) partials
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(RED_THREADS))
)
@__name(t"reduce_merge_thread_{Op.name}_{dtype}")
def _reduce_merge_thread_kernel[
    Op: ReduceOp, dtype: DType
](
    out_ptr: UnsafePointer[Scalar[Op.out_dtype[dtype]()], MutAnyOrigin],
    ws_ptr: UnsafePointer[Scalar[Op.acc_dtype[dtype]()], ImmutAnyOrigin],
    outputs_arg: Int64,
    splits_arg: Int64,
    reduce_arg: Int64,
):
    """Many outputs: one thread each. The workspace is split-major, so the
    threads of a warp read consecutive addresses at every step."""
    var outputs = Int(outputs_arg)
    var splits = Int(splits_arg)
    comptime acc = Op.acc_dtype[dtype]()
    var o = Int(block_idx.x) * RED_THREADS + Int(thread_idx.x)
    if o >= outputs:
        return
    var total = Op.identity[acc, 1]()[0]
    for k in range(splits):
        total = Op.combine(total, ws_ptr[k * outputs + o])
    out_ptr[o] = Op.finish[out_dt=Op.out_dtype[dtype]()](total, Int(reduce_arg))


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(RED_THREADS))
)
@__name(t"reduce_merge_block_{Op.name}_{dtype}")
def _reduce_merge_block_kernel[
    Op: ReduceOp, dtype: DType
](
    out_ptr: UnsafePointer[Scalar[Op.out_dtype[dtype]()], MutAnyOrigin],
    ws_ptr: UnsafePointer[Scalar[Op.acc_dtype[dtype]()], ImmutAnyOrigin],
    outputs_arg: Int64,
    splits_arg: Int64,
    reduce_arg: Int64,
):
    """Few outputs (the full-reduction end of the range): one block each,
    because a single thread walking hundreds of partials would serialize the
    tail of the launch."""
    var outputs = Int(outputs_arg)
    var splits = Int(splits_arg)
    comptime acc = Op.acc_dtype[dtype]()
    var o = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var total = Op.identity[acc, 1]()[0]
    for k in range(tid, splits, RED_THREADS):
        total = Op.combine(total, ws_ptr[k * outputs + o])
    var block_total = _block_fold[Op, acc, RED_THREADS](tid, total)
    if tid == 0:
        out_ptr[o] = Op.finish[out_dt=Op.out_dtype[dtype]()](
            block_total, Int(reduce_arg)
        )


# ---------------------------------------------------------------------------
# Host dispatch
# ---------------------------------------------------------------------------


@always_inline
def _red_splits(
    base_blocks: Int, reduce_n: Int, total: Int, min_per_split: Int, target: Int
) -> Int:
    """How many ways to split the reduced axis so the grid fills the device.

    `base_blocks` is the grid the layout produces with no splitting at all
    (one block per output for the contiguous kernel, one per column tile for
    the strided one -- NOT the output count). Splitting is pure overhead once
    that already fills the device, so it only kicks in below `target`; it never
    shards the reduced axis finer than `min_per_split`; and it declines
    entirely below `RED_MIN_SPLIT_ELEMS` elements of `total` work, which is
    what makes the small-shape end degrade cleanly to a single launch.
    """
    if (
        base_blocks >= target
        or total < RED_MIN_SPLIT_ELEMS
        or reduce_n < 2 * min_per_split
    ):
        return 1
    var by_fill = ceildiv(target, base_blocks)
    var by_work = reduce_n // min_per_split
    return max(min(by_fill, by_work), 1)


@always_inline
def _reduce_generic[
    Op: ReduceOp, dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    outer: Int,
    reduce_n: Int,
    inner: Int,
    ctx: DeviceContext,
) raises:
    """Reduce `in` viewed as (outer, reduce_n, inner) into `outer * inner`
    contiguous outputs, with `Op`'s algebra."""
    comptime acc = Op.acc_dtype[dtype]()
    comptime out_dt = Op.out_dtype[dtype]()
    var out_ptr = _make_ptr[out_dt](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var outputs = outer * inner

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var o = Int(idx[0].value())
            var base = (o // inner) * reduce_n * inner + (o % inner)
            var total = Op.identity[acc, 1]()[0]
            for r in range(reduce_n):
                total = Op.combine(
                    total, Op.map[acc=acc](in_ptr[base + r * inner])
                )
            out_ptr[o] = Op.finish[out_dt=out_dt](total, reduce_n)

        _parallel_for[func](outputs, ctx)
        return

    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        var target = RED_BLOCKS_PER_SM * _device_sm_count(ctx)
        var mout = out_ptr.as_unsafe_any_origin()
        var min_ = in_ptr.as_unsafe_any_origin().as_immutable()

        var base_blocks = outputs
        # The contiguous kernel wants a whole 16-byte vector per thread before
        # a shard is worth its own block; the strided kernel wants a few rows.
        var min_per_split = RED_THREADS * (16 // size_of[dtype]())
        if inner != 1:
            base_blocks = outer * ceildiv(inner, RED_THREADS)
            min_per_split = RED_MIN_ROWS_PER_SPLIT
        var splits = _red_splits(
            base_blocks, reduce_n, outputs * reduce_n, min_per_split, target
        )

        if splits == 1:
            # Fused: stage 1 writes the finished value; `ws` is never touched.
            var no_ws = _make_ptr[acc](0).as_unsafe_any_origin()
            if inner == 1:
                comptime V = 16 // size_of[dtype]()
                comptime GROUPS = RED_THREADS // WARP_SIZE
                # A warp per row when a block per row would leave each thread
                # with too few vectors to amortize the cross-warp fold -- but
                # only while the narrower group still fills the device, since it
                # divides the grid by GROUPS.
                var warp_blocks = ceildiv(outputs, GROUPS)
                if (
                    reduce_n // V < RED_THREADS * RED_MIN_VECS_PER_THREAD
                    and warp_blocks >= target
                ):
                    _enqueue_cached[
                        _reduce_contig_kernel[Op, dtype, WARP_SIZE]
                    ](
                        ctx,
                        String(t"reduce_contig_{Op.name}_{dtype}_w"),
                        warp_blocks,
                        1,
                        1,
                        RED_THREADS,
                        mout,
                        no_ws,
                        min_,
                        Int64(reduce_n),
                        Int64(outputs),
                        Int64(1),
                    )
                    return
                _enqueue_cached[_reduce_contig_kernel[Op, dtype, RED_THREADS]](
                    ctx,
                    String(t"reduce_contig_{Op.name}_{dtype}_b"),
                    base_blocks,
                    1,
                    1,
                    RED_THREADS,
                    mout,
                    no_ws,
                    min_,
                    Int64(reduce_n),
                    Int64(outputs),
                    Int64(1),
                )
            else:
                _enqueue_cached[_reduce_strided_kernel[Op, dtype]](
                    ctx,
                    String(t"reduce_strided_{Op.name}_{dtype}"),
                    base_blocks,
                    1,
                    1,
                    RED_THREADS,
                    mout,
                    no_ws,
                    min_,
                    Int64(reduce_n),
                    Int64(inner),
                    Int64(outputs),
                    Int64(1),
                )
            return

        # Split: one accumulator per (output, split) into a stream-ordered
        # workspace, merged by stage 2.
        var ws = ctx.enqueue_create_buffer[acc](outputs * splits)
        var ws_ptr = ws.unsafe_ptr().as_unsafe_any_origin()
        if inner == 1:
            # The split path always gives a whole block to one (row, shard):
            # splitting only happens when the rows alone cannot fill the device,
            # which is the opposite of the warp-per-row regime's precondition.
            _enqueue_cached[_reduce_contig_kernel[Op, dtype, RED_THREADS]](
                ctx,
                String(t"reduce_contig_{Op.name}_{dtype}_b"),
                base_blocks,
                splits,
                1,
                RED_THREADS,
                mout,
                ws_ptr,
                min_,
                Int64(reduce_n),
                Int64(outputs),
                Int64(splits),
            )
        else:
            _enqueue_cached[_reduce_strided_kernel[Op, dtype]](
                ctx,
                String(t"reduce_strided_{Op.name}_{dtype}"),
                base_blocks,
                splits,
                1,
                RED_THREADS,
                mout,
                ws_ptr,
                min_,
                Int64(reduce_n),
                Int64(inner),
                Int64(outputs),
                Int64(splits),
            )

        # Few outputs cannot keep the device busy one thread each, so they get
        # a block apiece; many outputs would waste 255 of every 256 threads
        # that way and get a thread apiece instead.
        if outputs <= _device_sm_count(ctx):
            _enqueue_cached[_reduce_merge_block_kernel[Op, dtype]](
                ctx,
                String(t"reduce_merge_block_{Op.name}_{dtype}"),
                outputs,
                1,
                1,
                RED_THREADS,
                mout,
                ws_ptr.as_immutable(),
                Int64(outputs),
                Int64(splits),
                Int64(reduce_n),
            )
        else:
            _enqueue_cached[_reduce_merge_thread_kernel[Op, dtype]](
                ctx,
                String(t"reduce_merge_thread_{Op.name}_{dtype}"),
                ceildiv(outputs, RED_THREADS),
                1,
                1,
                RED_THREADS,
                mout,
                ws_ptr.as_immutable(),
                Int64(outputs),
                Int64(splits),
                Int64(reduce_n),
            )
        # Dropping `ws` schedules a stream-ordered free after the kernels.
        _ = ws^


# ---------------------------------------------------------------------------
# The bridge, also written once. `_rowred_spec_into_go[SumOp]` and
# `_rowred_spec_into_go[MaxOp]` are the SAME code; the registration sites in
# reduction_ops.mojo and nn_ops.mojo pick the accumulator.
# ---------------------------------------------------------------------------


def _rowred_spec_into_go[
    Op: ReduceOp
](
    a_o: PyObjectPtr,
    rdims_t: PyObjectPtr,
    keepdim_o: PyObjectPtr,
    out_o: PyObjectPtr,
) raises:
    """TensorSpec entry for one scalar reduction (docs/tensor_spec_design.md).

    Python parses the dim spec and owns dtype promotion and output allocation;
    this side derives the (outer, reduce, inner) geometry, validates the
    preallocated output and launches. A reduce interval that is adjacent and
    ascending over a contiguous operand is read WHERE IT LIES, whatever its
    position -- the strided-axis kernel is what lets a `dim=0` reduction skip
    the transposed full-tensor copy Python would otherwise materialize.
    Anything else raises a real NotImplementedError into Python ("take the
    classic permute+materialize path").
    """
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    if not _dtype_supported[Op.dtypes](a.dtype):
        raise Error("mojo spec reduce: unsupported dtype ", a.dtype)

    var outer = 0
    var reduce_n = 0
    var inner = 0
    if not _adjacent_reduce_geom(a, rdims_t, outer, reduce_n, inner):
        # Non-adjacent or strided: Python permuted and materialized before the
        # call, so this only has to reproduce the rejection message.
        var rows = 0
        var cols = 0
        var out_rank = 0
        var oshape = IndexList[MAX_RANK](1)
        _reduce_spec_geom(a, rdims_t, keepdim_o, rows, cols, out_rank, oshape)
        outer = rows
        reduce_n = cols
        inner = 1
    var outputs = outer * inner
    comptime if Op.errors_on_empty_axis:
        # torch refuses this one; raising hands it back to Python, which
        # declines and lets torch raise its own message.
        if reduce_n == 0 and outputs > 0:
            raise Error("mojo spec reduce: empty reduce dim")
    var ctx = a.ctx()
    comptime for dt in Op.dtypes:
        comptime if _dtype_arg_on[0, dt]():
            if a.dtype == dt:
                _check_into_sized(a, out, outputs, Op.out_dtype[dt]())
                if outputs > 0:
                    _reduce_generic[Op, dt](
                        out.ptr, a.ptr, outer, reduce_n, inner, ctx
                    )
