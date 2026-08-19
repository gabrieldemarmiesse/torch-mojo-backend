# ===----------------------------------------------------------------------=== #
# Fast eager-mode data-movement kernels for mojo_device: strided permute
# copies (transpose/permute materialization), narrow copies (split/slice
# along one dim), dtype casts, and cond ? a : b selection.
#
# Raw-pointer convention (see docs/strided_owning_tensors_design.md): every
# Python-visible function takes raw element-aligned data addresses (ints,
# storage offset already applied) plus the device's DeviceContext pointer —
# there is no `max.driver.Buffer` on this side any more. Pure-copy kernels
# (PermuteCopy/NarrowCopyDst) are handed an explicit `itemsize`
# int (1/2/4/8) computed on the Python side instead of reading a dtype off a
# buffer; Cast/WhereSelect take the operand dtype(s) as plain ints
# (`max.dtype.DType.value`) and rebuild the Mojo `DType` via
# `_raw_dtype_int`.
# ===----------------------------------------------------------------------=== #

from std.atomic import Atomic, Ordering
from std.os import abort
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.collections import InlineArray
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv
from max.gpu.host import DeviceContext
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import is_amd_gpu, is_nvidia_gpu
from std.sys.info import has_accelerator, has_apple_gpu_accelerator, size_of
from std.utils.coord import Coord

from max.algorithm import elementwise
from std.utils import IndexList

from std.python._cpython import PyObjectPtr, Py_ssize_t

from op_utils import (
    GS_THREADS,
    MAX_RANK,
    _BW_MAX_BLOCKS,
    _LLC_BYTES,
    _bw_blocks,
    _bw_flat_blocks,
    _fill_bits,
    _fill_bits_dtype,
    _MAX_GRID_Y,
    _T2D_ROWS,
    _enqueue_cached,
    _enqueue_cached_2d,
    _gs_blocks,
    _t2d_tile,
    _transpose2d_kernel,
    _make_ptr,
    _parallel_for,
    _parallel_for_dt,
    _scratch_contig,
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _raw_ret_none,
    _raw_tuple_int,
    _raw_tuple_len,
    _device_sm_count,
    _spec_dispatcher3,
    _spec_ptr,
    _spec_unsupported,
)

from variant_gates import (
    _dtype_arg_on,
    _dtype_arg_width_on,
    _dtype_out_on,
    _op_on,
    _register_call,
)


# Strided kernels that work on rank-<=8 tensors pad shapes/strides to this
# rank on the Python side (leading dims of size 1 / stride 0); MAX_RANK is
# the shared op_utils constant.

# Dtypes ScatterDim dispatches on: it needs the real dtype (a scalar value is
# cast to it) so element-size dispatch is not enough.
comptime SCATTER_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.bool,
]


# ---------------------------------------------------------------------------
# Permute copy: materialize an arbitrary permutation of a contiguous tensor
# of rank <= 4. The Python side pads the *output* shape to 4 dims with
# leading 1s and passes, for each output dim, the corresponding stride in
# the *source* buffer (in elements). Kernels are specialized on element
# byte-size, not dtype, since this is a pure copy.
# ---------------------------------------------------------------------------


def _permute_copy_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    d1_arg: Int64,
    d2_arg: Int64,
    d3_arg: Int64,
    s0_arg: Int64,
    s1_arg: Int64,
    s2_arg: Int64,
    s3_arg: Int64,
    total_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var d1 = Int(d1_arg)
    var d2 = Int(d2_arg)
    var d3 = Int(d3_arg)
    var s0 = Int(s0_arg)
    var s1 = Int(s1_arg)
    var s2 = Int(s2_arg)
    var s3 = Int(s3_arg)
    var total = Int(total_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while i < total:
        var i3 = i % d3
        var rest = i // d3
        var i2 = rest % d2
        rest = rest // d2
        var i1 = rest % d1
        var i0 = rest // d1
        out_ptr[i] = in_ptr[i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3]
        i += gstride


def _permute_copy_rows4_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    d1_arg: Int64,
    d2_arg: Int64,
    d3_4_arg: Int64,
    s0_arg: Int64,
    s1_arg: Int64,
    s2_arg: Int64,
    nchunks_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var d1 = Int(d1_arg)
    var d2 = Int(d2_arg)
    var d3_4 = Int(d3_4_arg)
    var s0 = Int(s0_arg)
    var s1 = Int(s1_arg)
    var s2 = Int(s2_arg)
    var nchunks = Int(nchunks_arg)
    # Innermost dim contiguous in BOTH buffers (s3 == 1): each thread moves
    # one 4-element vector, so the coordinate div/mod chain runs once per 4
    # elements and every access is a vector load/store.  This is the
    # post-SDPA `.contiguous()` shape: a batched gather of contiguous rows.
    comptime vec_align = 4 * size_of[dtype]()
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while c < nchunks:
        var j = c % d3_4
        var rest = c // d3_4
        var i2 = rest % d2
        rest = rest // d2
        var i1 = rest % d1
        var i0 = rest // d1
        out_ptr.store[width=4, alignment=vec_align](
            c * 4,
            in_ptr.load[width=4, alignment=vec_align](
                i0 * s0 + i1 * s1 + i2 * s2 + j * 4
            ),
        )
        c += gstride


def _permute_copy_rowloop_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    d1_arg: Int64,
    d2_arg: Int64,
    d3_4_arg: Int64,
    s0_arg: Int64,
    s1_arg: Int64,
    s2_arg: Int64,
    nrows_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var d1 = Int(d1_arg)
    var d2 = Int(d2_arg)
    var d3_4 = Int(d3_4_arg)
    var s0 = Int(s0_arg)
    var s1 = Int(s1_arg)
    var s2 = Int(s2_arg)
    var nrows = Int(nrows_arg)
    # One thread per (i0, i1, i2) row: the div/mod chain runs once per d3
    # elements and the sequential vector moves give each thread load-level
    # parallelism.  Measured ~2.5x the chunk kernel's throughput on Apple at
    # the post-SDPA shape; only used when there are enough rows to fill the
    # machine.
    comptime vec_align = 4 * size_of[dtype]()
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while r < nrows:
        var i2 = r % d2
        var rest = r // d2
        var i1 = rest % d1
        var i0 = rest // d1
        var src = i0 * s0 + i1 * s1 + i2 * s2
        var dst = r * d3_4 * 4
        for j in range(d3_4):
            out_ptr.store[width=4, alignment=vec_align](
                dst + j * 4,
                in_ptr.load[width=4, alignment=vec_align](src + j * 4),
            )
        r += gstride


@always_inline

# A permutation whose innermost extent is contiguous in *both* operands
# (`s3 == 1`) is not a transpose at all: it is a gather of `d3`-element runs.
# nanoGPT's `view(B, T, nh, hs).transpose(1, 2)` is exactly that, and it is the
# most common permutation an attention block issues. The generic kernel above
# copies one element per thread, so a BF16 lane uses 2 bytes of a 16-byte access
# and pays three integer divisions for them; here consecutive lanes take
# consecutive *vectors* of the contiguous destination, so the stores are wide and
# contiguous, the loads are whole runs, and the three divisions are amortized
# over a whole vector.
def _run_gather_kernel[
    dtype: DType, VEC: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    d1_arg: Int64,
    d2_arg: Int64,
    spv_arg: Int64,
    s0_arg: Int64,
    s1_arg: Int64,
    s2_arg: Int64,
    slots_arg: Int64,
):
    """`out[i0,i1,i2,i3] = in[i0*s0 + i1*s1 + i2*s2 + i3]`, VEC elements a time.

    `spv` is the number of VEC-wide vectors in one run (`d3 // VEC`) and `slots`
    the number of vectors in the whole copy; the caller has checked that every
    stride and the run length are multiples of VEC and that both base addresses
    are vector-aligned.
    """
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var d1 = Int(d1_arg)
    var d2 = Int(d2_arg)
    var spv = Int(spv_arg)
    var s0 = Int(s0_arg)
    var s1 = Int(s1_arg)
    var s2 = Int(s2_arg)
    var slots = Int(slots_arg)
    comptime ALIGN = min(16, VEC * size_of[dtype]())
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while j < slots:
        var off = j % spv
        var run = j // spv
        var i2 = run % d2
        var rest = run // d2
        var i1 = rest % d1
        var i0 = rest // d1
        out_ptr.store[width=VEC, alignment=ALIGN](
            j * VEC,
            in_ptr.load[width=VEC, alignment=ALIGN](
                i0 * s0 + i1 * s1 + i2 * s2 + off * VEC
            ),
        )
        j += gstride


@always_inline
def _permute_copy[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    d0: Int,
    d1: Int,
    d2: Int,
    d3: Int,
    s0: Int,
    s1: Int,
    s2: Int,
    s3: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var total = d0 * d1 * d2 * d3

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value())
            var i3 = i % d3
            var rest = i // d3
            var i2 = rest % d2
            rest = rest // d2
            var i1 = rest % d1
            var i0 = rest // d1
            out_ptr[i] = in_ptr[i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3]

        elementwise[func, simd_width=1](Coord(total), ctx)
    else:
        comptime if has_accelerator():
            # The innermost extent contiguous in both operands: gather runs, in
            # the widest vector the runtime extents, strides and addresses admit.
            # Every candidate width is a compile-time regime; which one runs is a
            # runtime decision, and when none fits the copy declines to the
            # general element-at-a-time kernel below, which masks every extent.
            @always_inline
            @parameter
            def _try_run_gather[VEC: Int]() raises -> Bool:
                comptime ALIGN = min(16, VEC * size_of[dtype]())
                if (
                    d3 % VEC != 0
                    or s0 % VEC != 0
                    or s1 % VEC != 0
                    or s2 % VEC != 0
                    or out_addr % ALIGN != 0
                    or in_addr % ALIGN != 0
                ):
                    return False
                var slots = total // VEC
                _enqueue_cached[_run_gather_kernel[dtype, VEC]](
                    ctx,
                    String(t"dm_rungather_{dtype}_v{VEC}"),
                    _gs_blocks(slots),
                    1,
                    1,
                    GS_THREADS,
                    out_ptr.as_unsafe_any_origin(),
                    in_ptr.as_unsafe_any_origin().as_immutable(),
                    Int64(d1),
                    Int64(d2),
                    Int64(d3 // VEC),
                    Int64(s0),
                    Int64(s1),
                    Int64(s2),
                    Int64(slots),
                )
                return True

            # Apple: the measured rows4/rowloop kernels beat the generic
            # gather on the transpose-materialize hot case (0.39 vs 0.52 ms
            # at the post-SDPA clone shape, pinned clocks); keep them first.
            comptime if has_apple_gpu_accelerator():
                comptime apple_vec_align = 4 * size_of[dtype]()
                if (
                    total > 0
                    and s3 == 1
                    and d3 % 4 == 0
                    and (s0 | s1 | s2) % 4 == 0
                    and (out_addr | in_addr) % apple_vec_align == 0
                ):
                    var nrows = d0 * d1 * d2
                    if nrows >= 4096:
                        _enqueue_cached[_permute_copy_rowloop_kernel[dtype]](
                            ctx,
                            String(t"dm_permute_rowloop_{dtype}"),
                            _gs_blocks(nrows),
                            1,
                            1,
                            GS_THREADS,
                            out_ptr.as_unsafe_any_origin(),
                            in_ptr.as_unsafe_any_origin().as_immutable(),
                            Int64(d1),
                            Int64(d2),
                            Int64(d3 // 4),
                            Int64(s0),
                            Int64(s1),
                            Int64(s2),
                            Int64(nrows),
                        )
                        return
                    var nchunks = total // 4
                    _enqueue_cached[_permute_copy_rows4_kernel[dtype]](
                        ctx,
                        String(t"dm_permute_rows4_{dtype}"),
                        _gs_blocks(nchunks),
                        1,
                        1,
                        GS_THREADS,
                        out_ptr.as_unsafe_any_origin(),
                        in_ptr.as_unsafe_any_origin().as_immutable(),
                        Int64(d1),
                        Int64(d2),
                        Int64(d3 // 4),
                        Int64(s0),
                        Int64(s1),
                        Int64(s2),
                        Int64(nchunks),
                    )
                    return

            comptime V32 = 32 // size_of[dtype]()
            comptime V16 = 16 // size_of[dtype]()
            comptime V8 = 8 // size_of[dtype]()
            if s3 == 1 and d3 > 1 and total >= 1024:
                # 32-byte gather accesses intermittently wedge the Metal
                # driver queue (verified on M4; see tensor_holder.mojo), so
                # Apple starts at the 16-byte regime.
                comptime if not has_apple_gpu_accelerator():
                    if _try_run_gather[V32]():
                        return
                comptime if V16 < V32:
                    if _try_run_gather[V16]():
                        return
                comptime if V8 < V16:
                    if _try_run_gather[V8]():
                        return

            # A batched transpose of the innermost two dims is by far the most
            # common permutation -- the eager SDPA backward does four per layer --
            # and the generic kernel below reads one element per thread down a
            # column, so it never reaches a useful fraction of bandwidth. When the
            # permutation is exactly that, hand it to the tiled LDS transpose,
            # which stages a TILE x TILE block through shared memory so both the
            # reads and the writes are contiguous.
            #
            # `s2 == 1` says the output's row index walks the source contiguously,
            # i.e. source columns are output rows; `s3 >= d2` says the output's
            # column index steps by the source's row pitch. The leading pair
            # collapses into one batch axis only if its two strides are uniform,
            # hence `s0 == d1 * s1`.
            comptime TILE = _t2d_tile[dtype]()
            var batch = d0 * d1
            if (
                d2 > 1
                and d3 > 1
                and total >= 1024
                and s2 == 1
                and s3 >= d2
                and (d0 == 1 or s0 == d1 * s1)
                and (batch == 1 or s1 >= d3 * s3)
            ):
                _enqueue_cached[_transpose2d_kernel[dtype]](
                    ctx,
                    String(t"transpose2d_{dtype}"),
                    (d3 + TILE - 1) // TILE,
                    min((d2 + TILE - 1) // TILE, _MAX_GRID_Y),
                    min(batch, _MAX_GRID_Y),
                    TILE * _T2D_ROWS,
                    out_ptr.as_unsafe_any_origin(),
                    in_ptr.as_unsafe_any_origin().as_immutable(),
                    Int64(d2),
                    Int64(d3),
                    Int64(s3),
                    Int64(batch),
                    Int64(d2 * d3),
                    Int64(s1 if batch > 1 else 0),
                )
                return
            _enqueue_cached[_permute_copy_kernel[dtype]](
                ctx,
                String(t"dm_permute_{dtype}"),
                _gs_blocks(total),
                1,
                1,
                GS_THREADS,
                out_ptr.as_unsafe_any_origin(),
                in_ptr.as_unsafe_any_origin().as_immutable(),
                Int64(d1),
                Int64(d2),
                Int64(d3),
                Int64(s0),
                Int64(s1),
                Int64(s2),
                Int64(s3),
                Int64(total),
            )
        else:
            raise Error("no GPU accelerator available at compile time")


def _permute_copy_go(
    out_ptr: PyObjectPtr,
    in_ptr: PyObjectPtr,
    dims: PyObjectPtr,
    strides: PyObjectPtr,
    itemsize_o: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr)
    var in_addr = _raw_int(in_ptr)
    var d0 = _raw_tuple_int(dims, 0)
    var d1 = _raw_tuple_int(dims, 1)
    var d2 = _raw_tuple_int(dims, 2)
    var d3 = _raw_tuple_int(dims, 3)
    var s0 = _raw_tuple_int(strides, 0)
    var s1 = _raw_tuple_int(strides, 1)
    var s2 = _raw_tuple_int(strides, 2)
    var s3 = _raw_tuple_int(strides, 3)
    var itemsize = _raw_int(itemsize_o)
    var ctx = _raw_ctx(ctx_ptr)

    comptime if _dtype_arg_width_on[0, 32]():
        if itemsize == 4:
            _permute_copy[DType.uint32](
                out_addr, in_addr, d0, d1, d2, d3, s0, s1, s2, s3, ctx
            )
        else:
            raise Error("permute specialization/itemsize mismatch")
    elif _dtype_arg_width_on[0, 16]():
        if itemsize == 2:
            _permute_copy[DType.uint16](
                out_addr, in_addr, d0, d1, d2, d3, s0, s1, s2, s3, ctx
            )
        else:
            raise Error("permute specialization/itemsize mismatch")
    elif _dtype_arg_width_on[0, 64]():
        if itemsize == 8:
            _permute_copy[DType.uint64](
                out_addr, in_addr, d0, d1, d2, d3, s0, s1, s2, s3, ctx
            )
        else:
            raise Error("permute specialization/itemsize mismatch")
    elif _dtype_arg_width_on[0, 8]():
        if itemsize == 1:
            _permute_copy[DType.uint8](
                out_addr, in_addr, d0, d1, d2, d3, s0, s1, s2, s3, ctx
            )
        else:
            raise Error("permute specialization/itemsize mismatch")
    else:
        raise Error("unsupported element size for fast permute")


# ---------------------------------------------------------------------------
# Batched N-input concatenation: ONE launch fills the output from every
# contiguous input, whatever the input count (two, three, or the sixty-four
# of a split-backward reassembly).
#
# After the caller flattens the cat dim, the output is `outer` rows of
# `dst_stride` elements and input k contributes `row_len_k` contiguous
# elements at `dst_off_k` inside every row, read from
# `src_k[o * row_len_k ...]`. Work is cut into fixed-size tiles, so wildly
# unequal input sizes stay balanced and no block is ever launched over an
# input that has no work for it: block (x, y) takes tile x of row y, then
# grid-strides over both. The input owning a tile is found by binary search
# over the per-input tile prefix sums -- once per tile per block,
# warp-uniformly, against a parameter array.
#
# `width` is a runtime-dispatched comptime parameter, not a hardcoded 4: the
# caller passes the element count that makes one access 16 bytes wide (8 for
# bf16, 4 for f32, 2 for f64) when every base address, row length and row
# stride is 16-byte aligned, and 1 otherwise -- the same kernel then runs
# element-wise, so no shape is excluded from the batched path.
# ---------------------------------------------------------------------------

# Inputs covered by one launch: 64 descriptors x 32 bytes = 2 KiB of kernel
# parameters, inside every backend's parameter budget (CUDA's is 4 KiB).
# Longer tensor lists are launched in batches of this many; nothing about the
# kernel changes.
comptime CAT_SEG_CAP = 64

# Metal translates only pointer-TYPED kernel arguments into GPU addresses; a
# raw address smuggled as data reads back zeros (see the header of
# foreach_elementwise_kernels.mojo, where the same constraint is documented
# and verified). Apple therefore takes a variant with real pointer arguments,
# and correspondingly fewer inputs per launch. `_cat_slots_kernel` and
# `_cat_pick` write those arguments out one by one, so this count and their
# parameter lists move together.
comptime CAT_PTR_SLOTS = 8

# Widest access every backend supports, and the contract with the caller:
# `aten_fast._cat_vector_width` passes the element count that fills this many
# bytes only when every address, row length and row stride in the call is
# aligned to it, and 1 otherwise.
comptime CAT_VECTOR_BYTES = 16

comptime CAT_CAP = CAT_PTR_SLOTS if has_apple_gpu_accelerator() else CAT_SEG_CAP

# Bytes one thread copies per tile. A tile is the granule the owning input
# is looked up for, so this is what amortizes that lookup, and it is
# expressed in BYTES rather than slots so the element path gets the same
# amortization as the 16-byte one instead of a slice of work 8x thinner.
#
# Fitted on an H100 PCIe at the benchmark suite's pinned 1395MHz;
# ours/stock device time, one launch in every cell:
#
#   bytes per thread            32      64     128     256
#   64 x 262144      bf16    1.034   1.024   1.024   1.033
#   2 x 8388608      bf16    1.012   0.989   1.003   0.990
#   64 x [357,789]   bf16    1.044   1.059   1.093   1.280
#   37 x 12345       bf16    1.430   1.355   1.856   2.660
#   8 x [2048,1024]  f32     1.005   1.003   1.143   1.551
#
# The 16-byte path is DRAM bound and flat (~1.75 TB/s of this card's 2.04);
# what moves is the two ends. Too little work per thread leaves the lookup
# exposed -- at ONE slot a thread (16 bytes wide, 2 element-wise) the same
# rows read 1.146 and 4.26. Too much wastes the tail: rows shorter than a
# tile idle the surplus threads, which is the [2048,1024] row (its 1024
# elements are a quarter of a 128-byte tile). 32 is the balance point on
# this card; the two ends are architectural, so the balance may not be.
comptime CAT_BYTES_PER_THREAD = 32


@always_inline
def _cat_tile_slots[dtype: DType, width: Int]() -> Int:
    """Vector slots one block covers per tile: CAT_BYTES_PER_THREAD each."""
    return GS_THREADS * max(
        1, CAT_BYTES_PER_THREAD // (width * size_of[dtype]())
    )


# Bound on the blocks of one launch. A concatenation is pure streaming
# traffic, so the grid covers the tiles exactly (op_utils' "cover the slots
# exactly once the copy streams from HBM"); this cap only keeps a
# pathological tile count from overflowing the grid.
comptime CAT_MAX_BLOCKS = 1 << 22


struct CatSeg(DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable):
    """One input's contribution to every output row.

    `tile_end` is the exclusive prefix sum of per-row tiles over the inputs
    of this launch, which is what makes the owner of a tile a binary search.
    An input with no elements keeps the previous `tile_end` and is therefore
    never selected.
    """

    comptime device_type: AnyType = Self

    var src_addr: Int  # element-aligned base address (unused on Metal)
    var nvec: Int  # vector slots this input contributes per output row
    var dst_off: Int  # element offset of this input inside an output row
    var tile_end: Int  # exclusive prefix sum of tiles over the launch's inputs

    def __init__(
        out self, src_addr: Int, nvec: Int, dst_off: Int, tile_end: Int
    ):
        self.src_addr = src_addr
        self.nvec = nvec
        self.dst_off = dst_off
        self.tile_end = tile_end

    def _to_device_type(
        self,
        mut encoder: Some[DeviceTypeEncoder],
        target: MutOpaquePointer[_],
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "CatSeg"


@always_inline
def _cat_owner(
    segs: InlineArray[CatSeg, CAT_CAP], nseg: Int, tile: Int
) -> Tuple[Int, Int]:
    """(input owning `tile`, first tile of that input): the smallest index
    whose exclusive tile prefix sum is past `tile`."""
    var lo = 0
    var hi = nseg - 1
    while lo < hi:
        var mid = (lo + hi) // 2
        if segs[mid].tile_end <= tile:
            lo = mid + 1
        else:
            hi = mid
    var first = 0
    if lo != 0:
        first = segs[lo - 1].tile_end
    return lo, first


@always_inline
def _cat_copy_rows[
    dtype: DType, width: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    src_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    seg: CatSeg,
    slot: Int,
    outer: Int,
    dst_stride: Int,
):
    """Copy this thread's slots of `seg` out of every output row.

    The slots are `GS_THREADS` apart so each of the unrolled accesses stays
    coalesced across the block, and the loads of one row are independent of
    each other, which is what gives a thread its memory-level parallelism.
    """
    comptime align = width * size_of[dtype]()
    comptime ilp = _cat_tile_slots[dtype, width]() // GS_THREADS
    var row_len = seg.nvec * width
    var src_index = slot * width
    var dst_index = seg.dst_off + slot * width
    var row = Int(block_idx.y)
    src_index += row * row_len
    dst_index += row * dst_stride
    while row < outer:

        @parameter
        for step in range(ilp):
            if slot + step * GS_THREADS < seg.nvec:
                out_ptr.store[width=width, alignment=align](
                    dst_index + step * GS_THREADS * width,
                    src_ptr.load[width=width, alignment=align](
                        src_index + step * GS_THREADS * width
                    ),
                )
        row += Int(grid_dim.y)
        src_index += Int(grid_dim.y) * row_len
        dst_index += Int(grid_dim.y) * dst_stride


def _cat_batched_kernel[
    dtype: DType, width: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    segs: InlineArray[CatSeg, CAT_CAP],
    nseg_arg: Int64,
    tiles_arg: Int64,
    outer_arg: Int64,
    dst_stride_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var nseg = Int(nseg_arg)
    var tiles = Int(tiles_arg)
    var outer = Int(outer_arg)
    var dst_stride = Int(dst_stride_arg)
    var lane = Int(thread_idx.x)
    var tile = Int(block_idx.x)
    while tile < tiles:
        var owner, first = _cat_owner(segs, nseg, tile)
        var seg = segs[owner]
        var slot = (tile - first) * _cat_tile_slots[dtype, width]() + lane
        if slot < seg.nvec:
            _cat_copy_rows[dtype, width](
                out_ptr,
                _make_ptr[dtype](seg.src_addr)
                .as_unsafe_any_origin()
                .as_immutable(),
                seg,
                slot,
                outer,
                dst_stride,
            )
        tile += Int(grid_dim.x)


@always_inline
def _cat_pick[
    dtype: DType
](
    slot: Int,
    p0: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    p1: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    p2: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    p3: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    p4: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    p5: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    p6: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    p7: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
) -> UnsafePointer[Scalar[dtype], ImmutAnyOrigin]:
    """Select one of the pointer ARGUMENTS (copying them into an array and
    indexing that miscompiles on Metal -- see foreach_elementwise_kernels)."""
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


def _cat_slots_kernel[
    dtype: DType, width: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    p0: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    p1: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    p2: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    p3: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    p4: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    p5: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    p6: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    p7: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    segs: InlineArray[CatSeg, CAT_CAP],
    nseg_arg: Int64,
    tiles_arg: Int64,
    outer_arg: Int64,
    dst_stride_arg: Int64,
):
    """`_cat_batched_kernel` with the sources as pointer arguments (Apple)."""
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var nseg = Int(nseg_arg)
    var tiles = Int(tiles_arg)
    var outer = Int(outer_arg)
    var dst_stride = Int(dst_stride_arg)
    var lane = Int(thread_idx.x)
    var tile = Int(block_idx.x)
    while tile < tiles:
        var owner, first = _cat_owner(segs, nseg, tile)
        var seg = segs[owner]
        var slot = (tile - first) * _cat_tile_slots[dtype, width]() + lane
        if slot < seg.nvec:
            _cat_copy_rows[dtype, width](
                out_ptr,
                _cat_pick[dtype](owner, p0, p1, p2, p3, p4, p5, p6, p7),
                seg,
                slot,
                outer,
                dst_stride,
            )
        tile += Int(grid_dim.x)


@always_inline
def _cat_slot_ptr[
    dtype: DType
](segs: InlineArray[CatSeg, CAT_CAP], nseg: Int, index: Int) -> UnsafePointer[
    Scalar[dtype], ImmutAnyOrigin
]:
    """A translatable pointer for every pointer argument: padding slots
    repeat the first real source and are never dereferenced (their `nvec`
    is zero and no tile selects them)."""
    var addr = segs[0].src_addr
    if index < nseg:
        addr = segs[index].src_addr
    return _make_ptr[dtype](addr).as_unsafe_any_origin().as_immutable()


@always_inline
def _cat_launch_width[
    dtype: DType, width: Int
](
    out_addr: Int,
    srcs: PyObjectPtr,
    lens: PyObjectPtr,
    n: Int,
    outer: Int,
    dst_stride: Int,
    ctx: DeviceContext,
) raises:
    comptime tile_slots = _cat_tile_slots[dtype, width]()
    var out_ptr = _make_ptr[dtype](out_addr).as_unsafe_any_origin()
    var dst_off = 0
    var index = 0
    while index < n:
        var segs = InlineArray[CatSeg, CAT_CAP](fill=CatSeg(0, 0, 0, 0))
        var tiles = 0
        var nseg = 0
        while index < n and nseg < CAT_CAP:
            var row_len = _raw_tuple_int(lens, index)
            var nvec = row_len // width
            tiles += (nvec + tile_slots - 1) // tile_slots
            segs[nseg] = CatSeg(
                _raw_tuple_int(srcs, index), nvec, dst_off, tiles
            )
            dst_off += row_len
            nseg += 1
            index += 1
        if tiles == 0:
            continue
        var gy = min(outer, _MAX_GRID_Y)
        var gx = min(tiles, max(1, CAT_MAX_BLOCKS // gy))
        comptime if has_apple_gpu_accelerator():
            _enqueue_cached[_cat_slots_kernel[dtype, width]](
                ctx,
                String(t"dm_cat_slots_{dtype}_{width}"),
                gx,
                gy,
                1,
                GS_THREADS,
                out_ptr,
                _cat_slot_ptr[dtype](segs, nseg, 0),
                _cat_slot_ptr[dtype](segs, nseg, 1),
                _cat_slot_ptr[dtype](segs, nseg, 2),
                _cat_slot_ptr[dtype](segs, nseg, 3),
                _cat_slot_ptr[dtype](segs, nseg, 4),
                _cat_slot_ptr[dtype](segs, nseg, 5),
                _cat_slot_ptr[dtype](segs, nseg, 6),
                _cat_slot_ptr[dtype](segs, nseg, 7),
                segs,
                Int64(nseg),
                Int64(tiles),
                Int64(outer),
                Int64(dst_stride),
            )
        else:
            _enqueue_cached[_cat_batched_kernel[dtype, width]](
                ctx,
                String(t"dm_cat_batched_{dtype}_{width}"),
                gx,
                gy,
                1,
                GS_THREADS,
                out_ptr,
                segs,
                Int64(nseg),
                Int64(tiles),
                Int64(outer),
                Int64(dst_stride),
            )


@always_inline
def _cat_launch[
    dtype: DType
](
    out_addr: Int,
    srcs: PyObjectPtr,
    lens: PyObjectPtr,
    n: Int,
    outer: Int,
    dst_stride: Int,
    width: Int,
    ctx: DeviceContext,
) raises:
    comptime WIDE = CAT_VECTOR_BYTES // size_of[dtype]()
    if width == WIDE:
        _cat_launch_width[dtype, WIDE](
            out_addr, srcs, lens, n, outer, dst_stride, ctx
        )
    elif width == 1:
        _cat_launch_width[dtype, 1](
            out_addr, srcs, lens, n, outer, dst_stride, ctx
        )
    else:
        raise Error("unsupported vector width for batched cat")


def _cat_n_go(
    out_ptr_o: PyObjectPtr,
    srcs_o: PyObjectPtr,
    lens_o: PyObjectPtr,
    outer_o: PyObjectPtr,
    dst_stride_o: PyObjectPtr,
    itemsize_o: PyObjectPtr,
    width_o: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr_o)
    var outer = _raw_int(outer_o)
    var dst_stride = _raw_int(dst_stride_o)
    var itemsize = _raw_int(itemsize_o)
    var width = _raw_int(width_o)
    var ctx = _raw_ctx(ctx_ptr)
    var n = _raw_tuple_len(srcs_o)

    if ctx.api() == "cpu" or n <= 0 or outer <= 0 or dst_stride <= 0:
        raise Error("batched cat preconditions not met")
    comptime if has_accelerator():
        comptime if _dtype_arg_width_on[0, 32]():
            if itemsize != 4:
                raise Error("cat specialization/itemsize mismatch")
            _cat_launch[DType.uint32](
                out_addr, srcs_o, lens_o, n, outer, dst_stride, width, ctx
            )
        elif _dtype_arg_width_on[0, 16]():
            if itemsize != 2:
                raise Error("cat specialization/itemsize mismatch")
            _cat_launch[DType.uint16](
                out_addr, srcs_o, lens_o, n, outer, dst_stride, width, ctx
            )
        elif _dtype_arg_width_on[0, 64]():
            if itemsize != 8:
                raise Error("cat specialization/itemsize mismatch")
            _cat_launch[DType.uint64](
                out_addr, srcs_o, lens_o, n, outer, dst_stride, width, ctx
            )
        elif _dtype_arg_width_on[0, 8]():
            if itemsize != 1:
                raise Error("cat specialization/itemsize mismatch")
            _cat_launch[DType.uint8](
                out_addr, srcs_o, lens_o, n, outer, dst_stride, width, ctx
            )
        else:
            raise Error("unsupported element size for batched cat")
    else:
        raise Error("no GPU accelerator available at compile time")


def _cat_n_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _cat_n_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
        )
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


# ---------------------------------------------------------------------------
# Narrow copy, destination-strided: the
# *source* is fully contiguous (`outer` blocks of `copy_len` elements) and
# lands in the destination at `outer_index * dst_stride + dst_offset`.
# Looping this over the inputs implements concatenation along any dim.
# ---------------------------------------------------------------------------


def _narrow_copy_dst_kernel2d[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst_stride_arg: Int64,
    copy_len4_arg: Int64,
    dst_offset_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var dst_stride = Int(dst_stride_arg)
    var copy_len4 = Int(copy_len4_arg)
    var dst_offset = Int(dst_offset_arg)
    comptime vec_align = 4 * size_of[dtype]()
    var o = Int(block_idx.y)
    var src_base = o * copy_len4 * 4
    var dst_base = o * dst_stride + dst_offset
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var cstride = Int(grid_dim.x) * Int(block_dim.x)
    while c < copy_len4:
        var j = c * 4
        out_ptr.store[width=4, alignment=vec_align](
            dst_base + j,
            in_ptr.load[width=4, alignment=vec_align](src_base + j),
        )
        c += cstride


def _narrow_copy_dst_kernel4[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst_stride_arg: Int64,
    copy_len_arg: Int64,
    dst_offset_arg: Int64,
    nchunks_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var dst_stride = Int(dst_stride_arg)
    var copy_len = Int(copy_len_arg)
    var dst_offset = Int(dst_offset_arg)
    var nchunks = Int(nchunks_arg)
    comptime vec_align = 4 * size_of[dtype]()
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while c < nchunks:
        var i = c * 4
        var o = i // copy_len
        var j = i % copy_len
        var v = in_ptr.load[width=4, alignment=vec_align](i)
        out_ptr.store[width=4, alignment=vec_align](
            o * dst_stride + dst_offset + j, v
        )
        c += gstride


def _narrow_copy_dst_kernel1[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst_stride_arg: Int64,
    copy_len_arg: Int64,
    dst_offset_arg: Int64,
    total_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var dst_stride = Int(dst_stride_arg)
    var copy_len = Int(copy_len_arg)
    var dst_offset = Int(dst_offset_arg)
    var total = Int(total_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while i < total:
        var o = i // copy_len
        var j = i % copy_len
        out_ptr[o * dst_stride + dst_offset + j] = in_ptr[i]
        i += gstride


@always_inline
def _narrow_copy_dst[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    outer: Int,
    dst_stride: Int,
    copy_len: Int,
    dst_offset: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    if ctx.api() != "cpu":
        comptime if has_accelerator():
            if (
                copy_len % 4 == 0
                and dst_stride % 4 == 0
                and dst_offset % 4 == 0
            ):
                # Vector fast path: float4 loads/stores (buffer bases are
                # over-aligned and every index below is a multiple of 4
                # elements). KV-cache concatenation hits this on each
                # decode step. One grid row per outer block keeps the
                # indexing division-free; the 1D chunk kernel covers the
                # (rare) outer counts past the grid-dim cap.
                var copy_len4 = copy_len // 4
                if outer <= 65535:
                    var gx = min((copy_len4 + GS_THREADS - 1) // GS_THREADS, 32)
                    _enqueue_cached[_narrow_copy_dst_kernel2d[dtype]](
                        ctx,
                        String(t"dm_narrowdst2d_{dtype}"),
                        max(gx, 1),
                        outer,
                        1,
                        GS_THREADS,
                        out_ptr.as_unsafe_any_origin(),
                        in_ptr.as_unsafe_any_origin().as_immutable(),
                        Int64(dst_stride),
                        Int64(copy_len4),
                        Int64(dst_offset),
                    )
                    return
                var nchunks = outer * copy_len4
                _enqueue_cached[_narrow_copy_dst_kernel4[dtype]](
                    ctx,
                    String(t"dm_narrowdst4_{dtype}"),
                    _gs_blocks(nchunks),
                    1,
                    1,
                    GS_THREADS,
                    out_ptr.as_unsafe_any_origin(),
                    in_ptr.as_unsafe_any_origin().as_immutable(),
                    Int64(dst_stride),
                    Int64(copy_len),
                    Int64(dst_offset),
                    Int64(nchunks),
                )
            else:
                var total = outer * copy_len
                _enqueue_cached[_narrow_copy_dst_kernel1[dtype]](
                    ctx,
                    String(t"dm_narrowdst1_{dtype}"),
                    _gs_blocks(total),
                    1,
                    1,
                    GS_THREADS,
                    out_ptr.as_unsafe_any_origin(),
                    in_ptr.as_unsafe_any_origin().as_immutable(),
                    Int64(dst_stride),
                    Int64(copy_len),
                    Int64(dst_offset),
                    Int64(total),
                )
            return
        else:
            raise Error("no GPU accelerator available at compile time")

    # No CPU-side "func4" fast path here (there was one, briefly): CPU
    # `elementwise[..., simd_width=1]` does not guarantee exactly
    # `Coord(outer * copy_len // 4)` calls to the callback with no overrun --
    # observed writing one extra width-4 store (4 elements) past `out`'s
    # allocation on this host. Silent when it lands in slack space, a
    # segfault when it does not: reproduced deterministically via
    # `test_matches_cpu_stack_mojo_int64`, which concatenates several small
    # int64 tensors back to back so a later allocation lands where an
    # earlier call's overrun wrote. The accelerator path above is untouched
    # -- its "float4" fast path launches a real GPU kernel
    # (`_narrow_copy_dst_kernel2d` / `kernel4`) with its own explicit
    # grid/thread bounds, not this CPU `elementwise` call, so it does not
    # share this bug. One scalar store per element is what CPU gets until
    # the overrun is root-caused inside `elementwise` itself.
    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var o = i // copy_len
        var j = i % copy_len
        out_ptr[o * dst_stride + dst_offset + j] = in_ptr[i]

    elementwise[func, simd_width=1](Coord(outer * copy_len), ctx)


def _narrow_copy_dst_go(
    out_ptr: PyObjectPtr,
    in_ptr: PyObjectPtr,
    outer: PyObjectPtr,
    dst_stride: PyObjectPtr,
    copy_len: PyObjectPtr,
    dst_offset: PyObjectPtr,
    itemsize_o: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr)
    var in_addr = _raw_int(in_ptr)
    var outer_val = _raw_int(outer)
    var dst_stride_val = _raw_int(dst_stride)
    var copy_len_val = _raw_int(copy_len)
    var dst_offset_val = _raw_int(dst_offset)
    var itemsize = _raw_int(itemsize_o)
    var ctx = _raw_ctx(ctx_ptr)

    comptime if _dtype_arg_width_on[0, 32]():
        if itemsize != 4:
            raise Error("narrow-copy specialization/itemsize mismatch")
        _narrow_copy_dst[DType.uint32](
            out_addr,
            in_addr,
            outer_val,
            dst_stride_val,
            copy_len_val,
            dst_offset_val,
            ctx,
        )
    elif _dtype_arg_width_on[0, 16]():
        if itemsize != 2:
            raise Error("narrow-copy specialization/itemsize mismatch")
        _narrow_copy_dst[DType.uint16](
            out_addr,
            in_addr,
            outer_val,
            dst_stride_val,
            copy_len_val,
            dst_offset_val,
            ctx,
        )
    elif _dtype_arg_width_on[0, 64]():
        if itemsize != 8:
            raise Error("narrow-copy specialization/itemsize mismatch")
        _narrow_copy_dst[DType.uint64](
            out_addr,
            in_addr,
            outer_val,
            dst_stride_val,
            copy_len_val,
            dst_offset_val,
            ctx,
        )
    elif _dtype_arg_width_on[0, 8]():
        if itemsize != 1:
            raise Error("narrow-copy specialization/itemsize mismatch")
        _narrow_copy_dst[DType.uint8](
            out_addr,
            in_addr,
            outer_val,
            dst_stride_val,
            copy_len_val,
            dst_offset_val,
            ctx,
        )
    else:
        raise Error("unsupported element size for fast narrow copy")


# ---------------------------------------------------------------------------
# Where: out[i] = cond ? a : b. The output is contiguous with dims padded
# to rank 4; cond (bool) and both operands are indexed with their own
# strides (0 on broadcast dims), so scalar operands are 1-element buffers
# with all-zero strides. A pure selection copy — kernels are specialized on
# element byte-size, not dtype.
#
# Tiered like logic_ops' binary dispatch (`_binary_bcast`/`_bin_flat_vec_kernel`):
# a flat-vec tier for the common case -- cond flat-contiguous over the
# output's extent, and each of a/b either flat-contiguous too or a single
# broadcast element (e.g. masked_fill's 0-d value tensor) -- skips the rank-4
# coordinate decomposition entirely (no integer division at all) and
# vector-loads/stores 16 bytes per thread; a general strided fallback covers
# anything else (a genuinely broadcasting mask, a non-contiguous operand, an
# unaligned view). Both go through `_enqueue_cached`, never the stdlib
# `elementwise` GPU dispatcher used previously: its per-call
# `compile_function` plus the six-division coordinate chain run on every
# element made this kernel 4-5x slower than stock PyTorch on a 357x789 case
# that should cost single-digit microseconds (see `_cast_vec_kernel`'s
# comment above for the same fix, applied earlier to CastSpec).
# ---------------------------------------------------------------------------


@__name(t"where_flat_vec_{dtype}")
def _where_flat_vec_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    cond_ptr: UnsafePointer[Scalar[DType.bool], ImmutAnyOrigin],
    a_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    total_arg: Int64,
    a_bcast_arg: Int64,
    b_bcast_arg: Int64,
):
    """16-byte-vectorized select for the no-broadcast, all-contiguous case:
    `cond` is flat over the output's extent (the launcher falls back to the
    strided kernel on the rare shape where it is not), and each of `a`/`b`
    is either flat too or a single broadcast element (`a_bcast`/`b_bcast`,
    e.g. a 0-d scalar operand), read once and splatted. Mirrors
    `_bin_flat_vec_kernel` in logic_ops.mojo.
    """
    comptime VW = 16 // size_of[dtype]()
    comptime vec_align = VW * size_of[dtype]()
    comptime cond_align = VW  # bool is 1 byte; matches the launcher's gate
    var total = Int(total_arg)
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    var nvec = total // VW

    @always_inline
    @parameter
    def pass_over[a_b: Bool, b_b: Bool]():
        var a_splat = SIMD[dtype, VW](a_ptr[0]) if a_b else SIMD[dtype, VW](0)
        var b_splat = SIMD[dtype, VW](b_ptr[0]) if b_b else SIMD[dtype, VW](0)
        var c = tid
        while c < nvec:
            var i = c * VW
            var cond = cond_ptr.load[width=VW, alignment=cond_align](i)
            var a = a_splat if a_b else a_ptr.load[
                width=VW, alignment=vec_align
            ](i)
            var b = b_splat if b_b else b_ptr.load[
                width=VW, alignment=vec_align
            ](i)
            out_ptr.store[width=VW, alignment=vec_align](i, cond.select(a, b))
            c += gstride
        var tail = total - nvec * VW
        if tid < tail:
            var i = nvec * VW + tid
            var a1 = a_splat[0] if a_b else a_ptr[i]
            var b1 = b_splat[0] if b_b else b_ptr[i]
            out_ptr[i] = a1 if cond_ptr[i] else b1

    # The splat flags are loop-invariant, so each arm is INSTANTIATED rather
    # than branched on per iteration (see `_bin_flat_vec_kernel`'s note on
    # the measured cost of a per-iteration uniform branch).
    if a_bcast_arg != 0:
        if b_bcast_arg != 0:
            pass_over[True, True]()
        else:
            pass_over[True, False]()
    elif b_bcast_arg != 0:
        pass_over[False, True]()
    else:
        pass_over[False, False]()


@__name(t"where_strided_{dtype}")
def _where_bcast_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    cond_ptr: UnsafePointer[Scalar[DType.bool], ImmutAnyOrigin],
    a_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    d1_arg: Int64,
    d2_arg: Int64,
    d3_arg: Int64,
    cs0_arg: Int64,
    cs1_arg: Int64,
    cs2_arg: Int64,
    cs3_arg: Int64,
    a_s0_arg: Int64,
    a_s1_arg: Int64,
    a_s2_arg: Int64,
    a_s3_arg: Int64,
    b_s0_arg: Int64,
    b_s1_arg: Int64,
    b_s2_arg: Int64,
    b_s3_arg: Int64,
    total_arg: Int64,
):
    """Fully general arm: any broadcast strides, one element per thread.

    Pays six integer divisions/moduli per element, so the launcher reaches
    it only when the flat-vec tier's layout does not apply -- a genuinely
    broadcasting mask/operand, or a non-16-byte-aligned view.
    """
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var d1 = Int(d1_arg)
    var d2 = Int(d2_arg)
    var d3 = Int(d3_arg)
    var cs0 = Int(cs0_arg)
    var cs1 = Int(cs1_arg)
    var cs2 = Int(cs2_arg)
    var cs3 = Int(cs3_arg)
    var a_s0 = Int(a_s0_arg)
    var a_s1 = Int(a_s1_arg)
    var a_s2 = Int(a_s2_arg)
    var a_s3 = Int(a_s3_arg)
    var b_s0 = Int(b_s0_arg)
    var b_s1 = Int(b_s1_arg)
    var b_s2 = Int(b_s2_arg)
    var b_s3 = Int(b_s3_arg)
    var total = Int(total_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while i < total:
        var i3 = i % d3
        var rest = i // d3
        var i2 = rest % d2
        rest = rest // d2
        var i1 = rest % d1
        var i0 = rest // d1
        var cbase = i0 * cs0 + i1 * cs1 + i2 * cs2 + i3 * cs3
        var abase = i0 * a_s0 + i1 * a_s1 + i2 * a_s2 + i3 * a_s3
        var bbase = i0 * b_s0 + i1 * b_s1 + i2 * b_s2 + i3 * b_s3
        out_ptr[i] = a_ptr[abase] if cond_ptr[cbase] else b_ptr[bbase]
        i += gstride


@always_inline
def _where_bcast[
    dtype: DType
](
    out_addr: Int,
    cond_addr: Int,
    a_addr: Int,
    b_addr: Int,
    d1: Int,
    d2: Int,
    d3: Int,
    cs0: Int,
    cs1: Int,
    cs2: Int,
    cs3: Int,
    a_s0: Int,
    a_s1: Int,
    a_s2: Int,
    a_s3: Int,
    b_s0: Int,
    b_s1: Int,
    b_s2: Int,
    b_s3: Int,
    total: Int,
    ctx: DeviceContext,
) raises:
    if total == 0:
        return
    var out_ptr = _make_ptr[dtype](out_addr)
    var cond_ptr = _make_ptr[DType.bool](cond_addr)
    var a_ptr = _make_ptr[dtype](a_addr)
    var b_ptr = _make_ptr[dtype](b_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(out_ptr, cond_ptr, a_ptr, b_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value())
            var i3 = i % d3
            var rest = i // d3
            var i2 = rest % d2
            rest = rest // d2
            var i1 = rest % d1
            var i0 = rest // d1
            var cbase = i0 * cs0 + i1 * cs1 + i2 * cs2 + i3 * cs3
            var abase = i0 * a_s0 + i1 * a_s1 + i2 * a_s2 + i3 * a_s3
            var bbase = i0 * b_s0 + i1 * b_s1 + i2 * b_s2 + i3 * b_s3
            out_ptr[i] = a_ptr[abase] if cond_ptr[cbase] else b_ptr[bbase]

        elementwise[func, simd_width=1](Coord(total), ctx)
        return

    comptime if dtype == DType.float64 and has_apple_gpu_accelerator():
        raise Error("float64 is not supported on Apple GPU")
    else:
        comptime if not has_accelerator():
            raise Error("no GPU accelerator available at compile time")
        else:
            comptime VW = 16 // size_of[dtype]()
            var d0 = total // max(1, d1 * d2 * d3)
            var cont3 = d3
            var cont2 = d2 * d3
            var cont1 = d1 * d2 * d3
            # An operand whose every stride is 0 is a single element -- what
            # masked_fill's 0-d value tensor looks like. The flat kernel
            # reads it once and splats it, so it neither breaks the flat
            # layout nor needs 16B alignment.
            var a_scalar = a_s0 == 0 and a_s1 == 0 and a_s2 == 0 and a_s3 == 0
            var b_scalar = b_s0 == 0 and b_s1 == 0 and b_s2 == 0 and b_s3 == 0
            var cond_flat = (
                (cs3 == 1 or d3 == 1)
                and (cs2 == cont3 or d2 == 1)
                and (cs1 == cont2 or d1 == 1)
                and (cs0 == cont1 or d0 == 1)
            )
            var a_flat = a_scalar or (
                (a_s3 == 1 or d3 == 1)
                and (a_s2 == cont3 or d2 == 1)
                and (a_s1 == cont2 or d1 == 1)
                and (a_s0 == cont1 or d0 == 1)
            )
            var b_flat = b_scalar or (
                (b_s3 == 1 or d3 == 1)
                and (b_s2 == cont3 or d2 == 1)
                and (b_s1 == cont2 or d1 == 1)
                and (b_s0 == cont1 or d0 == 1)
            )
            var aligned = (
                out_addr % 16 == 0
                and cond_addr % VW == 0
                and (a_scalar or a_addr % 16 == 0)
                and (b_scalar or b_addr % 16 == 0)
            )
            if aligned and cond_flat and a_flat and b_flat:
                # Bytes actually moved: a splatted operand reads one element
                # for the whole launch, so it does not count toward the
                # residency decision (see `_bw_flat_blocks`).
                var traffic = total * (
                    1  # cond: one byte per element, always read
                    + (0 if a_scalar else size_of[dtype]())
                    + (0 if b_scalar else size_of[dtype]())
                    + size_of[dtype]()
                )
                var slots = max(1, total // VW)
                _enqueue_cached[_where_flat_vec_kernel[dtype]](
                    ctx,
                    String(t"dm_where_fv_{dtype}"),
                    _bw_flat_blocks(slots, traffic),
                    1,
                    1,
                    GS_THREADS,
                    out_ptr.as_unsafe_any_origin(),
                    cond_ptr.as_unsafe_any_origin().as_immutable(),
                    a_ptr.as_unsafe_any_origin().as_immutable(),
                    b_ptr.as_unsafe_any_origin().as_immutable(),
                    Int64(total),
                    Int64(a_scalar),
                    Int64(b_scalar),
                )
                return

            _enqueue_cached[_where_bcast_kernel[dtype]](
                ctx,
                String(t"dm_where_bc_{dtype}"),
                _gs_blocks(total),
                1,
                1,
                GS_THREADS,
                out_ptr.as_unsafe_any_origin(),
                cond_ptr.as_unsafe_any_origin().as_immutable(),
                a_ptr.as_unsafe_any_origin().as_immutable(),
                b_ptr.as_unsafe_any_origin().as_immutable(),
                Int64(d1),
                Int64(d2),
                Int64(d3),
                Int64(cs0),
                Int64(cs1),
                Int64(cs2),
                Int64(cs3),
                Int64(a_s0),
                Int64(a_s1),
                Int64(a_s2),
                Int64(a_s3),
                Int64(b_s0),
                Int64(b_s1),
                Int64(b_s2),
                Int64(b_s3),
                Int64(total),
            )


# ---------------------------------------------------------------------------
# MaskedFillScalar: out[i] = cond[i] ? value : b[i], where `value` is a
# plain scalar baked into the kernel launch -- no device buffer, no separate
# materialization launch. Serves masked_fill(_).Scalar specifically:
# masked_fill(_).Tensor and where.self keep going through WhereSelect, whose
# `a` operand is already a real device tensor, so there is nothing to save
# there. Before this existed, the Scalar overload materialized `value` into
# a 0-d device tensor through the Fill kernel first (`_scalar_tensor_0d` in
# aten_fast.py) and then ran WhereSelect -- a second full kernel launch to
# move one repeated constant that this kernel now carries as an argument.
#
# Shares WhereSelect's rank-4 strided convention and flat-vec/strided
# tiering, and (like FillSpec) its element-WIDTH dispatch: `value` is
# narrowed to the real dtype and reinterpreted as same-width bits exactly
# once, on the host, via `_fill_bits` -- the kernels below never see a
# floating-point value, only bits to move, so float16/bfloat16 share one
# uint16 instantiation and so on.
# ---------------------------------------------------------------------------


@__name(t"masked_fill_scalar_flat_vec_{dtype}")
def _masked_fill_scalar_flat_vec_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    cond_ptr: UnsafePointer[Scalar[DType.bool], ImmutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value: Scalar[dtype],
    total_arg: Int64,
    b_bcast_arg: Int64,
):
    """`_where_flat_vec_kernel` with `a` fixed to an immediate."""
    comptime VW = 16 // size_of[dtype]()
    comptime vec_align = VW * size_of[dtype]()
    comptime cond_align = VW
    var total = Int(total_arg)
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    var nvec = total // VW
    var a_splat = SIMD[dtype, VW](value)

    @always_inline
    @parameter
    def pass_over[b_b: Bool]():
        var b_splat = SIMD[dtype, VW](b_ptr[0]) if b_b else SIMD[dtype, VW](0)
        var c = tid
        while c < nvec:
            var i = c * VW
            var cond = cond_ptr.load[width=VW, alignment=cond_align](i)
            var b = b_splat if b_b else b_ptr.load[
                width=VW, alignment=vec_align
            ](i)
            out_ptr.store[width=VW, alignment=vec_align](
                i, cond.select(a_splat, b)
            )
            c += gstride
        var tail = total - nvec * VW
        if tid < tail:
            var i = nvec * VW + tid
            var b1 = b_splat[0] if b_b else b_ptr[i]
            out_ptr[i] = value if cond_ptr[i] else b1

    if b_bcast_arg != 0:
        pass_over[True]()
    else:
        pass_over[False]()


@__name(t"masked_fill_scalar_strided_{dtype}")
def _masked_fill_scalar_bcast_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    cond_ptr: UnsafePointer[Scalar[DType.bool], ImmutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value: Scalar[dtype],
    d1_arg: Int64,
    d2_arg: Int64,
    d3_arg: Int64,
    cs0_arg: Int64,
    cs1_arg: Int64,
    cs2_arg: Int64,
    cs3_arg: Int64,
    b_s0_arg: Int64,
    b_s1_arg: Int64,
    b_s2_arg: Int64,
    b_s3_arg: Int64,
    total_arg: Int64,
):
    """`_where_bcast_kernel` with `a` fixed to an immediate."""
    var d1 = Int(d1_arg)
    var d2 = Int(d2_arg)
    var d3 = Int(d3_arg)
    var cs0 = Int(cs0_arg)
    var cs1 = Int(cs1_arg)
    var cs2 = Int(cs2_arg)
    var cs3 = Int(cs3_arg)
    var b_s0 = Int(b_s0_arg)
    var b_s1 = Int(b_s1_arg)
    var b_s2 = Int(b_s2_arg)
    var b_s3 = Int(b_s3_arg)
    var total = Int(total_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while i < total:
        var i3 = i % d3
        var rest = i // d3
        var i2 = rest % d2
        rest = rest // d2
        var i1 = rest % d1
        var i0 = rest // d1
        var cbase = i0 * cs0 + i1 * cs1 + i2 * cs2 + i3 * cs3
        var bbase = i0 * b_s0 + i1 * b_s1 + i2 * b_s2 + i3 * b_s3
        out_ptr[i] = value if cond_ptr[cbase] else b_ptr[bbase]
        i += gstride


@always_inline
def _masked_fill_scalar_bcast[
    dtype: DType
](
    out_addr: Int,
    cond_addr: Int,
    b_addr: Int,
    value: Scalar[dtype],
    d1: Int,
    d2: Int,
    d3: Int,
    cs0: Int,
    cs1: Int,
    cs2: Int,
    cs3: Int,
    b_s0: Int,
    b_s1: Int,
    b_s2: Int,
    b_s3: Int,
    total: Int,
    ctx: DeviceContext,
) raises:
    """Same tiering as `_where_bcast`, minus the `a` operand entirely: an
    element-WIDTH dtype (`_fill_bits_dtype`'s uintN), never the real one --
    the caller has already narrowed `value` to it."""
    if total == 0:
        return
    var out_ptr = _make_ptr[dtype](out_addr)
    var cond_ptr = _make_ptr[DType.bool](cond_addr)
    var b_ptr = _make_ptr[dtype](b_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(out_ptr, cond_ptr, b_ptr, value)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value())
            var i3 = i % d3
            var rest = i // d3
            var i2 = rest % d2
            rest = rest // d2
            var i1 = rest % d1
            var i0 = rest // d1
            var cbase = i0 * cs0 + i1 * cs1 + i2 * cs2 + i3 * cs3
            var bbase = i0 * b_s0 + i1 * b_s1 + i2 * b_s2 + i3 * b_s3
            out_ptr[i] = value if cond_ptr[cbase] else b_ptr[bbase]

        elementwise[func, simd_width=1](Coord(total), ctx)
        return

    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        comptime VW = 16 // size_of[dtype]()
        var d0 = total // max(1, d1 * d2 * d3)
        var cont3 = d3
        var cont2 = d2 * d3
        var cont1 = d1 * d2 * d3
        var b_scalar = b_s0 == 0 and b_s1 == 0 and b_s2 == 0 and b_s3 == 0
        var cond_flat = (
            (cs3 == 1 or d3 == 1)
            and (cs2 == cont3 or d2 == 1)
            and (cs1 == cont2 or d1 == 1)
            and (cs0 == cont1 or d0 == 1)
        )
        var b_flat = b_scalar or (
            (b_s3 == 1 or d3 == 1)
            and (b_s2 == cont3 or d2 == 1)
            and (b_s1 == cont2 or d1 == 1)
            and (b_s0 == cont1 or d0 == 1)
        )
        var aligned = (
            out_addr % 16 == 0
            and cond_addr % VW == 0
            and (b_scalar or b_addr % 16 == 0)
        )
        if aligned and cond_flat and b_flat:
            var traffic = total * (
                1 + (0 if b_scalar else size_of[dtype]()) + size_of[dtype]()
            )
            var slots = max(1, total // VW)
            _enqueue_cached[_masked_fill_scalar_flat_vec_kernel[dtype]](
                ctx,
                String(t"dm_mfs_fv_{dtype}"),
                _bw_flat_blocks(slots, traffic),
                1,
                1,
                GS_THREADS,
                out_ptr.as_unsafe_any_origin(),
                cond_ptr.as_unsafe_any_origin().as_immutable(),
                b_ptr.as_unsafe_any_origin().as_immutable(),
                value,
                Int64(total),
                Int64(b_scalar),
            )
            return

        _enqueue_cached[_masked_fill_scalar_bcast_kernel[dtype]](
            ctx,
            String(t"dm_mfs_bc_{dtype}"),
            _gs_blocks(total),
            1,
            1,
            GS_THREADS,
            out_ptr.as_unsafe_any_origin(),
            cond_ptr.as_unsafe_any_origin().as_immutable(),
            b_ptr.as_unsafe_any_origin().as_immutable(),
            value,
            Int64(d1),
            Int64(d2),
            Int64(d3),
            Int64(cs0),
            Int64(cs1),
            Int64(cs2),
            Int64(cs3),
            Int64(b_s0),
            Int64(b_s1),
            Int64(b_s2),
            Int64(b_s3),
            Int64(total),
        )


# `value` crosses the WIDTH-dispatch boundary as bits (`_fill_bits_dtype`),
# so this dispatcher is parametrized on the REAL dtype only to narrow the
# incoming Float64 correctly (float32 vs. its same-width int32/uint32 read
# very different bit patterns from the same numeric value) -- exactly the
# split `_fill`/`_fill_contig` already use for the plain Fill kernel.
# Scoped to the float dtypes masked_fill's fast eager path targets
# (`_FLOAT_DTYPES` in aten_fast.py); any other dtype's Scalar overload keeps
# going through the slower materialize-then-WhereSelect route in Python.
comptime _MASKED_FILL_SCALAR_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
]


def _masked_fill_scalar_go(
    out_ptr: PyObjectPtr,
    cond_ptr: PyObjectPtr,
    b_ptr: PyObjectPtr,
    params: PyObjectPtr,  # (d0..d3, cs0..cs3, bs0..bs3)
    value: PyObjectPtr,
    dtype_o: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr)
    var cond_addr = _raw_int(cond_ptr)
    var b_addr = _raw_int(b_ptr)
    var d0 = _raw_tuple_int(params, 0)
    var d1 = _raw_tuple_int(params, 1)
    var d2 = _raw_tuple_int(params, 2)
    var d3 = _raw_tuple_int(params, 3)
    var cs0 = _raw_tuple_int(params, 4)
    var cs1 = _raw_tuple_int(params, 5)
    var cs2 = _raw_tuple_int(params, 6)
    var cs3 = _raw_tuple_int(params, 7)
    var b_s0 = _raw_tuple_int(params, 8)
    var b_s1 = _raw_tuple_int(params, 9)
    var b_s2 = _raw_tuple_int(params, 10)
    var b_s3 = _raw_tuple_int(params, 11)
    var value_v = _raw_f64(value)
    var dtype_val = _raw_dtype_int(dtype_o)
    var total = d0 * d1 * d2 * d3
    var ctx = _raw_ctx(ctx_ptr)

    @always_inline
    @parameter
    def run[dt: DType]() raises:
        comptime BITS = _fill_bits_dtype[dt]()
        _masked_fill_scalar_bcast[BITS](
            out_addr,
            cond_addr,
            b_addr,
            _fill_bits[dt, BITS](value_v),
            d1,
            d2,
            d3,
            cs0,
            cs1,
            cs2,
            cs3,
            b_s0,
            b_s1,
            b_s2,
            b_s3,
            total,
            ctx,
        )

    # arg_dtypes on the Python side is (cond.dtype, b.dtype) -- index 0 is
    # always bool (the mask), so the real dtype the kernel specializes on is
    # index 1, matching `_launch_masked_fill_scalar`'s `arg_dtypes=(cond,
    # b)` ordering (and `_where_select_go`'s own `_dtype_arg_width_on[1,
    # ...]`, gated the same way for the same reason).
    var handled = False
    comptime for dt in _MASKED_FILL_SCALAR_DTYPES:
        comptime if _dtype_arg_on[1, dt]():
            if dtype_val == dt:
                run[dt]()
                handled = True
    if not handled:
        raise Error(
            "unsupported dtype for fast masked_fill scalar: "
            + String(dtype_val)
        )


@always_inline
def _dtype_size(dtype: DType) raises -> Int:
    """Element size in bytes for WhereSelect's `dtype` arg.

    `_where_bcast` (and its flat-vec/strided kernels) is a pure bit-move
    (SIMD select, no arithmetic), so it only needs to be specialized per
    byte-size, not per exact dtype.
    """
    if dtype == DType.float32 or dtype == DType.int32 or dtype == DType.uint32:
        return 4
    if (
        dtype == DType.float16
        or dtype == DType.bfloat16
        or dtype == DType.int16
        or dtype == DType.uint16
    ):
        return 2
    if dtype == DType.float64 or dtype == DType.int64 or dtype == DType.uint64:
        return 8
    if dtype == DType.int8 or dtype == DType.uint8 or dtype == DType.bool:
        return 1
    raise Error("unsupported dtype for fast where: " + String(dtype))


def _where_select_go(
    out_ptr: PyObjectPtr,
    cond_ptr: PyObjectPtr,
    a_ptr: PyObjectPtr,
    b_ptr: PyObjectPtr,
    params: PyObjectPtr,  # (d0..d3, cs0..cs3, as0..as3, bs0..bs3)
    dtype_o: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr)
    var cond_addr = _raw_int(cond_ptr)
    var a_addr = _raw_int(a_ptr)
    var b_addr = _raw_int(b_ptr)
    var d0 = _raw_tuple_int(params, 0)
    var d1 = _raw_tuple_int(params, 1)
    var d2 = _raw_tuple_int(params, 2)
    var d3 = _raw_tuple_int(params, 3)
    var cs0 = _raw_tuple_int(params, 4)
    var cs1 = _raw_tuple_int(params, 5)
    var cs2 = _raw_tuple_int(params, 6)
    var cs3 = _raw_tuple_int(params, 7)
    var a_s0 = _raw_tuple_int(params, 8)
    var a_s1 = _raw_tuple_int(params, 9)
    var a_s2 = _raw_tuple_int(params, 10)
    var a_s3 = _raw_tuple_int(params, 11)
    var b_s0 = _raw_tuple_int(params, 12)
    var b_s1 = _raw_tuple_int(params, 13)
    var b_s2 = _raw_tuple_int(params, 14)
    var b_s3 = _raw_tuple_int(params, 15)
    var total = d0 * d1 * d2 * d3
    var dtype = _raw_dtype_int(dtype_o)
    var ctx = _raw_ctx(ctx_ptr)

    var size = _dtype_size(dtype)
    comptime if _dtype_arg_width_on[1, 32]():
        if size != 4:
            raise Error("where specialization/dtype mismatch")
        _where_bcast[DType.uint32](
            out_addr,
            cond_addr,
            a_addr,
            b_addr,
            d1,
            d2,
            d3,
            cs0,
            cs1,
            cs2,
            cs3,
            a_s0,
            a_s1,
            a_s2,
            a_s3,
            b_s0,
            b_s1,
            b_s2,
            b_s3,
            total,
            ctx,
        )
    elif _dtype_arg_width_on[1, 16]():
        if size != 2:
            raise Error("where specialization/dtype mismatch")
        _where_bcast[DType.uint16](
            out_addr,
            cond_addr,
            a_addr,
            b_addr,
            d1,
            d2,
            d3,
            cs0,
            cs1,
            cs2,
            cs3,
            a_s0,
            a_s1,
            a_s2,
            a_s3,
            b_s0,
            b_s1,
            b_s2,
            b_s3,
            total,
            ctx,
        )
    elif _dtype_arg_width_on[1, 64]():
        if size != 8:
            raise Error("where specialization/dtype mismatch")
        _where_bcast[DType.uint64](
            out_addr,
            cond_addr,
            a_addr,
            b_addr,
            d1,
            d2,
            d3,
            cs0,
            cs1,
            cs2,
            cs3,
            a_s0,
            a_s1,
            a_s2,
            a_s3,
            b_s0,
            b_s1,
            b_s2,
            b_s3,
            total,
            ctx,
        )
    elif _dtype_arg_width_on[1, 8]():
        if size != 1:
            raise Error("where specialization/dtype mismatch")
        _where_bcast[DType.uint8](
            out_addr,
            cond_addr,
            a_addr,
            b_addr,
            d1,
            d2,
            d3,
            cs0,
            cs1,
            cs2,
            cs3,
            a_s0,
            a_s1,
            a_s2,
            a_s3,
            b_s0,
            b_s1,
            b_s2,
            b_s3,
            total,
            ctx,
        )
    else:
        raise Error("unsupported element size for fast where")


# ---------------------------------------------------------------------------
# Elementwise dtype cast between contiguous buffers of the same shape.
#
# A cast is pure bandwidth, so the only thing that matters is how many bytes a
# lane moves. This used to go through `elementwise` with `simd_width=1`: a
# BF16 store used 2 bytes of a 16-byte access, and every element paid its own
# address arithmetic and loop iteration. That sustained 3.3-3.7 TB/s of the
# ~4.1 TB/s a streaming copy gets on gfx942; one VEC-wide slot per thread with
# the grid sized to cover the slots exactly reaches 4.0-4.4, and going through
# `_enqueue_cached` also drops `elementwise`'s per-call `compile_function`.
#
# VEC is a compile-time regime and which one runs is a runtime decision: the
# widest whose access is naturally aligned for BOTH operands, down to 1. That
# check has to be on the addresses themselves -- a tensor can start at any
# element offset inside its storage (`x[1:]`), and the `alignment` parameter
# `elementwise` passes its body is the vector width, not a promise about the
# base pointer. An element count that is not a multiple of VEC leaves at most
# VEC-1 elements over, which the first VEC-1 threads finish one at a time.
# ---------------------------------------------------------------------------

# Grid geometry is `_bw_blocks` in op_utils, which carries the measurements.
comptime CAST_THREADS = GS_THREADS


def _cast_vec_kernel[
    src: DType, dst: DType, VEC: Int
](
    out_ptr: UnsafePointer[Scalar[dst], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[src], ImmutAnyOrigin],
    nvec_arg: Int64,
    size_arg: Int64,
):
    """`out[i] = cast(in[i])` for `size` elements, VEC of them per thread."""
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var nvec = Int(nvec_arg)
    var size = Int(size_arg)
    comptime IALIGN = min(16, VEC * size_of[src]())
    comptime OALIGN = min(16, VEC * size_of[dst]())
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    var j = tid
    while j < nvec:
        var v = in_ptr.load[width=VEC, alignment=IALIGN](j * VEC)
        comptime if dst == DType.bool:
            # An `i1` vector is not a storable value. Torch's bool is one byte
            # holding 0 or 1, so build those bytes and store them.
            out_ptr.bitcast[Scalar[DType.uint8]]().store[
                width=VEC, alignment=OALIGN
            ](
                j * VEC,
                v.ne(SIMD[src, VEC](0)).select(
                    SIMD[DType.uint8, VEC](1), SIMD[DType.uint8, VEC](0)
                ),
            )
        else:
            out_ptr.store[width=VEC, alignment=OALIGN](j * VEC, v.cast[dst]())
        j += gstride
    # The tail is at most VEC-1 elements and the grid is never narrower than
    # one CAST_THREADS-wide block, so the leading threads cover all of it.
    var t = nvec * VEC + tid
    if t < size:
        var a = in_ptr[t]
        comptime if dst == DType.bool:
            # rc1: Scalar[dst](Bool) requires an integral dtype; build the
            # concrete bool scalar first (cast is the identity here).
            out_ptr[t] = Scalar[DType.bool](a != Scalar[src](0)).cast[dst]()
        else:
            out_ptr[t] = a.cast[dst]()


@always_inline
def _cast[
    src: DType, dst: DType
](out_addr: Int, in_addr: Int, size: Int, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[dst](out_addr)
    var in_ptr = _make_ptr[src](in_addr)
    if size == 0:
        return

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value())
            var a = in_ptr[i]
            comptime if dst == DType.bool:
                # rc1: Scalar[dst](Bool) requires an integral dtype; build the
                # concrete bool scalar first (cast is the identity here).
                out_ptr[i] = Scalar[DType.bool](a != Scalar[src](0)).cast[dst]()
            else:
                out_ptr[i] = a.cast[dst]()

        elementwise[func, simd_width=1](Coord(size), ctx)
        return

    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:

        @always_inline
        @parameter
        def _try_cast[VEC: Int]() raises -> Bool:
            comptime IALIGN = min(16, VEC * size_of[src]())
            comptime OALIGN = min(16, VEC * size_of[dst]())
            if in_addr % IALIGN != 0 or out_addr % OALIGN != 0:
                return False
            var nvec = size // VEC
            # One 16-byte access per thread on the wider operand. At the widest
            # VEC that is one slot per thread and the grid covers the slots
            # exactly; a narrower VEC (taken only when the addresses are not
            # vector-aligned) gets proportionally more slots per thread instead
            # of a proportionally larger grid, which measured 49.8 us against
            # 112 us for one 4-byte element per thread on [48, 1024, 768].
            comptime SLOTS = max(
                1, 16 // (VEC * max(size_of[src](), size_of[dst]()))
            )
            _enqueue_cached[_cast_vec_kernel[src, dst, VEC]](
                ctx,
                String(t"dm_cast_{src}_{dst}_v{VEC}"),
                _bw_blocks(
                    nvec,
                    SLOTS,
                    SLOTS == 1
                    and size * (size_of[src]() + size_of[dst]()) <= _LLC_BYTES,
                    ctx,
                ),
                1,
                1,
                CAST_THREADS,
                out_ptr.as_unsafe_any_origin(),
                in_ptr.as_unsafe_any_origin().as_immutable(),
                Int64(nvec),
                Int64(size),
            )
            return True

        # 16 bytes per lane on the wider operand; the narrower one moves half
        # of that. Wider than 16 measured no better on either nanoGPT shape.
        comptime WIDEST = 16 // max(size_of[src](), size_of[dst]())
        comptime if WIDEST >= 16:
            if _try_cast[16]():
                return
        comptime if WIDEST >= 8:
            if _try_cast[8]():
                return
        comptime if WIDEST >= 4:
            if _try_cast[4]():
                return
        comptime if WIDEST >= 2:
            if _try_cast[2]():
                return
        # VEC=1 needs no alignment, so this always launches.
        _ = _try_cast[1]()


# The dtypes fast cast supports on either end. Both the src and dst
# dispatch loops iterate this list at compile time.
comptime CAST_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.int64,
    DType.int32,
    DType.uint8,
    DType.bool,
]


@always_inline
def _cast_to[
    src: DType
](
    dst: DType, out_addr: Int, in_addr: Int, size: Int, ctx: DeviceContext
) raises:
    var handled = False
    comptime for dst_dt in CAST_DTYPES:
        comptime if _dtype_out_on[0, dst_dt]():
            if dst == dst_dt:
                _cast[src, dst_dt](out_addr, in_addr, size, ctx)
                handled = True
    if not handled:
        raise Error(
            "unsupported destination dtype for fast cast: " + String(dst)
        )


# ---------------------------------------------------------------------------
# TileCopy: out[coords] = in[coords % in_shape] over a rank-<=8 index space.
# Materializes aten::repeat: the Python side left-pads the input shape with
# 1s to the output rank, computes out_shape[d] = padded_in_shape[d] *
# repeats[d], and hands the (contiguous) input's row-major strides. Broadcast
# padded dims (in_shape 1) reduce to `coord % 1 == 0`. Layout-only -> element
# size dispatch.
# ---------------------------------------------------------------------------


@always_inline
def _tile_copy[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    out_shape: IndexList[MAX_RANK],
    in_shape: IndexList[MAX_RANK],
    in_strides: IndexList[MAX_RANK],
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var total = 1
    for i in range(MAX_RANK):
        total *= out_shape[i]
    if total == 0:
        return

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr, out_shape, in_shape, in_strides)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var rest = i
        var src_off = 0

        comptime for d in range(MAX_RANK - 1, 0, -1):
            var coord = rest % out_shape[d]
            rest = rest // out_shape[d]
            src_off += (coord % in_shape[d]) * in_strides[d]
        src_off += (rest % in_shape[0]) * in_strides[0]
        out_ptr[i] = in_ptr[src_off]

    _parallel_for[func](total, ctx)


def _tile_copy_go(
    out_ptr: PyObjectPtr,
    in_ptr: PyObjectPtr,
    out_shape_t: PyObjectPtr,
    in_shape_t: PyObjectPtr,
    in_strides_t: PyObjectPtr,
    itemsize_o: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr)
    var in_addr = _raw_int(in_ptr)
    var out_shape = IndexList[MAX_RANK](1)
    var in_shape = IndexList[MAX_RANK](1)
    var in_strides = IndexList[MAX_RANK](0)
    for i in range(MAX_RANK):
        out_shape[i] = _raw_tuple_int(out_shape_t, i)
        in_shape[i] = _raw_tuple_int(in_shape_t, i)
        in_strides[i] = _raw_tuple_int(in_strides_t, i)
    var itemsize = _raw_int(itemsize_o)
    var ctx = _raw_ctx(ctx_ptr)

    comptime if _dtype_arg_width_on[0, 32]():
        if itemsize != 4:
            raise Error("tile-copy specialization/itemsize mismatch")
        _tile_copy[DType.uint32](
            out_addr, in_addr, out_shape, in_shape, in_strides, ctx
        )
    elif _dtype_arg_width_on[0, 16]():
        if itemsize != 2:
            raise Error("tile-copy specialization/itemsize mismatch")
        _tile_copy[DType.uint16](
            out_addr, in_addr, out_shape, in_shape, in_strides, ctx
        )
    elif _dtype_arg_width_on[0, 64]():
        if itemsize != 8:
            raise Error("tile-copy specialization/itemsize mismatch")
        _tile_copy[DType.uint64](
            out_addr, in_addr, out_shape, in_shape, in_strides, ctx
        )
    elif _dtype_arg_width_on[0, 8]():
        if itemsize != 1:
            raise Error("tile-copy specialization/itemsize mismatch")
        _tile_copy[DType.uint8](
            out_addr, in_addr, out_shape, in_shape, in_strides, ctx
        )
    else:
        raise Error("TileCopy: unsupported element size ", itemsize)


# ---------------------------------------------------------------------------
# RepeatTiled: aten::repeat when it reduces to a rank-2 tiled copy.
#
#   out[copy * rows + ir, s * cols + c] = in[ir, c]
#
# for ir in [0, rows), c in [0, cols), s in [0, r1), copy in [0, ncopies).
# `rows x cols` is the contiguous input, `r1` the repeat factor along the
# columns and `ncopies` the number of stacked copies of the resulting
# (rows, cols * r1) base block -- `repeats[-2]` times the product of every
# repeat factor left of it, which is exact whenever the padded input shape is
# 1 on all but its last two dims (the Python side checks that and keeps
# `_tile_copy` above for everything else).
#
# WHY IT IS A SEPARATE KERNEL. `_tile_copy` walks a rank-8-PADDED index space
# one output element per thread, so it pays ~8 integer divisions and 16
# moduli per element even for a rank-2 tensor, and lands at ~4% of achievable
# bandwidth. Writing the mapping the way above rather than as
# `out[i, j] = in[i % rows, j % cols]` makes every index a loop counter: grid
# x carries `c`, y carries the OUTPUT row, `s` is an inner loop, and not one
# integer division survives either kernel's inner loop (verified in the
# emitted PTX, not assumed). Measured on an H100 PCIe at 1395 MHz, kineto
# DEVICE time against stock CUDA: 1024x1024 r(4,4) 41.4us vs 73.6
# (`_tile_copy`: 1086), 357x789 r(3,5) 10.6 vs 19.7 (301), 8192x64 r(1,16)
# 14.7 vs 32.2 (594), 64x8192 r(16,1) 16.5 vs 26.9 (595).
#
# THE COPY AXIS IS FOLDED INTO THE OUTPUT ROW, not given a grid dimension of
# its own. Output rows are contiguous ACROSS copies as well as within one, so
# `copy * rows + ir` is just an output row index and the whole launch is
# (row interior) x (output rows). Splitting them cost a grid dimension AND
# left blocks idle whenever `rows` was smaller than the block's y extent: at
# rows=1, cols=3 only 3 of 255 threads survived the row guard and the copy
# ran 86.5us against 45.2 for the general `_tile_copy` it replaced. The price
# is one modulo per THREAD to recover `ir = orow % rows`; the per-step
# advance is uniform, so the host computes it and the kernel carries `ir`
# with an add and a conditional subtract, exactly like the flat kernel's
# source column.
#
# TWO REGIMES, dispatched on the row length in BYTES (SEG_MIN_BYTES below)
# and on whether the segment kernel's grid can fill the device
# (`_repeat_seg_blocks`). Layout-only -> element-size dispatch, like
# `_tile_copy`.
# ---------------------------------------------------------------------------

# Threads per block (threads_x * threads_y) for both repeat kernels. 256 is
# this package's house block size (GS_THREADS, CAST_THREADS) rather than a
# fitted number, and the H100 ladder in RESULTS.md was measured at it; no
# other block size has been measured for these kernels on any card.
comptime REPEAT_THREADS = 256

# The row length, in BYTES, at or above which the segment kernel takes over
# from the flat one. It is the size of the contiguous chunk one input row
# contributes to a warp's store, and 64 bytes is two 32-byte DRAM sectors --
# the point where a per-row chunk stops being a partial transaction. Measured
# ladder on an H100 (flat vs seg, streamed us, same output size): 16B
# 7.20/14.93, 32B 7.04/7.80, 48B 7.16/10.38 | 64B 5.11/4.76, 96B 7.06/6.67,
# 256B 17.96/16.01, so the crossover is between 48 and 64 bytes there. It is a
# transaction granularity rather than a fitted model constant, but it has only
# been measured on that card; re-measure on MI300X and Metal.
comptime SEG_MIN_BYTES = 64

# Widest access the automatic width choice will take, in BYTES. 16 is the
# widest access the hardware has, so this is an ISA ceiling rather than a
# fitted constant; what was measured (on an H100, and only there) is that
# asking for more does not help -- 32 bytes a thread, VEC=8 on f32, only
# doubles the per-thread footprint and shrinks the grid, and came out worse
# on every shape: 57.5 vs 43.9us on 1024x1024, 35.1 vs 16.1 on 8192x64, 25.4
# vs 17.1 on 64x8192.
comptime REPEAT_MAX_BYTES = 16

# Both kernels advance their 32-bit column counter by a whole grid stride, so
# a single row within ~2048 elements of 2^31 could wrap it: value-correct
# (the counters are unsigned) but the loop bound would stop terminating on
# schedule. A row that long is an 8GB operand; the caller keeps the general
# path for it rather than widening every counter.
comptime _REPEAT_MAX_EXTENT = 0x7FFF_F000


@always_inline
def _repeat_quantize(full: Int, cap: Int) -> Int:
    """Largest block count <= `cap` giving every block the SAME number of
    grid-stride iterations.

    Capping a grid at a budget leaves `full / cap` iterations per block on
    average but `ceil(full / cap)` for the unlucky ones, and a launch is as
    long as its slowest block: 8192 rows over 7296 blocks means 896 blocks do
    two rows while 6400 do one, so the kernel takes twice the average work.
    Rounding the grid DOWN to `ceil(full / k)` blocks of exactly `k`
    iterations costs nothing and removes the tail (measured on 8192x64
    r(1,16): 34.7us -> 24.6us).

    Op-agnostic, and strictly better than the bare `min(full, _MAX_GRID_Y)`
    that `_permute_copy` and `_transpose2d` cap their batch grids with. It is
    kept local rather than promoted to `op_utils` on purpose: the two sites
    that would adopt it have no benchmark node of their own yet, and moving
    the helper alone would invalidate the compile cache of every extension
    for no behavioural gain.
    """
    if full <= cap:
        return max(1, full)
    var k = ceildiv(full, cap)
    return ceildiv(full, k)


@always_inline
def _repeat_seg_blocks(nseg: Int, nout: Int) -> Int:
    """Blocks the segment kernel's grid would have for this geometry.

    The segment kernel's `r1` loop is SERIAL -- one load feeding `r1` stores
    is the whole point of it -- so its grid is (row interior) x (output rows)
    and knows nothing about how wide `r1` makes the output. A repeat that is
    wide precisely BECAUSE `r1` is large (a short input row copied thousands
    of times across, with few output rows) therefore leaves it with almost no
    grid at all: rows=1, cols=64, r1=9375 put a 600k-element copy in ONE
    block and took 87.6us against 45.3 for the general path. The flat
    kernel's x axis spans the whole output row, `r1` included, so it is the
    one with parallelism there.

    The caller hands over to it when this count cannot put a block on every
    SM. That floor is a property of the device (`_device_sm_count`), not a
    tuned number, and the two regimes are three orders of magnitude apart:
    8192x64 r(1,16), which wants the segment kernel, has 512 blocks here
    against the pathological case's 1.
    """
    var tx = min(nseg, REPEAT_THREADS)
    var ty = max(1, REPEAT_THREADS // tx)
    return ceildiv(nseg, tx) * ceildiv(nout, ty)


@always_inline
def _repeat_first_row(orow: UInt32, rows: UInt32, nout: UInt32) -> UInt32:
    """The input row output row `orow` reads, `orow % rows`, without paying a
    division for the two shapes that do not need one.

    This runs once per THREAD, not per element, and both branches are on
    kernel ARGUMENTS, so every thread of the launch takes the same one. It is
    still worth skipping: 64x8192 r(16,1) does one 16-byte load and one
    16-byte store per thread -- the worst instruction-to-byte ratio either
    kernel reaches -- and the bare modulo cost it 6.5% (16.3 -> 17.3us).
    `rows == 1` is the shape the `r1 == 1` flattening in `_repeat_tiled`
    turns that case into; `nout == rows` is `ncopies == 1`, where the output
    row IS the input row.
    """
    if rows == 1:
        return 0
    if nout == rows:
        return orow
    return orow % rows


# Index math is 32-bit on purpose in both kernels. `Int` is 64 bits on device,
# so every counter update costs two instructions; only the row base ADDRESSES
# are 64-bit, and they are computed once per row rather than per element. The
# caller keeps `_tile_copy` for operands whose extents do not fit in 31 bits
# (a >2G-element row or row count, i.e. >8GB of tensor) instead of making
# every launch pay 64-bit counters.


@__name(t"repeat_seg_rowmajor_{dtype}_v{VEC}")
def _repeat_seg_kernel[
    dtype: DType, VEC: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
    cout_arg: Int64,
    nout_arg: Int64,
    r1_arg: Int64,
    iradv_arg: Int64,
):
    """x = inside one input row, y = OUTPUT rows (`rows * ncopies`); `r1` is an
    inner loop, so one load feeds `r1` stores and the input row is read once
    instead of `r1` times. No division in either loop."""
    # 16 bytes is the widest access the hardware has; a wider VEC issues two.
    comptime ALIGN = min(16, VEC * size_of[dtype]())
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays 32-bit.
    var rows = UInt32(Int(rows_arg))
    var cols = UInt32(Int(cols_arg))
    var cout = Int(cout_arg)
    var nout = UInt32(Int(nout_arg))
    var r1 = Int(r1_arg)
    var iradv = UInt32(Int(iradv_arg))

    var c0 = (
        UInt32(Int(block_idx.x)) * UInt32(Int(block_dim.x))
        + UInt32(Int(thread_idx.x))
    ) * UInt32(VEC)
    var cstride = (
        UInt32(Int(grid_dim.x)) * UInt32(Int(block_dim.x)) * UInt32(VEC)
    )
    var orow = UInt32(Int(block_idx.y)) * UInt32(Int(block_dim.y)) + UInt32(
        Int(thread_idx.y)
    )
    var orowstride = UInt32(Int(grid_dim.y)) * UInt32(Int(block_dim.y))
    var ir = _repeat_first_row(orow, rows, nout)

    while orow < nout:
        var in_base = Int(ir) * Int(cols)
        var out_base = Int(orow) * cout
        var c = c0
        while c < cols:
            var v = in_ptr.load[width=VEC, alignment=ALIGN](in_base + Int(c))
            var o = out_base + Int(c)
            for _ in range(r1):
                out_ptr.store[width=VEC, alignment=ALIGN](o, v)
                o += Int(cols)
            c += cstride
        orow += orowstride
        ir += iradv
        if ir >= rows:
            ir -= rows


@__name(t"repeat_flat_rowmajor_{dtype}_v{VEC}")
def _repeat_flat_kernel[
    dtype: DType, VEC: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
    cout_arg: Int64,
    nout_arg: Int64,
    cadv_arg: Int64,
    iradv_arg: Int64,
):
    """For rows too short to fill a transaction, and for outputs whose width
    comes from `r1` rather than from `cols`: x indexes the output row FLAT
    (across segment boundaries), y = OUTPUT rows (`rows * ncopies`).

    Output rows are adjacent in memory, so with `threads_x == nslots` the
    block's linear thread id maps to consecutive output elements even across a
    row boundary and a warp stays fully coalesced over, say, a 40-element row.
    The price is two moduli per THREAD -- one to place its first source
    column, one to place its first source row; both per-step advances are
    uniform, so the host computes them and the kernel carries each with an add
    and a conditional subtract.
    """
    comptime ALIGN = min(16, VEC * size_of[dtype]())
    var rows = UInt32(Int(rows_arg))
    var cols = UInt32(Int(cols_arg))
    var cout = Int(cout_arg)
    var nout = UInt32(Int(nout_arg))
    var cadv = UInt32(Int(cadv_arg))
    var iradv = UInt32(Int(iradv_arg))
    var nslots = UInt32(Int(cout_arg) // VEC)

    var slot0 = UInt32(Int(block_idx.x)) * UInt32(Int(block_dim.x)) + UInt32(
        Int(thread_idx.x)
    )
    var xstride = UInt32(Int(grid_dim.x)) * UInt32(Int(block_dim.x))
    var orow = UInt32(Int(block_idx.y)) * UInt32(Int(block_dim.y)) + UInt32(
        Int(thread_idx.y)
    )
    var orowstride = UInt32(Int(grid_dim.y)) * UInt32(Int(block_dim.y))
    if slot0 >= nslots:
        return
    var c0 = (slot0 * UInt32(VEC)) % cols
    var ir = _repeat_first_row(orow, rows, nout)

    while orow < nout:
        var in_base = Int(ir) * Int(cols)
        var out_base = Int(orow) * cout
        var slot = slot0
        var c = c0
        while slot < nslots:
            out_ptr.store[width=VEC, alignment=ALIGN](
                out_base + Int(slot) * VEC,
                in_ptr.load[width=VEC, alignment=ALIGN](in_base + Int(c)),
            )
            slot += xstride
            c += cadv
            if c >= cols:
                c -= cols
        orow += orowstride
        ir += iradv
        if ir >= rows:
            ir -= rows


@always_inline
def _repeat_seg_launch[
    dtype: DType, VEC: Int
](
    out_addr: Int,
    in_addr: Int,
    rows: Int,
    cols: Int,
    cout: Int,
    r1: Int,
    nout: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var nseg = cols // VEC  # vector slots inside one input row
    var tx = min(nseg, REPEAT_THREADS)
    var ty = max(1, REPEAT_THREADS // tx)
    var gy = _repeat_quantize(ceildiv(nout, ty), _MAX_GRID_Y)
    _enqueue_cached_2d[_repeat_seg_kernel[dtype, VEC]](
        ctx,
        String(t"dm_rptseg_{dtype}_v{VEC}"),
        _repeat_quantize(ceildiv(nseg, tx), _BW_MAX_BLOCKS),
        gy,
        1,
        tx,
        ty,
        out_ptr.as_unsafe_any_origin(),
        in_ptr.as_unsafe_any_origin().as_immutable(),
        Int64(rows),
        Int64(cols),
        Int64(cout),
        Int64(nout),
        Int64(r1),
        Int64((gy * ty) % rows),
    )


@always_inline
def _repeat_flat_launch[
    dtype: DType, VEC: Int
](
    out_addr: Int,
    in_addr: Int,
    rows: Int,
    cols: Int,
    cout: Int,
    nout: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var nslots = cout // VEC
    # One output row's worth of threads, capped at a block: `threads_x ==
    # nslots` is what keeps a warp coalesced across a row boundary. Rounding
    # it up to a warp instead -- the obvious "a warp must not straddle two
    # rows" rule -- measured 33.4us against 13.5 on 65536x8 r(1,5), because it
    # left 7/8 of every block idle on a 40-element row.
    var tx = min(nslots, REPEAT_THREADS)
    var ty = max(1, REPEAT_THREADS // tx)
    var gx = _repeat_quantize(ceildiv(nslots, tx), _BW_MAX_BLOCKS)
    var gy = _repeat_quantize(ceildiv(nout, ty), _MAX_GRID_Y)
    _enqueue_cached_2d[_repeat_flat_kernel[dtype, VEC]](
        ctx,
        String(t"dm_rptflat_{dtype}_v{VEC}"),
        gx,
        gy,
        1,
        tx,
        ty,
        out_ptr.as_unsafe_any_origin(),
        in_ptr.as_unsafe_any_origin().as_immutable(),
        Int64(rows),
        Int64(cols),
        Int64(cout),
        Int64(nout),
        Int64((gx * tx * VEC) % cols),
        Int64((gy * ty) % rows),
    )


@always_inline
def _repeat_tiled[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    rows: Int,
    cols: Int,
    r1: Int,
    ncopies: Int,
    ctx: DeviceContext,
) raises:
    """Pick the vector width from the RUNTIME addresses and the kernel from the
    row length and the grid it would get, then launch the EXACT grid.

    The exact grid (one loop iteration per thread, capped only by the 65535
    limit on grid.y with `_repeat_quantize` keeping every block's iteration
    count equal) is a result rather than a default: while the kernel still had
    four integer divisions per thread the grid mattered enormously -- 66.5us
    against 21.7 for a one-wave grid on 357x789 -- and with the divisions gone
    the whole wave-budget ladder is flat to within 2% on every measured shape.
    The device is consulted for ONE thing only, and not to size the grid: the
    segment kernel is skipped when its grid cannot put a block on every SM
    (`_repeat_seg_blocks`).
    """
    if rows <= 0 or cols <= 0 or r1 <= 0 or ncopies <= 0:
        return
    if cols * r1 > _REPEAT_MAX_EXTENT or rows * ncopies > _REPEAT_MAX_EXTENT:
        raise Error("RepeatTiled: extent does not fit a 32-bit counter")

    # `r1 == 1` means nothing is tiled ALONG a row, so the whole input block
    # is one contiguous run and the output is just `ncopies` copies of it: the
    # (rows, cols) input flattens to a single row of rows*cols, which is the
    # same bytes in the same order. That is a strict simplification -- it
    # removes the input-row modulo entirely (rows becomes 1) and hands the
    # segment kernel a long row to spread over the grid instead of a short one
    # -- and it is the shape `x.repeat(k, 1)` produces.
    var erows = rows
    var ecols = cols
    if r1 == 1 and rows > 1 and rows * cols <= _REPEAT_MAX_EXTENT:
        erows = 1
        ecols = rows * cols
    var cout = ecols * r1
    var nout = erows * ncopies  # output rows, copies folded in
    var wide = ecols * size_of[dtype]() >= SEG_MIN_BYTES
    var sm_count = _device_sm_count(ctx)

    @always_inline
    @parameter
    def _launch[V: Int]() raises:
        if wide and _repeat_seg_blocks(ecols // V, nout) >= sm_count:
            _repeat_seg_launch[dtype, V](
                out_addr, in_addr, erows, ecols, cout, r1, nout, ctx
            )
        else:
            _repeat_flat_launch[dtype, V](
                out_addr, in_addr, erows, ecols, cout, nout, ctx
            )

    # A VEC-wide access needs a VEC*itemsize-byte aligned address (capped at
    # the 16-byte hardware maximum) on BOTH operands, and needs `ecols` to be
    # a multiple of VEC -- otherwise a row start is not aligned and a vector
    # would straddle the `j % cols` wrap. Gate on the runtime ADDRESSES: an
    # offset view (`x[1:]`) satisfies every divisibility test and still faults
    # (proven by a negative control that forces the 16-byte path onto a
    # 4-byte-offset base and dies with CUDA_ERROR_MISALIGNED_ADDRESS).
    var handled = False
    comptime auto_max = max(1, REPEAT_MAX_BYTES // size_of[dtype]())
    comptime for V in [16, 8, 4, 2]:
        # Descending, so the widest access up to REPEAT_MAX_BYTES that the
        # shape and the addresses allow wins. Only the widths that can ever
        # apply to this element size are instantiated. VEC > 4 is reached only
        # by the 1- and 2-byte dtypes, which were not part of the f32
        # measurements above.
        comptime if V <= auto_max:
            comptime ALIGN = min(16, V * size_of[dtype]())
            if (
                not handled
                and ecols % V == 0
                and out_addr % ALIGN == 0
                and in_addr % ALIGN == 0
            ):
                _launch[V]()
                handled = True
    if not handled:
        # Scalar path. Consecutive threads still touch consecutive output
        # elements, so the stores stay perfectly coalesced; only the
        # instruction count grows.
        _launch[1]()


def _repeat_tiled_go(
    out_ptr: PyObjectPtr,
    in_ptr: PyObjectPtr,
    rows_o: PyObjectPtr,
    cols_o: PyObjectPtr,
    r1_o: PyObjectPtr,
    ncopies_o: PyObjectPtr,
    itemsize_o: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr)
    var in_addr = _raw_int(in_ptr)
    var rows = _raw_int(rows_o)
    var cols = _raw_int(cols_o)
    var r1 = _raw_int(r1_o)
    var ncopies = _raw_int(ncopies_o)
    var itemsize = _raw_int(itemsize_o)
    var ctx = _raw_ctx(ctx_ptr)

    comptime if not has_accelerator():
        raise Error("RepeatTiled: no accelerator in this build")
    elif _dtype_arg_width_on[0, 32]():
        if itemsize != 4:
            raise Error("repeat-tiled specialization/itemsize mismatch")
        _repeat_tiled[DType.uint32](
            out_addr, in_addr, rows, cols, r1, ncopies, ctx
        )
    elif _dtype_arg_width_on[0, 16]():
        if itemsize != 2:
            raise Error("repeat-tiled specialization/itemsize mismatch")
        _repeat_tiled[DType.uint16](
            out_addr, in_addr, rows, cols, r1, ncopies, ctx
        )
    elif _dtype_arg_width_on[0, 64]():
        if itemsize != 8:
            raise Error("repeat-tiled specialization/itemsize mismatch")
        _repeat_tiled[DType.uint64](
            out_addr, in_addr, rows, cols, r1, ncopies, ctx
        )
    elif _dtype_arg_width_on[0, 8]():
        if itemsize != 1:
            raise Error("repeat-tiled specialization/itemsize mismatch")
        _repeat_tiled[DType.uint8](
            out_addr, in_addr, rows, cols, r1, ncopies, ctx
        )
    else:
        raise Error("RepeatTiled: unsupported element size ", itemsize)


# ---------------------------------------------------------------------------
# TriangularCopy: out = in where the (row, col) is on the kept side of the
# diagonal, else 0. Implements aten::tril (upper == 0, keep col <= row + diag)
# and aten::triu (upper == 1, keep col >= row + diag) over a batch of
# (rows, cols) matrices (batch = numel / (rows * cols)). Both operands are
# contiguous; copy-or-zero, so element-size dispatch (0 bytes == 0 for every
# dtype).
# ---------------------------------------------------------------------------


@always_inline
def _triangular_copy[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    batch: Int,
    rows: Int,
    cols: Int,
    diagonal: Int,
    upper: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var total = batch * rows * cols

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var c = i % cols
        var r = (i // cols) % rows
        var keep: Bool
        if upper != 0:
            keep = c >= r + diagonal
        else:
            keep = c <= r + diagonal
        if keep:
            out_ptr[i] = in_ptr[i]
        else:
            out_ptr[i] = Scalar[dtype](0)

    _parallel_for[func](total, ctx)


def _triangular_copy_go(
    out_ptr: PyObjectPtr,
    in_ptr: PyObjectPtr,
    batch_o: PyObjectPtr,
    rows_o: PyObjectPtr,
    cols_o: PyObjectPtr,
    diagonal_o: PyObjectPtr,
    upper_o: PyObjectPtr,
    itemsize_o: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr)
    var in_addr = _raw_int(in_ptr)
    var batch = _raw_int(batch_o)
    var rows = _raw_int(rows_o)
    var cols = _raw_int(cols_o)
    var diagonal = _raw_int(diagonal_o)
    var upper = _raw_int(upper_o)
    var itemsize = _raw_int(itemsize_o)
    var ctx = _raw_ctx(ctx_ptr)

    comptime if _dtype_arg_width_on[0, 32]():
        if itemsize != 4:
            raise Error("triangular specialization/itemsize mismatch")
        _triangular_copy[DType.uint32](
            out_addr, in_addr, batch, rows, cols, diagonal, upper, ctx
        )
    elif _dtype_arg_width_on[0, 16]():
        if itemsize != 2:
            raise Error("triangular specialization/itemsize mismatch")
        _triangular_copy[DType.uint16](
            out_addr, in_addr, batch, rows, cols, diagonal, upper, ctx
        )
    elif _dtype_arg_width_on[0, 64]():
        if itemsize != 8:
            raise Error("triangular specialization/itemsize mismatch")
        _triangular_copy[DType.uint64](
            out_addr, in_addr, batch, rows, cols, diagonal, upper, ctx
        )
    elif _dtype_arg_width_on[0, 8]():
        if itemsize != 1:
            raise Error("triangular specialization/itemsize mismatch")
        _triangular_copy[DType.uint8](
            out_addr, in_addr, batch, rows, cols, diagonal, upper, ctx
        )
    else:
        raise Error("TriangularCopy: unsupported element size ", itemsize)


# ---------------------------------------------------------------------------
# GatherRows: out[i] = in[wrap(idx[i // row_len]) * row_len + i % row_len],
# a gather of whole rows along dim 0 of a contiguous input. row_len =
# prod(in_shape[1:]); negative indices wrap by adding size0 = in_shape[0].
# Implements the single-int-index-on-dim-0 case of aten::index.Tensor.
# Element-size dispatch for the payload, int32/int64 for the index.
# ---------------------------------------------------------------------------


@always_inline
def _gather_rows[
    dtype: DType, idx_dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    idx_addr: Int,
    n_indices: Int,
    row_len: Int,
    size0: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var idx_ptr = _make_ptr[idx_dtype](idx_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr, idx_ptr)
    def func[width: Int, alignment: Int = 1](coord: Coord):
        var i = Int(coord[0].value())
        var row = Int(idx_ptr[i // row_len])
        if row < 0:
            row += size0
        out_ptr[i] = in_ptr[row * row_len + i % row_len]

    _parallel_for[func](n_indices * row_len, ctx)


@always_inline
def _gather_rows_idx[
    idx_dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    idx_addr: Int,
    n_indices: Int,
    row_len: Int,
    size0: Int,
    itemsize: Int,
    ctx: DeviceContext,
) raises:
    comptime if _dtype_arg_width_on[0, 32]():
        if itemsize != 4:
            raise Error("gather specialization/itemsize mismatch")
        _gather_rows[DType.uint32, idx_dtype](
            out_addr, in_addr, idx_addr, n_indices, row_len, size0, ctx
        )
    elif _dtype_arg_width_on[0, 16]():
        if itemsize != 2:
            raise Error("gather specialization/itemsize mismatch")
        _gather_rows[DType.uint16, idx_dtype](
            out_addr, in_addr, idx_addr, n_indices, row_len, size0, ctx
        )
    elif _dtype_arg_width_on[0, 64]():
        if itemsize != 8:
            raise Error("gather specialization/itemsize mismatch")
        _gather_rows[DType.uint64, idx_dtype](
            out_addr, in_addr, idx_addr, n_indices, row_len, size0, ctx
        )
    elif _dtype_arg_width_on[0, 8]():
        if itemsize != 1:
            raise Error("gather specialization/itemsize mismatch")
        _gather_rows[DType.uint8, idx_dtype](
            out_addr, in_addr, idx_addr, n_indices, row_len, size0, ctx
        )
    else:
        raise Error("GatherRows: unsupported element size ", itemsize)


def _gather_rows_go(
    out_ptr: PyObjectPtr,
    in_ptr: PyObjectPtr,
    idx_ptr: PyObjectPtr,
    idx_dtype_o: PyObjectPtr,
    n_indices_o: PyObjectPtr,
    row_len_o: PyObjectPtr,
    size0_o: PyObjectPtr,
    itemsize_o: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr)
    var in_addr = _raw_int(in_ptr)
    var idx_addr = _raw_int(idx_ptr)
    var idx_dtype = _raw_dtype_int(idx_dtype_o)
    var n_indices = _raw_int(n_indices_o)
    var row_len = _raw_int(row_len_o)
    var size0 = _raw_int(size0_o)
    var itemsize = _raw_int(itemsize_o)
    var ctx = _raw_ctx(ctx_ptr)

    comptime if _dtype_arg_on[1, DType.int64]():
        if idx_dtype != DType.int64:
            raise Error("gather specialization/index dtype mismatch")
        _gather_rows_idx[DType.int64](
            out_addr,
            in_addr,
            idx_addr,
            n_indices,
            row_len,
            size0,
            itemsize,
            ctx,
        )
    elif _dtype_arg_on[1, DType.int32]():
        if idx_dtype != DType.int32:
            raise Error("gather specialization/index dtype mismatch")
        _gather_rows_idx[DType.int32](
            out_addr,
            in_addr,
            idx_addr,
            n_indices,
            row_len,
            size0,
            itemsize,
            ctx,
        )
    else:
        raise Error("GatherRows: unsupported index dtype ", idx_dtype)


# ---------------------------------------------------------------------------
# GatherDim: out[coord] = in[coord with coord[dim] := index[coord]], the read
# mirror of ScatterDim below, over a rank-<=4 index space described by
# explicit strides (padded to rank 4 with leading 0/1). Serves
#   * aten::gather      -- dims = index.shape, real index strides
#   * aten::index_select -- dims = out.shape, index strides 0 everywhere but
#     `dim` (a 1-D index broadcast across the untouched coordinates), which is
#     why one kernel covers both and no separate index_select body exists.
# Element-size dispatch for the payload (a pure copy), int32/int64 for the
# index. NEGATIVE INDICES ARE NOT WRAPPED and nothing is bounds checked, which
# is torch's own contract for these two ops: CPU raises before launching,
# CUDA's ScatterGatherKernel.cu only device-asserts, so a negative index is
# undefined there too. (aten::index.Tensor / aten::index_put_ do wrap; those
# route through GatherRows / a `remainder` on the index instead.)
# ---------------------------------------------------------------------------


@always_inline
def _gather_dim[
    dtype: DType, idx_dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    index_addr: Int,
    d0: Int,
    d1: Int,
    d2: Int,
    d3: Int,
    os0: Int,
    os1: Int,
    os2: Int,
    os3: Int,
    ss0: Int,
    ss1: Int,
    ss2: Int,
    ss3: Int,
    xs0: Int,
    xs1: Int,
    xs2: Int,
    xs3: Int,
    dim_padded: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var index_ptr = _make_ptr[idx_dtype](index_addr)
    var total = d0 * d1 * d2 * d3

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr, index_ptr)
    def func[width: Int, alignment: Int = 1](coord: Coord):
        var i = Int(coord[0].value())
        var i3 = i % d3
        var rest = i // d3
        var i2 = rest % d2
        rest = rest // d2
        var i1 = rest % d1
        var i0 = rest // d1
        var source = Int(index_ptr[i0 * xs0 + i1 * xs1 + i2 * xs2 + i3 * xs3])
        var in_off = i0 * ss0 + i1 * ss1 + i2 * ss2 + i3 * ss3
        # Replace the coordinate along `dim_padded` with the gather source.
        if dim_padded == 0:
            in_off += (source - i0) * ss0
        elif dim_padded == 1:
            in_off += (source - i1) * ss1
        elif dim_padded == 2:
            in_off += (source - i2) * ss2
        else:
            in_off += (source - i3) * ss3
        out_ptr[i0 * os0 + i1 * os1 + i2 * os2 + i3 * os3] = in_ptr[in_off]

    _parallel_for[func](total, ctx)


@always_inline
def _gather_dim_idx[
    idx_dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    index_addr: Int,
    d0: Int,
    d1: Int,
    d2: Int,
    d3: Int,
    os0: Int,
    os1: Int,
    os2: Int,
    os3: Int,
    ss0: Int,
    ss1: Int,
    ss2: Int,
    ss3: Int,
    xs0: Int,
    xs1: Int,
    xs2: Int,
    xs3: Int,
    dim_padded: Int,
    itemsize: Int,
    ctx: DeviceContext,
) raises:
    comptime if _dtype_arg_width_on[0, 32]():
        if itemsize != 4:
            raise Error("gather_dim specialization/itemsize mismatch")
        _gather_dim[DType.uint32, idx_dtype](
            out_addr,
            in_addr,
            index_addr,
            d0,
            d1,
            d2,
            d3,
            os0,
            os1,
            os2,
            os3,
            ss0,
            ss1,
            ss2,
            ss3,
            xs0,
            xs1,
            xs2,
            xs3,
            dim_padded,
            ctx,
        )
    elif _dtype_arg_width_on[0, 16]():
        if itemsize != 2:
            raise Error("gather_dim specialization/itemsize mismatch")
        _gather_dim[DType.uint16, idx_dtype](
            out_addr,
            in_addr,
            index_addr,
            d0,
            d1,
            d2,
            d3,
            os0,
            os1,
            os2,
            os3,
            ss0,
            ss1,
            ss2,
            ss3,
            xs0,
            xs1,
            xs2,
            xs3,
            dim_padded,
            ctx,
        )
    elif _dtype_arg_width_on[0, 64]():
        if itemsize != 8:
            raise Error("gather_dim specialization/itemsize mismatch")
        _gather_dim[DType.uint64, idx_dtype](
            out_addr,
            in_addr,
            index_addr,
            d0,
            d1,
            d2,
            d3,
            os0,
            os1,
            os2,
            os3,
            ss0,
            ss1,
            ss2,
            ss3,
            xs0,
            xs1,
            xs2,
            xs3,
            dim_padded,
            ctx,
        )
    elif _dtype_arg_width_on[0, 8]():
        if itemsize != 1:
            raise Error("gather_dim specialization/itemsize mismatch")
        _gather_dim[DType.uint8, idx_dtype](
            out_addr,
            in_addr,
            index_addr,
            d0,
            d1,
            d2,
            d3,
            os0,
            os1,
            os2,
            os3,
            ss0,
            ss1,
            ss2,
            ss3,
            xs0,
            xs1,
            xs2,
            xs3,
            dim_padded,
            ctx,
        )
    else:
        raise Error("GatherDim: unsupported element size ", itemsize)


def _gather_dim_go(
    out_ptr: PyObjectPtr,
    in_ptr: PyObjectPtr,
    index_ptr: PyObjectPtr,
    params: PyObjectPtr,  # (d0..d3, os0..os3, ss0..ss3, xs0..xs3, dim_padded)
    idx_dtype_o: PyObjectPtr,
    itemsize_o: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr)
    var in_addr = _raw_int(in_ptr)
    var index_addr = _raw_int(index_ptr)
    var d0 = _raw_tuple_int(params, 0)
    var d1 = _raw_tuple_int(params, 1)
    var d2 = _raw_tuple_int(params, 2)
    var d3 = _raw_tuple_int(params, 3)
    var os0 = _raw_tuple_int(params, 4)
    var os1 = _raw_tuple_int(params, 5)
    var os2 = _raw_tuple_int(params, 6)
    var os3 = _raw_tuple_int(params, 7)
    var ss0 = _raw_tuple_int(params, 8)
    var ss1 = _raw_tuple_int(params, 9)
    var ss2 = _raw_tuple_int(params, 10)
    var ss3 = _raw_tuple_int(params, 11)
    var xs0 = _raw_tuple_int(params, 12)
    var xs1 = _raw_tuple_int(params, 13)
    var xs2 = _raw_tuple_int(params, 14)
    var xs3 = _raw_tuple_int(params, 15)
    var dim_padded = _raw_tuple_int(params, 16)
    var idx_dtype = _raw_dtype_int(idx_dtype_o)
    var itemsize = _raw_int(itemsize_o)
    var ctx = _raw_ctx(ctx_ptr)

    comptime if _dtype_arg_on[1, DType.int64]():
        if idx_dtype != DType.int64:
            raise Error("gather_dim specialization/index dtype mismatch")
        _gather_dim_idx[DType.int64](
            out_addr,
            in_addr,
            index_addr,
            d0,
            d1,
            d2,
            d3,
            os0,
            os1,
            os2,
            os3,
            ss0,
            ss1,
            ss2,
            ss3,
            xs0,
            xs1,
            xs2,
            xs3,
            dim_padded,
            itemsize,
            ctx,
        )
    elif _dtype_arg_on[1, DType.int32]():
        if idx_dtype != DType.int32:
            raise Error("gather_dim specialization/index dtype mismatch")
        _gather_dim_idx[DType.int32](
            out_addr,
            in_addr,
            index_addr,
            d0,
            d1,
            d2,
            d3,
            os0,
            os1,
            os2,
            os3,
            ss0,
            ss1,
            ss2,
            ss3,
            xs0,
            xs1,
            xs2,
            xs3,
            dim_padded,
            itemsize,
            ctx,
        )
    else:
        raise Error("GatherDim: unsupported index dtype ", idx_dtype)


# ---------------------------------------------------------------------------
# ScatterDim: out[coord with coord[dim] := index[coord]] = src[coord] (or a
# scalar value). Implements aten::scatter.src / aten::scatter.value over a
# rank-<=4 index space; `out` is a contiguous clone of self, `index` is
# int64, and everything is described by explicit strides (padded to rank 4
# with leading 0). Match torch: no bounds checking, last-write-wins on
# duplicate targets. Dispatches on dtype (the scalar value is cast to it).
# ---------------------------------------------------------------------------


@always_inline
def _scatter_dim[
    dtype: DType
](
    out_addr: Int,
    index_addr: Int,
    src_addr: Int,
    d0: Int,
    d1: Int,
    d2: Int,
    d3: Int,
    os0: Int,
    os1: Int,
    os2: Int,
    os3: Int,
    ss0: Int,
    ss1: Int,
    ss2: Int,
    ss3: Int,
    xs0: Int,
    xs1: Int,
    xs2: Int,
    xs3: Int,
    dim_padded: Int,
    is_value: Int,
    value: Float64,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var index_ptr = _make_ptr[DType.int64](index_addr)
    var src_ptr = _make_ptr[dtype](src_addr)
    var scalar = value.cast[dtype]()
    var total = d0 * d1 * d2 * d3

    @always_inline
    @parameter
    @__copy_capture(out_ptr, index_ptr, src_ptr, scalar)
    def func[width: Int, alignment: Int = 1](coord: Coord):
        var i = Int(coord[0].value())
        var i3 = i % d3
        var rest = i // d3
        var i2 = rest % d2
        rest = rest // d2
        var i1 = rest % d1
        var i0 = rest // d1
        var target = Int(index_ptr[i0 * xs0 + i1 * xs1 + i2 * xs2 + i3 * xs3])
        var out_off = i0 * os0 + i1 * os1 + i2 * os2 + i3 * os3
        # Replace the coordinate along `dim_padded` with the scatter target.
        if dim_padded == 0:
            out_off += (target - i0) * os0
        elif dim_padded == 1:
            out_off += (target - i1) * os1
        elif dim_padded == 2:
            out_off += (target - i2) * os2
        else:
            out_off += (target - i3) * os3
        if is_value != 0:
            out_ptr[out_off] = scalar
        else:
            out_ptr[out_off] = src_ptr[
                i0 * ss0 + i1 * ss1 + i2 * ss2 + i3 * ss3
            ]

    _parallel_for_dt[dtype, func](total, ctx)


def _scatter_dim_go(
    out_ptr: PyObjectPtr,
    index_ptr: PyObjectPtr,
    src_ptr: PyObjectPtr,
    params: PyObjectPtr,  # (d0..d3, os0..os3, ss0..ss3, xs0..xs3, dim_padded)
    is_value_o: PyObjectPtr,
    value_o: PyObjectPtr,
    dtype_o: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr)
    var index_addr = _raw_int(index_ptr)
    var src_addr = _raw_int(src_ptr)
    var d0 = _raw_tuple_int(params, 0)
    var d1 = _raw_tuple_int(params, 1)
    var d2 = _raw_tuple_int(params, 2)
    var d3 = _raw_tuple_int(params, 3)
    var os0 = _raw_tuple_int(params, 4)
    var os1 = _raw_tuple_int(params, 5)
    var os2 = _raw_tuple_int(params, 6)
    var os3 = _raw_tuple_int(params, 7)
    var ss0 = _raw_tuple_int(params, 8)
    var ss1 = _raw_tuple_int(params, 9)
    var ss2 = _raw_tuple_int(params, 10)
    var ss3 = _raw_tuple_int(params, 11)
    var xs0 = _raw_tuple_int(params, 12)
    var xs1 = _raw_tuple_int(params, 13)
    var xs2 = _raw_tuple_int(params, 14)
    var xs3 = _raw_tuple_int(params, 15)
    var dim_padded = _raw_tuple_int(params, 16)
    var is_value = _raw_int(is_value_o)
    var value = _raw_f64(value_o)
    var dtype = _raw_dtype_int(dtype_o)
    var ctx = _raw_ctx(ctx_ptr)

    var handled = False
    comptime for dt in SCATTER_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _scatter_dim[dt](
                    out_addr,
                    index_addr,
                    src_addr,
                    d0,
                    d1,
                    d2,
                    d3,
                    os0,
                    os1,
                    os2,
                    os3,
                    ss0,
                    ss1,
                    ss2,
                    ss3,
                    xs0,
                    xs1,
                    xs2,
                    xs3,
                    dim_padded,
                    is_value,
                    value,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error("ScatterDim: unsupported dtype ", dtype)


# ---------------------------------------------------------------------------
# ScatterAddDim: out[coord with coord[dim] := index[coord]] += src[coord] over
# the same rank-<=4 index space and stride convention as ScatterDim above,
# reducing with a relaxed device-scope atomic add. Serves
#   * aten::scatter_add   -- real index strides
#   * aten::index_add     -- index strides 0 everywhere but `dim`
#   * aten::index_put(accumulate=True) -- dim 0, index strides (1, 0, ...)
# Which dtypes may be asked for is decided on the PYTHON side
# (`_SCATTER_ADD_DTYPES` in aten_fast.py): `Atomic.fetch_add` has no lowering
# for sub-32-bit integers on the GPU targets, so int8/int16/uint8/bool are
# declined before a build is requested rather than reaching the compiler as an
# unsatisfiable specialization.
#
# This is deliberately NOT ScatterDim parametrized on the reduction, which is
# what it looks like it should be. Every way of writing that -- a
# `comptime if` around the value/src branch, one inside each arm of it, an
# `@always_inline` store helper called from both -- moved the REDUCE=0
# assembly (2 to 4 of ScatterDim's 4 sm_90a kernels changed, +/-3 predicates
# and +/-25 b64 registers), while a bare unused comptime parameter on
# `_scatter_dim` changed nothing. Metaprogramming that rewrites the existing
# kernel's code is not a refactor, so the reduction lives here instead. The
# duplication is smaller than it looks: scatter_add has no scalar-value
# overload, so this body has no `is_value` branch and no `value` argument.
#
# Match torch: no bounds checking, and the accumulation order of colliding
# writes is unspecified -- torch's own CUDA scatter_add is atomic too
# (ScatterGatherKernel.cu's ReduceAdd calls fastAtomicAdd), so neither
# implementation is bitwise reproducible for float dtypes.
# `torch.use_deterministic_algorithms(True)` is not honored: torch switches to
# a sort-based reduction there and this does not.
#
# The deterministic alternative, if it is ever wanted: a scatter along `dim`
# can only collide between entries that agree on every coordinate EXCEPT
# `dim` (the index replaces that one coordinate and no other), so giving one
# thread a whole `dim`-column and accumulating it serially is race-free with
# no atomics at all, for every dtype and every backend. The catch is
# parallelism -- the thread count becomes total/dims[dim], which is 64 threads
# for a (65536, 64) index scattered along dim 0 -- so it would have to be a
# second kernel chosen by a runtime dispatch, not a replacement for this one.
# ---------------------------------------------------------------------------


@always_inline
def _atomic_scope() -> StaticString:
    # Device-scope atomics skip system-scope ordering cost on the discrete
    # targets; every other accelerator keeps the portable default scope.
    # (embedding_backward_kernels.mojo has an f32-only twin of this with an
    # extra sm_90 `red.v4.f32` vector path; this one is scalar and generic over
    # the dtype, so they are deliberately not shared.)
    comptime if is_nvidia_gpu():
        return "device"
    elif is_amd_gpu():
        return "agent"
    else:
        return ""


@always_inline
def _scatter_add_dim[
    dtype: DType
](
    out_addr: Int,
    index_addr: Int,
    src_addr: Int,
    d0: Int,
    d1: Int,
    d2: Int,
    d3: Int,
    os0: Int,
    os1: Int,
    os2: Int,
    os3: Int,
    ss0: Int,
    ss1: Int,
    ss2: Int,
    ss3: Int,
    xs0: Int,
    xs1: Int,
    xs2: Int,
    xs3: Int,
    dim_padded: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var index_ptr = _make_ptr[DType.int64](index_addr)
    var src_ptr = _make_ptr[dtype](src_addr)
    var total = d0 * d1 * d2 * d3

    @always_inline
    @parameter
    @__copy_capture(out_ptr, index_ptr, src_ptr)
    def func[width: Int, alignment: Int = 1](coord: Coord):
        var i = Int(coord[0].value())
        var i3 = i % d3
        var rest = i // d3
        var i2 = rest % d2
        rest = rest // d2
        var i1 = rest % d1
        var i0 = rest // d1
        var target = Int(index_ptr[i0 * xs0 + i1 * xs1 + i2 * xs2 + i3 * xs3])
        var out_off = i0 * os0 + i1 * os1 + i2 * os2 + i3 * os3
        # Replace the coordinate along `dim_padded` with the scatter target.
        if dim_padded == 0:
            out_off += (target - i0) * os0
        elif dim_padded == 1:
            out_off += (target - i1) * os1
        elif dim_padded == 2:
            out_off += (target - i2) * os2
        else:
            out_off += (target - i3) * os3
        # Relaxed ordering is enough: nothing else in the launch reads `out`.
        _ = Atomic[dtype, scope=_atomic_scope()].fetch_add[
            ordering=Ordering.RELAXED
        ](out_ptr + out_off, src_ptr[i0 * ss0 + i1 * ss1 + i2 * ss2 + i3 * ss3])

    _parallel_for_dt[dtype, func](total, ctx)


def _scatter_add_dim_go(
    out_ptr: PyObjectPtr,
    index_ptr: PyObjectPtr,
    src_ptr: PyObjectPtr,
    params: PyObjectPtr,  # (d0..d3, os0..os3, ss0..ss3, xs0..xs3, dim_padded)
    dtype_o: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr)
    var index_addr = _raw_int(index_ptr)
    var src_addr = _raw_int(src_ptr)
    var d0 = _raw_tuple_int(params, 0)
    var d1 = _raw_tuple_int(params, 1)
    var d2 = _raw_tuple_int(params, 2)
    var d3 = _raw_tuple_int(params, 3)
    var os0 = _raw_tuple_int(params, 4)
    var os1 = _raw_tuple_int(params, 5)
    var os2 = _raw_tuple_int(params, 6)
    var os3 = _raw_tuple_int(params, 7)
    var ss0 = _raw_tuple_int(params, 8)
    var ss1 = _raw_tuple_int(params, 9)
    var ss2 = _raw_tuple_int(params, 10)
    var ss3 = _raw_tuple_int(params, 11)
    var xs0 = _raw_tuple_int(params, 12)
    var xs1 = _raw_tuple_int(params, 13)
    var xs2 = _raw_tuple_int(params, 14)
    var xs3 = _raw_tuple_int(params, 15)
    var dim_padded = _raw_tuple_int(params, 16)
    var dtype = _raw_dtype_int(dtype_o)
    var ctx = _raw_ctx(ctx_ptr)

    var handled = False
    comptime for dt in SCATTER_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _scatter_add_dim[dt](
                    out_addr,
                    index_addr,
                    src_addr,
                    d0,
                    d1,
                    d2,
                    d3,
                    os0,
                    os1,
                    os2,
                    os3,
                    ss0,
                    ss1,
                    ss2,
                    ss3,
                    xs0,
                    xs1,
                    xs2,
                    xs3,
                    dim_padded,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error("ScatterAddDim: unsupported dtype ", dtype)


# ---------------------------------------------------------------------------
# METH_FASTCALL wrappers: raw CPython argument unpacking (no owning
# PythonObject per argument). Argument types are guaranteed by the internal
# Python callers; raise sites are unsupported-dtype guards gated upstream.
# ---------------------------------------------------------------------------


def _permute_copy_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _permute_copy_go(args[0], args[1], args[2], args[3], args[4], args[5])
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


def _narrow_copy_dst_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _narrow_copy_dst_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
        )
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


def _where_select_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _where_select_go(
            args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        )
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


def _masked_fill_scalar_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _masked_fill_scalar_go(
            args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        )
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


def _tile_copy_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _tile_copy_go(
            args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        )
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


def _repeat_tiled_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _repeat_tiled_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
        )
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


def _triangular_copy_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _triangular_copy_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
            args[8],
        )
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


def _gather_rows_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _gather_rows_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
            args[8],
        )
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


def _gather_dim_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _gather_dim_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
        )
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


def _scatter_dim_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _scatter_dim_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
        )
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


def _scatter_add_dim_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _scatter_add_dim_go(
            args[0], args[1], args[2], args[3], args[4], args[5]
        )
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


def _cast_spec_into_go(
    a_o: PyObjectPtr, out_dtype_o: PyObjectPtr, out_o: PyObjectPtr
) raises:
    """Cast into a caller-allocated contiguous output."""
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    var dst = _raw_dtype_int(out_dtype_o)
    var src_ok = False
    comptime for dt in CAST_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if a.dtype == dt:
                src_ok = True
    var dst_ok = False
    comptime for dt in CAST_DTYPES:
        comptime if _dtype_out_on[0, dt]():
            if dst == dt:
                dst_ok = True
    if not (src_ok and dst_ok):
        raise Error("mojo spec cast into: unsupported dtype pair")
    if out.dtype != dst or out.numel != a.numel or not out.contig:
        raise Error("mojo spec cast into: output buffer mismatch")
    if out.ctx_ptr != a.ctx_ptr:
        raise Error("mojo spec cast into: output on a different device")

    var ctx = a.ctx()
    var addr = out.ptr
    if a.numel > 0:
        if a.contig:
            comptime for src_dt in CAST_DTYPES:
                comptime if _dtype_arg_on[0, src_dt]():
                    if a.dtype == src_dt:
                        _cast_to[src_dt](dst, addr, a.ptr, a.numel, ctx)
        else:
            var tmp = _scratch_contig(a, ctx)
            var tmp_addr = Int(tmp.unsafe_ptr())
            comptime for src_dt in CAST_DTYPES:
                comptime if _dtype_arg_on[0, src_dt]():
                    if a.dtype == src_dt:
                        _cast_to[src_dt](dst, addr, tmp_addr, a.numel, ctx)
            _ = tmp^


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_data_movement_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("data_movement_ops")
        comptime if _op_on["CastSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[_cast_spec_into_go, "CastSpec"],
                docstring="(a_spec, out_dtype, out_spec)",
            )
        comptime if _op_on["PermuteCopy"]():
            _register_call(
                b,
                _permute_copy_dispatcher,
                docstring=(
                    "materialize a permutation of a contiguous tensor (rank"
                    " <= 4)"
                ),
            )
        comptime if _op_on["NarrowCopyDst"]():
            _register_call(
                b,
                _narrow_copy_dst_dispatcher,
                docstring=(
                    "copy `outer` contiguous blocks of `copy_len` elements to a"
                    " destination stride/offset (concatenation)"
                ),
            )
        comptime if _op_on["CatN"]():
            _register_call(
                b,
                _cat_n_dispatcher,
                docstring=(
                    "(out, src_addrs, row_lens, outer, dst_stride, itemsize,"
                    " width, ctx); batched N-input concat rows, one launch per"
                    " CAT_SEG_CAP inputs"
                ),
            )
        comptime if _op_on["WhereSelect"]():
            _register_call(
                b,
                _where_select_dispatcher,
                docstring="out = cond ? a : b (broadcast strides, any dtype)",
            )
        comptime if _op_on["MaskedFillScalar"]():
            _register_call(
                b,
                _masked_fill_scalar_dispatcher,
                docstring=(
                    "out = cond ? value : b (value baked into the launch, no"
                    " device buffer; masked_fill(_).Scalar fast path, float"
                    " dtypes only)"
                ),
            )
        comptime if _op_on["TileCopy"]():
            _register_call(
                b,
                _tile_copy_dispatcher,
                docstring=(
                    "out[coords] = in[coords % in_shape] over a rank-8-padded"
                    " index space (aten::repeat; element-size dispatch)"
                ),
            )
        comptime if _op_on["RepeatTiled"]():
            _register_call(
                b,
                _repeat_tiled_dispatcher,
                docstring=(
                    "out[copy * rows + ir, s * cols + c] = in[ir, c]"
                    " (aten::repeat reduced to a rank-2 tiled copy;"
                    " element-size dispatch)"
                ),
            )
        comptime if _op_on["TriangularCopy"]():
            _register_call(
                b,
                _triangular_copy_dispatcher,
                docstring=(
                    "out = in on the kept side of the diagonal, else 0"
                    " (aten::tril/triu; element-size dispatch)"
                ),
            )
        comptime if _op_on["GatherRows"]():
            _register_call(
                b,
                _gather_rows_dispatcher,
                docstring=(
                    "out[i] = in[wrap(idx[i // row_len]) * row_len + i %"
                    " row_len] (gather rows along dim 0; element-size +"
                    " int32/int64 idx)"
                ),
            )
        comptime if _op_on["GatherDim"]():
            _register_call(
                b,
                _gather_dim_dispatcher,
                docstring=(
                    "out[coord] = in[coord with dim := index[coord]]"
                    " (aten::gather / aten::index_select, rank <= 4;"
                    " element-size + int32/int64 idx)"
                ),
            )
        comptime if _op_on["ScatterDim"]():
            _register_call(
                b,
                _scatter_dim_dispatcher,
                docstring=(
                    "out[coord with dim := index[coord]] = src[coord] or value"
                    " (aten::scatter.src/value, rank <= 4; dtype dispatch)"
                ),
            )
        comptime if _op_on["ScatterAddDim"]():
            _register_call(
                b,
                _scatter_add_dim_dispatcher,
                docstring=(
                    "out[coord with dim := index[coord]] += src[coord] with"
                    " relaxed atomics (aten::scatter_add / index_add /"
                    " index_put accumulate, rank <= 4; dtype dispatch)"
                ),
            )
        return b.finalize()
    except e:
        abort(t"failed to create data_movement_ops python module: {e}")
