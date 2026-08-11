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

from std.os import abort
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator, has_apple_gpu_accelerator, size_of
from std.utils.coord import Coord

from max.algorithm import elementwise
from std.utils import IndexList

from std.python._cpython import PyObjectPtr, Py_ssize_t

from op_utils import (
    GS_THREADS,
    MAX_RANK,
    _LLC_BYTES,
    _bw_blocks,
    _MAX_GRID_Y,
    _T2D_ROWS,
    _enqueue_cached,
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
# Fused two-input concatenation: out rows are `len1 + len2` contiguous
# elements, filled from in1's and in2's contiguous rows in one launch (the
# common cat([a, b], dim) case after outer/inner flattening). One grid row
# per outer block; threads grid-stride the output row's float4 chunks and
# pick their source by column, so the append never pays two kernel
# launches and a pipeline bubble between them.
# ---------------------------------------------------------------------------


def _cat2_kernel2d[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in1_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    in2_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    len1_4_arg: Int64,
    len2_4_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var len1_4 = Int(len1_4_arg)
    var len2_4 = Int(len2_4_arg)
    comptime vec_align = 4 * size_of[dtype]()
    var o = Int(block_idx.y)
    var total4 = len1_4 + len2_4
    var out_base = o * total4 * 4
    var in1_base = o * len1_4 * 4
    var in2_base = o * len2_4 * 4
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var cstride = Int(grid_dim.x) * Int(block_dim.x)
    while c < total4:
        var j = c * 4
        if c < len1_4:
            out_ptr.store[width=4, alignment=vec_align](
                out_base + j,
                in1_ptr.load[width=4, alignment=vec_align](in1_base + j),
            )
        else:
            var j2 = (c - len1_4) * 4
            out_ptr.store[width=4, alignment=vec_align](
                out_base + j,
                in2_ptr.load[width=4, alignment=vec_align](in2_base + j2),
            )
        c += cstride


def _cat2_go(
    out_ptr_o: PyObjectPtr,
    in1_ptr_o: PyObjectPtr,
    in2_ptr_o: PyObjectPtr,
    outer_o: PyObjectPtr,
    len1_o: PyObjectPtr,
    len2_o: PyObjectPtr,
    itemsize_o: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr_o)
    var in1_addr = _raw_int(in1_ptr_o)
    var in2_addr = _raw_int(in2_ptr_o)
    var outer = _raw_int(outer_o)
    var len1 = _raw_int(len1_o)
    var len2 = _raw_int(len2_o)
    var itemsize = _raw_int(itemsize_o)
    var ctx = _raw_ctx(ctx_ptr)

    if ctx.api() == "cpu" or len1 % 4 != 0 or len2 % 4 != 0 or outer > 65535:
        raise Error("cat2 fast path preconditions not met")
    comptime if has_accelerator():
        var total4 = (len1 + len2) // 4
        var gx = max(1, min((total4 + GS_THREADS - 1) // GS_THREADS, 32))
        comptime if _dtype_arg_width_on[0, 32]():
            if itemsize != 4:
                raise Error("cat2 specialization/itemsize mismatch")
            _enqueue_cached[_cat2_kernel2d[DType.uint32]](
                ctx,
                String("dm_cat2_u32"),
                gx,
                outer,
                1,
                GS_THREADS,
                _make_ptr[DType.uint32](out_addr).as_unsafe_any_origin(),
                _make_ptr[DType.uint32](in1_addr)
                .as_unsafe_any_origin()
                .as_immutable(),
                _make_ptr[DType.uint32](in2_addr)
                .as_unsafe_any_origin()
                .as_immutable(),
                Int64(len1 // 4),
                Int64(len2 // 4),
            )
        elif _dtype_arg_width_on[0, 16]():
            if itemsize != 2:
                raise Error("cat2 specialization/itemsize mismatch")
            _enqueue_cached[_cat2_kernel2d[DType.uint16]](
                ctx,
                String("dm_cat2_u16"),
                gx,
                outer,
                1,
                GS_THREADS,
                _make_ptr[DType.uint16](out_addr).as_unsafe_any_origin(),
                _make_ptr[DType.uint16](in1_addr)
                .as_unsafe_any_origin()
                .as_immutable(),
                _make_ptr[DType.uint16](in2_addr)
                .as_unsafe_any_origin()
                .as_immutable(),
                Int64(len1 // 4),
                Int64(len2 // 4),
            )
        elif _dtype_arg_width_on[0, 64]():
            if itemsize != 8:
                raise Error("cat2 specialization/itemsize mismatch")
            _enqueue_cached[_cat2_kernel2d[DType.uint64]](
                ctx,
                String("dm_cat2_u64"),
                gx,
                outer,
                1,
                GS_THREADS,
                _make_ptr[DType.uint64](out_addr).as_unsafe_any_origin(),
                _make_ptr[DType.uint64](in1_addr)
                .as_unsafe_any_origin()
                .as_immutable(),
                _make_ptr[DType.uint64](in2_addr)
                .as_unsafe_any_origin()
                .as_immutable(),
                Int64(len1 // 4),
                Int64(len2 // 4),
            )
        elif _dtype_arg_width_on[0, 8]():
            if itemsize != 1:
                raise Error("cat2 specialization/itemsize mismatch")
            _enqueue_cached[_cat2_kernel2d[DType.uint8]](
                ctx,
                String("dm_cat2_u8"),
                gx,
                outer,
                1,
                GS_THREADS,
                _make_ptr[DType.uint8](out_addr).as_unsafe_any_origin(),
                _make_ptr[DType.uint8](in1_addr)
                .as_unsafe_any_origin()
                .as_immutable(),
                _make_ptr[DType.uint8](in2_addr)
                .as_unsafe_any_origin()
                .as_immutable(),
                Int64(len1 // 4),
                Int64(len2 // 4),
            )
        else:
            raise Error("unsupported element size for cat2")
    else:
        raise Error("no GPU accelerator available at compile time")


def _cat2_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _cat2_go(
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

    if copy_len % 4 == 0 and dst_stride % 4 == 0 and dst_offset % 4 == 0:
        comptime vec_align = 4 * size_of[dtype]()

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func4[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value()) * 4
            var o = i // copy_len
            var j = i % copy_len
            var v = in_ptr.load[width=4, alignment=vec_align](i)
            out_ptr.store[width=4, alignment=vec_align](
                o * dst_stride + dst_offset + j, v
            )

        elementwise[func4, simd_width=1](Coord(outer * copy_len // 4), ctx)
        return

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
# ---------------------------------------------------------------------------


@always_inline
def _where_select[
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
    var out_ptr = _make_ptr[dtype](out_addr)
    var cond_ptr = _make_ptr[DType.bool](cond_addr)
    var a_ptr = _make_ptr[dtype](a_addr)
    var b_ptr = _make_ptr[dtype](b_addr)

    # Chunk-of-4 kernel: one thread selects 4 consecutive output elements,
    # so the div/mod coordinate chain (the cost that dominates the scalar
    # version — 6 integer divisions per element) runs once per chunk, and
    # last-dim strides of 1/0 use vector loads / splats.
    var vec_ok = (
        (cs3 == 0 or cs3 == 1)
        and (a_s3 == 0 or a_s3 == 1)
        and (b_s3 == 0 or b_s3 == 1)
        and d3 >= 4
    )

    @always_inline
    @parameter
    @__copy_capture(out_ptr, cond_ptr, a_ptr, b_ptr)
    def func4[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value()) * 4
        var i3 = i % d3
        var rest = i // d3
        var i2 = rest % d2
        rest = rest // d2
        var i1 = rest % d1
        var i0 = rest // d1
        if i3 + 4 <= d3 and i + 4 <= total:
            var cbase = i0 * cs0 + i1 * cs1 + i2 * cs2 + i3 * cs3
            var abase = i0 * a_s0 + i1 * a_s1 + i2 * a_s2 + i3 * a_s3
            var bbase = i0 * b_s0 + i1 * b_s1 + i2 * b_s2 + i3 * b_s3
            var cond: SIMD[DType.bool, 4]
            if cs3 == 1:
                cond = cond_ptr.load[width=4](cbase)
            else:
                cond = SIMD[DType.bool, 4](cond_ptr[cbase])
            var av: SIMD[dtype, 4]
            if a_s3 == 1:
                av = a_ptr.load[width=4](abase)
            else:
                av = SIMD[dtype, 4](a_ptr[abase])
            var bv: SIMD[dtype, 4]
            if b_s3 == 1:
                bv = b_ptr.load[width=4](bbase)
            else:
                bv = SIMD[dtype, 4](b_ptr[bbase])
            out_ptr.store(i, cond.select(av, bv))
        else:
            # Chunk crosses a row boundary (or the end): per-element math.
            for u in range(4):
                var j = i + u
                if j >= total:
                    return
                var j3 = j % d3
                var jrest = j // d3
                var j2 = jrest % d2
                jrest = jrest // d2
                var j1 = jrest % d1
                var j0 = jrest // d1
                if cond_ptr[j0 * cs0 + j1 * cs1 + j2 * cs2 + j3 * cs3]:
                    out_ptr[j] = a_ptr[
                        j0 * a_s0 + j1 * a_s1 + j2 * a_s2 + j3 * a_s3
                    ]
                else:
                    out_ptr[j] = b_ptr[
                        j0 * b_s0 + j1 * b_s1 + j2 * b_s2 + j3 * b_s3
                    ]

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
        if cond_ptr[i0 * cs0 + i1 * cs1 + i2 * cs2 + i3 * cs3]:
            out_ptr[i] = a_ptr[i0 * a_s0 + i1 * a_s1 + i2 * a_s2 + i3 * a_s3]
        else:
            out_ptr[i] = b_ptr[i0 * b_s0 + i1 * b_s1 + i2 * b_s2 + i3 * b_s3]

    if vec_ok:
        _parallel_for[func4]((total + 3) // 4, ctx)
    else:
        _parallel_for[func](total, ctx)


@always_inline
def _dtype_size(dtype: DType) raises -> Int:
    """Element size in bytes for WhereSelect's `dtype` arg.

    `_where_select` is a pure bit-move (SIMD select, no arithmetic), so it
    only needs to be specialized per byte-size, not per exact dtype.
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
        _where_select[DType.uint32](
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
        _where_select[DType.uint16](
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
        _where_select[DType.uint64](
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
        _where_select[DType.uint8](
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


# ---------------------------------------------------------------------------
# Fused three-input concatenation: the _cat2_kernel2d pattern extended to
# three contiguous sources (the split-backward reassembly cat), so the
# append pays one launch instead of three.
# ---------------------------------------------------------------------------


def _cat3_kernel2d[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in1_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    in2_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    in3_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    len1_4_arg: Int64,
    len2_4_arg: Int64,
    len3_4_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var len1_4 = Int(len1_4_arg)
    var len2_4 = Int(len2_4_arg)
    var len3_4 = Int(len3_4_arg)
    comptime vec_align = 4 * size_of[dtype]()
    var o = Int(block_idx.y)
    var total4 = len1_4 + len2_4 + len3_4
    var out_base = o * total4 * 4
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var cstride = Int(grid_dim.x) * Int(block_dim.x)
    while c < total4:
        var v: SIMD[dtype, 4]
        if c < len1_4:
            v = in1_ptr.load[width=4, alignment=vec_align]((o * len1_4 + c) * 4)
        elif c < len1_4 + len2_4:
            v = in2_ptr.load[width=4, alignment=vec_align](
                (o * len2_4 + c - len1_4) * 4
            )
        else:
            v = in3_ptr.load[width=4, alignment=vec_align](
                (o * len3_4 + c - len1_4 - len2_4) * 4
            )
        out_ptr.store[width=4, alignment=vec_align](out_base + c * 4, v)
        c += cstride


def _cat3_segloop_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in1_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    in2_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    in3_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    len1_4_arg: Int64,
    len2_4_arg: Int64,
    len3_4_arg: Int64,
    outer_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var len1_4 = Int(len1_4_arg)
    var len2_4 = Int(len2_4_arg)
    var len3_4 = Int(len3_4_arg)
    var outer = Int(outer_arg)
    # One thread per (row, source) segment: each thread streams its
    # segment's vectors sequentially (the access pattern this GPU streams
    # at full rate), with the source chosen once per thread.  Threads with
    # the same source are consecutive, so the branch is warp-uniform for
    # outer >= warp size.
    comptime vec_align = 4 * size_of[dtype]()
    var total4 = len1_4 + len2_4 + len3_4
    var nsegs = outer * 3
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while t < nsegs:
        var o = t % outer
        var s = t // outer
        var seg_len: Int
        var dst_off: Int
        if s == 0:
            seg_len = len1_4
            dst_off = 0
        elif s == 1:
            seg_len = len2_4
            dst_off = len1_4
        else:
            seg_len = len3_4
            dst_off = len1_4 + len2_4
        var sbase = o * seg_len * 4
        var dbase = (o * total4 + dst_off) * 4
        for j in range(seg_len):
            var v: SIMD[dtype, 4]
            if s == 0:
                v = in1_ptr.load[width=4, alignment=vec_align](sbase + j * 4)
            elif s == 1:
                v = in2_ptr.load[width=4, alignment=vec_align](sbase + j * 4)
            else:
                v = in3_ptr.load[width=4, alignment=vec_align](sbase + j * 4)
            out_ptr.store[width=4, alignment=vec_align](dbase + j * 4, v)
        t += gstride


def _cat3_go(
    out_ptr_o: PyObjectPtr,
    in1_ptr_o: PyObjectPtr,
    in2_ptr_o: PyObjectPtr,
    in3_ptr_o: PyObjectPtr,
    outer_o: PyObjectPtr,
    len1_o: PyObjectPtr,
    len2_o: PyObjectPtr,
    len3_o: PyObjectPtr,
    itemsize_o: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr_o)
    var in1_addr = _raw_int(in1_ptr_o)
    var in2_addr = _raw_int(in2_ptr_o)
    var in3_addr = _raw_int(in3_ptr_o)
    var outer = _raw_int(outer_o)
    var len1 = _raw_int(len1_o)
    var len2 = _raw_int(len2_o)
    var len3 = _raw_int(len3_o)
    var itemsize = _raw_int(itemsize_o)
    var ctx = _raw_ctx(ctx_ptr)

    if (
        ctx.api() == "cpu"
        or len1 % 4 != 0
        or len2 % 4 != 0
        or len3 % 4 != 0
        or outer > 65535
    ):
        raise Error("cat3 fast path preconditions not met")
    comptime if has_accelerator():
        var total4 = (len1 + len2 + len3) // 4
        var gx = max(1, min((total4 + GS_THREADS - 1) // GS_THREADS, 32))
        var nsegs = outer * 3
        var handled = False
        comptime for dt in [
            DType.uint32,
            DType.uint16,
            DType.uint64,
            DType.uint8,
        ]:
            comptime if (
                (dt == DType.uint32 and _dtype_arg_width_on[0, 32]())
                or (dt == DType.uint16 and _dtype_arg_width_on[0, 16]())
                or (dt == DType.uint64 and _dtype_arg_width_on[0, 64]())
                or (dt == DType.uint8 and _dtype_arg_width_on[0, 8]())
            ):
                if itemsize != size_of[dt]():
                    raise Error("cat3 specialization/itemsize mismatch")
                if nsegs >= 2048:
                    # Enough segments to fill the machine with one
                    # sequential stream per thread.
                    _enqueue_cached[_cat3_segloop_kernel[dt]](
                        ctx,
                        String(t"dm_cat3_seg_{dt}"),
                        _gs_blocks(nsegs),
                        1,
                        1,
                        GS_THREADS,
                        _make_ptr[dt](out_addr).as_unsafe_any_origin(),
                        _make_ptr[dt](in1_addr)
                        .as_unsafe_any_origin()
                        .as_immutable(),
                        _make_ptr[dt](in2_addr)
                        .as_unsafe_any_origin()
                        .as_immutable(),
                        _make_ptr[dt](in3_addr)
                        .as_unsafe_any_origin()
                        .as_immutable(),
                        Int64(len1 // 4),
                        Int64(len2 // 4),
                        Int64(len3 // 4),
                        Int64(outer),
                    )
                else:
                    _enqueue_cached[_cat3_kernel2d[dt]](
                        ctx,
                        String(t"dm_cat3_{dt}"),
                        gx,
                        outer,
                        1,
                        GS_THREADS,
                        _make_ptr[dt](out_addr).as_unsafe_any_origin(),
                        _make_ptr[dt](in1_addr)
                        .as_unsafe_any_origin()
                        .as_immutable(),
                        _make_ptr[dt](in2_addr)
                        .as_unsafe_any_origin()
                        .as_immutable(),
                        _make_ptr[dt](in3_addr)
                        .as_unsafe_any_origin()
                        .as_immutable(),
                        Int64(len1 // 4),
                        Int64(len2 // 4),
                        Int64(len3 // 4),
                    )
                handled = True
        if not handled:
            raise Error("unsupported element size for cat3")
    else:
        raise Error("no GPU accelerator available at compile time")


def _cat3_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _cat3_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
            args[8],
            args[9],
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
        comptime if _op_on["Cat2"]():
            _register_call(
                b,
                _cat2_dispatcher,
                docstring=(
                    "(out, in1, in2, outer, len1, len2, itemsize, ctx); fused"
                    " two-input concat rows"
                ),
            )
        comptime if _op_on["WhereSelect"]():
            _register_call(
                b,
                _where_select_dispatcher,
                docstring="out = cond ? a : b (broadcast strides, any dtype)",
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
        comptime if _op_on["ScatterDim"]():
            _register_call(
                b,
                _scatter_dim_dispatcher,
                docstring=(
                    "out[coord with dim := index[coord]] = src[coord] or value"
                    " (aten::scatter.src/value, rank <= 4; dtype dispatch)"
                ),
            )
        comptime if _op_on["Cat3"]():
            _register_call(
                b,
                _cat3_dispatcher,
                docstring=(
                    "(out, in1, in2, in3, outer, len1, len2, len3, itemsize,"
                    " ctx); fused three-input concat rows"
                ),
            )
        return b.finalize()
    except e:
        abort(t"failed to create data_movement_ops python module: {e}")
