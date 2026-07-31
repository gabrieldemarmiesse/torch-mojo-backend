# ===----------------------------------------------------------------------=== #
# Shared helpers for the fast eager-mode kernel modules.
#
# Mirrors `max._interpreter_ops.op_utils`: unwrap `max.driver.Buffer` Python
# objects into raw typed pointers, and rebuild the MAX DeviceContext from the
# pointer that `device._device_context_ptr()` hands us on the Python side.
# ===----------------------------------------------------------------------=== #

from std.algorithm.functional import elementwise
from std.builtin.device_passable import DevicePassable
from std.ffi import _get_global_or_null, external_call
from std.gpu import barrier, block_dim, block_idx, grid_dim, thread_idx
from std.gpu.host import DeviceBuffer, DeviceContext
from std.math import ceildiv
from std.memory import OpaquePointer, alloc, stack_allocation
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.sys.info import has_accelerator, has_apple_gpu_accelerator, size_of
from std.utils import IndexList
from std.utils.coord import Coord


# The floating-point dtypes the fast kernels specialize for. Dispatchers loop
# over this at compile time (`comptime for dt in FLOAT_DTYPES`) to pick the
# runtime dtype, which unrolls into the same `if dtype == ...` chain without
# repeating the call site once per dtype.
comptime FLOAT_DTYPES = [DType.float32, DType.float16, DType.bfloat16]


@always_inline
def _enqueue_cached[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(* args: * declared_arg_types) thin -> None,
    *Ts: DevicePassable,
](
    ctx: DeviceContext,
    key: String,
    gx: Int,
    gy: Int,
    gz: Int,
    threads: Int,
    *args: *Ts,
) raises:
    """Enqueue `func`, compiling it at most once per process and context.

    `ctx.enqueue_function[func]` re-runs `compile_function` on every call
    (~180µs even when the runtime's module cache hits); caching the
    `DeviceFunction` in the process-global registry — the same pattern the
    vendor BLAS handle uses — brings the enqueue cost down to a few µs.
    """
    var name = String(t"TMB_KERNEL_{key}_{ctx.id()}")
    comptime FuncT = type_of(ctx.compile_function[func]())

    if global_ptr := _get_global_or_null(name):
        var fptr = global_ptr.value().bitcast[FuncT]()
        ctx.enqueue_function(
            fptr[], *args, grid_dim=(gx, gy, gz), block_dim=(threads,)
        )
        return

    var compiled = ctx.compile_function[func]()
    var fptr = alloc[FuncT](1)
    fptr.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name),
        fptr.bitcast[NoneType](),
    )
    ctx.enqueue_function(
        fptr[], *args, grid_dim=(gx, gy, gz), block_dim=(threads,)
    )


@always_inline
def _enqueue_cached_2d[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(* args: * declared_arg_types) thin -> None,
    *Ts: DevicePassable,
](
    ctx: DeviceContext,
    key: String,
    gx: Int,
    gy: Int,
    gz: Int,
    threads_x: Int,
    threads_y: Int,
    *args: *Ts,
) raises:
    """Cached enqueue for kernels with a two-dimensional thread block."""
    var name = String(t"TMB_KERNEL_2D_{key}_{ctx.id()}")
    comptime FuncT = type_of(ctx.compile_function[func]())

    if global_ptr := _get_global_or_null(name):
        var fptr = global_ptr.value().bitcast[FuncT]()
        ctx.enqueue_function(
            fptr[],
            *args,
            grid_dim=(gx, gy, gz),
            block_dim=(threads_x, threads_y),
        )
        return

    var compiled = ctx.compile_function[func]()
    var fptr = alloc[FuncT](1)
    fptr.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name),
        fptr.bitcast[NoneType](),
    )
    ctx.enqueue_function(
        fptr[],
        *args,
        grid_dim=(gx, gy, gz),
        block_dim=(threads_x, threads_y),
    )


# Launch geometry for the cached grid-stride kernels that replace stdlib
# `elementwise` on GPU: `ctx.enqueue_function` costs ~20us per call on Metal
# but `elementwise` pays ~42us (it rebuilds and re-resolves its closure
# kernel every call), so the hot eager ops launch pre-compiled thin kernels
# through `_enqueue_cached` instead.
comptime GS_THREADS = 256


@always_inline
def _gs_blocks(total: Int) -> Int:
    """Grid size for a GS_THREADS-wide grid-stride launch."""
    return max(1, min((total + GS_THREADS - 1) // GS_THREADS, 4096))


# A bandwidth-bound kernel has two regimes and they want different grids, which
# `_gs_blocks`'s flat 4096-block cap gets wrong at both ends. Measured on the
# dtype cast (`_cast_vec_kernel`, one 16-byte slot per thread), TB/s of
# read+write traffic, median of 40 x 20 launches on gfx942:
#
#   traffic    grid = ceildiv(slots, 256)   4096 blocks   2 blocks per CU
#     14 MiB           2.84                    2.84            2.94
#     54 MiB           3.30                    3.30            3.60
#    216 MiB           4.00                    3.85            4.55
#    288 MiB           4.20                    4.19            4.34
#    384 MiB           3.97                    3.81            3.81
#    576 MiB           4.03                    3.74            3.76
#   14.8 GiB           4.06                    3.71            3.71
#
# The crossover sits just above the 256 MiB of Infinity Cache this part has,
# which is the reading: while the operands are cache-resident a few long-lived
# workgroups per CU beat many short ones, and once the copy streams from HBM
# the grid that covers the slots exactly wins. Both arms are the same kernel --
# the grid-stride loop just iterates more times in the resident arm.
#
# `resident` stays the CALLER's decision, and every caller has to measure it,
# because the block count that wins in that arm depends on how many bytes one
# thread moves. Two counter-examples already measured: the cast's unaligned
# fallback, whose threads make four 4-byte accesses instead of one 16-byte one,
# costs 98 us against 54 us in the resident arm; and the three-operand binary
# add, at 48 bytes a thread instead of 24, loses 3.6% in it (3.94 against 4.09
# TB/s at 216 MiB) even though the cast gains 14%.
comptime _LLC_BYTES = 256 * 1024 * 1024

# Only reached above ~1e9 slots; past it the caller's grid-stride loop iterates.
comptime _BW_MAX_BLOCKS = 1 << 22


@always_inline
def _bw_blocks(
    slots: Int, slots_per_thread: Int, resident: Bool, ctx: DeviceContext
) -> Int:
    """Grid for a GS_THREADS-wide, bandwidth-bound grid-stride launch.

    `slots` is the number of vector slots to cover and `slots_per_thread` how
    many of them one thread takes per grid pass (1 when a thread's single
    access is already 16 bytes wide). `resident` asks for the cache-resident
    arm; see above for why that is not something this helper can decide.
    """
    var per_block = GS_THREADS * slots_per_thread
    var blocks = (slots + per_block - 1) // per_block
    comptime SMALL_GRID = max(2 * ctx.default_device_info.sm_count, GS_THREADS)
    if resident:
        blocks = min(blocks, SMALL_GRID)
    return max(1, min(blocks, _BW_MAX_BLOCKS))


@always_inline
def _bw_flat_blocks(slots: Int, traffic_bytes: Int) -> Int:
    """Grid for a GS_THREADS-wide launch making one 16-byte access per operand.

    The three-operand binary kernel's crossover is not the cast's -- its threads
    move 48 bytes a slot, not 24 -- so it gets its own rule rather than
    `_bw_blocks`. Measured on gfx942, median of 40 x 20 launches:

      traffic   4096 blocks   exact grid = ceildiv(slots, 256)
      216 MiB   4.09 TB/s     3.81 TB/s     (bf16 [48, 1024, 768] add)
      432 MiB   121.7 us      114.0 us      (fp32 [48, 1024, 768] add)

    Keep the 4096-block cap while the operands are cache-resident; cover the
    slots exactly once they stream from HBM.
    """
    if traffic_bytes > _LLC_BYTES:
        return max(
            1, min((slots + GS_THREADS - 1) // GS_THREADS, _BW_MAX_BLOCKS)
        )
    return _gs_blocks(slots)


@always_inline
def _make_ptr[
    dtype: DType
](addr: Int) -> UnsafePointer[Scalar[dtype], MutUntrackedOrigin]:
    """Create a typed pointer from a raw integer address."""
    return UnsafePointer[Scalar[dtype], MutUntrackedOrigin](
        unsafe_from_address=addr
    )


def _get_ctx(device_context_ptr: PythonObject) raises -> DeviceContext:
    var addr = Int(py=device_context_ptr)
    return DeviceContext(
        OpaquePointer[MutUntrackedOrigin](unsafe_from_address=addr)
    )


# ---------------------------------------------------------------------------
# Raw-CPython argument unpacking for METH_FASTCALL dispatchers
# (`def_py_c_function`). The high-level `def_function` path pays an owning
# PythonObject per argument plus PyNumber round-trips per int — several
# hundred ns per argument. These helpers read the exact types aten_fast.py
# passes (ints, tuples of ints, driver.Buffer objects) directly, with
# borrowed references where possible. No type checking: the Python callers
# are internal and guarantee the shapes.
# ---------------------------------------------------------------------------


@always_inline
def _raw_int(obj: PyObjectPtr) -> Int:
    return Int(Python().cpython().PyLong_AsSsize_t(obj))


@always_inline
def _raw_f64(obj: PyObjectPtr) -> Float64:
    return Float64(Python().cpython().PyFloat_AsDouble(obj))


@always_inline
def _raw_tuple_int(t: PyObjectPtr, i: Int) -> Int:
    # PyTuple_GetItem returns a borrowed reference: no refcount traffic.
    ref cpy = Python().cpython()
    return Int(cpy.PyLong_AsSsize_t(cpy.PyTuple_GetItem(t, i)))


@always_inline
def _raw_dtype_int(obj: PyObjectPtr) -> DType:
    """DType from a Python int holding `max.dtype.DType.value`.

    The raw-pointer kernel convention passes dtypes as plain ints; this is
    the counterpart of the GetAttr-based `_raw_dtype` below (which reads a
    `driver.Buffer.dtype` and dies with the Buffer).
    """
    return DType._from_ui8(UInt8(_raw_int(obj))._mlir_value)


@always_inline
def _raw_dtype(buffer: PyObjectPtr) -> DType:
    ref cpy = Python().cpython()
    var dt = cpy.PyObject_GetAttrString(buffer, "dtype")
    var val = cpy.PyObject_GetAttrString(dt, "value")
    var v = Int(cpy.PyLong_AsSsize_t(val))
    cpy.Py_DecRef(val)
    cpy.Py_DecRef(dt)
    return DType._from_ui8(UInt8(v)._mlir_value)


@always_inline
def _raw_ctx(ptr_obj: PyObjectPtr) -> DeviceContext:
    return DeviceContext(
        OpaquePointer[MutUntrackedOrigin](unsafe_from_address=_raw_int(ptr_obj))
    )


@always_inline
def _raw_ret_none() -> PyObjectPtr:
    # The Python callers ignore the return value; 0 is an immortal cached
    # small int, so this is refcount-only.
    return Python().cpython().PyLong_FromSsize_t(0)


@always_inline
def _raw_tuple_f64(t: PyObjectPtr, i: Int) -> Float64:
    ref cpy = Python().cpython()
    return Float64(cpy.PyFloat_AsDouble(cpy.PyTuple_GetItem(t, i)))


@always_inline
def _raw_tuple_len(t: PyObjectPtr) -> Int:
    return Int(Python().cpython().PyObject_Length(t))


# ===========================================================================
# TensorSpec infrastructure — the single source of truth.
#
# `tensor_holder` is the sole registrar of the process-wide Python type
# objects for `TensorSpec` and `TensorHolder`. The eager module loader imports
# it before any other kernel module, allowing those modules to construct the
# shared types from this common Mojo definition. Specs are read through
# `_spec_ptr`, an unchecked bitcast that relies on this layout staying exact.
#
# INVARIANT: never define a per-module TensorSpec/TensorHolder variant —
# import these. Diverging layouts would turn the unchecked downcast into
# silent memory corruption.
#
# Spec ops (see docs/tensor_spec_design.md) do the whole op prologue in one
# boundary call: input checks, geometry, output alloc, kernel launch, and
# return `(holder, out_spec, shape_tuple, data_ptr)` via `_spec_result`.
# Errors are REAL: dispatchers catch Mojo errors and return
# `_spec_unsupported(e)`, which raises NotImplementedError into Python;
# the Python callers treat that as "take the classic path".
# ===========================================================================

# Strided kernels always work on shapes/strides padded to this rank
# (leading dims of size 1 / stride 0).
comptime MAX_RANK = 8


struct TensorHolder(Movable, Writable):
    """Owns one device allocation. Nothing else.

    `buf`'s destructor (run by this struct's destructor when the CPython
    refcount hits 0) calls `AsyncRT_DeviceBuffer_release`, which enqueues
    the stream-ordered free.
    """

    var buf: DeviceBuffer[DType.uint8]
    var nbytes: Int

    def __init__(out self, var buf: DeviceBuffer[DType.uint8], nbytes: Int):
        self.buf = buf^
        self.nbytes = nbytes

    def write_to(self, mut writer: Some[Writer]):
        # Writable is mandatory for types exposed via add_type (tp_repr).
        writer.write(
            "TensorHolder(ptr=",
            Int(self.buf.unsafe_ptr()),
            ", nbytes=",
            self.nbytes,
            ")",
        )

    @staticmethod
    def data_ptr(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        return PythonObject(Int(self_ptr[].buf.unsafe_ptr()))

    @staticmethod
    def get_nbytes(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        return PythonObject(self_ptr[].nbytes)


struct TensorSpec(Movable, Writable):
    """Layout descriptor for one mojo eager tensor. Effectively immutable:
    Python swaps which spec a tensor points to (rebind) rather than mutating
    one in place."""

    var ptr: Int  # data pointer, storage offset already applied
    var rank: Int
    var shape: IndexList[MAX_RANK]  # leading-padded with 1s
    var strides: IndexList[MAX_RANK]  # element strides, leading-padded with 0s
    var offset: Int  # storage offset in elements (informational)
    var dtype: DType
    var itemsize: Int
    var numel: Int
    var contig: Bool
    var ctx_ptr: Int  # DeviceContext address of the tensor's device

    def __init__(
        out self,
        ptr: Int,
        rank: Int,
        shape: IndexList[MAX_RANK],
        strides: IndexList[MAX_RANK],
        offset: Int,
        dtype: DType,
        itemsize: Int,
        numel: Int,
        contig: Bool,
        ctx_ptr: Int,
    ):
        self.ptr = ptr
        self.rank = rank
        self.shape = shape
        self.strides = strides
        self.offset = offset
        self.dtype = dtype
        self.itemsize = itemsize
        self.numel = numel
        self.contig = contig
        self.ctx_ptr = ctx_ptr

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "TensorSpec(ptr=",
            self.ptr,
            ", rank=",
            self.rank,
            ", numel=",
            self.numel,
            ", dtype=",
            self.dtype,
            ")",
        )

    @always_inline
    def dim(self, i: Int) -> Int:
        """Logical dim i (hides the leading-pad convention)."""
        return self.shape[MAX_RANK - self.rank + i]

    @always_inline
    def ctx(self) -> DeviceContext:
        return DeviceContext(
            OpaquePointer[MutUntrackedOrigin](unsafe_from_address=self.ctx_ptr)
        )


@always_inline
def _spec_ptr(o: PyObjectPtr) -> UnsafePointer[TensorSpec, MutAnyOrigin]:
    """The TensorSpec behind a borrowed spec argument — a pure pointer cast.

    Callers are internal and guarantee the type (never consults the type
    registry, so it works on specs registered by any kernel module)."""
    var obj = PythonObject(from_borrowed=o)
    return obj.unchecked_downcast_value_ptr[TensorSpec]().unsafe_origin_cast[
        MutAnyOrigin
    ]()


def _spec_unsupported(e: Error) -> PyObjectPtr:
    """Translate a Mojo Error into a real Python NotImplementedError: set the
    CPython error indicator and return null so the dispatcher signals failure
    (nothing is swallowed on the spec paths)."""
    ref cpy = Python().cpython()
    var msg = String(e)
    cpy.PyErr_SetString(
        cpy.get_error_global("PyExc_NotImplementedError"),
        msg.as_c_string_slice().unsafe_ptr().as_unsafe_any_origin(),
    )
    return PyObjectPtr()


@always_inline
def _row_major8(shape: IndexList[MAX_RANK], rank: Int) -> IndexList[MAX_RANK]:
    """Row-major element strides over the trailing `rank` slots (leading 0s)."""
    var strides = IndexList[MAX_RANK](0)
    var acc = 1
    for k in range(rank):
        var i = MAX_RANK - 1 - k
        strides[i] = acc
        acc *= shape[i]
    return strides


@always_inline
def _spec_group(
    var buf: DeviceBuffer[DType.uint8],
    addr: Int,
    nbytes: Int,
    rank: Int,
    shape: IndexList[MAX_RANK],
    dtype: DType,
    itemsize: Int,
    numel: Int,
    ctx_ptr: Int,
) raises -> PythonObject:
    """One (holder, out_spec, shape_tuple, data_ptr) group for a fresh
    contiguous output — everything Python needs to mint the torch wrapper."""
    var spec_obj = PythonObject(
        alloc=TensorSpec(
            ptr=addr,
            rank=rank,
            shape=shape,
            strides=_row_major8(shape, rank),
            offset=0,
            dtype=dtype,
            itemsize=itemsize,
            numel=numel,
            contig=True,
            ctx_ptr=ctx_ptr,
        )
    )
    var holder_obj = PythonObject(alloc=TensorHolder(buf=buf^, nbytes=nbytes))
    ref cpy = Python().cpython()
    var shape_tuple = cpy.PyTuple_New(rank)
    for i in range(rank):
        _ = cpy.PyTuple_SetItem(
            shape_tuple, i, cpy.PyLong_FromSsize_t(shape[MAX_RANK - rank + i])
        )
    return Python.tuple(
        holder_obj^,
        spec_obj^,
        PythonObject(from_owned=shape_tuple),
        PythonObject(addr),
    )


@always_inline
def _spec_result(
    var buf: DeviceBuffer[DType.uint8],
    addr: Int,
    nbytes: Int,
    rank: Int,
    shape: IndexList[MAX_RANK],
    dtype: DType,
    itemsize: Int,
    numel: Int,
    ctx_ptr: Int,
) raises -> PyObjectPtr:
    """Single-output spec-op result: one (holder, spec, shape, ptr) tuple."""
    var group = _spec_group(
        buf^, addr, nbytes, rank, shape, dtype, itemsize, numel, ctx_ptr
    )
    return group^.steal_data()


@always_inline
def _spec_result2(
    var buf1: DeviceBuffer[DType.uint8],
    addr1: Int,
    nbytes1: Int,
    rank1: Int,
    shape1: IndexList[MAX_RANK],
    dtype1: DType,
    itemsize1: Int,
    numel1: Int,
    var buf2: DeviceBuffer[DType.uint8],
    addr2: Int,
    nbytes2: Int,
    rank2: Int,
    shape2: IndexList[MAX_RANK],
    dtype2: DType,
    itemsize2: Int,
    numel2: Int,
    ctx_ptr: Int,
) raises -> PyObjectPtr:
    """Two-output spec-op result: ((holder, spec, shape, ptr) x 2) in ONE
    tuple, so multi-output ops stay one boundary call."""
    var g1 = _spec_group(
        buf1^, addr1, nbytes1, rank1, shape1, dtype1, itemsize1, numel1, ctx_ptr
    )
    var g2 = _spec_group(
        buf2^, addr2, nbytes2, rank2, shape2, dtype2, itemsize2, numel2, ctx_ptr
    )
    var result = Python.tuple(g1^, g2^)
    return result^.steal_data()


@always_inline
def _parallel_for[
    func: def[width: Int, alignment: Int = 1](Coord) capturing[_] -> None
](count: Int, ctx: DeviceContext) raises:
    if ctx.api() == "cpu":
        elementwise[func, simd_width=1](Coord(count), ctx)
    else:
        comptime if has_accelerator():
            elementwise[func, simd_width=1, target="gpu"](Coord(count), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")


@always_inline
def _parallel_for_dt[
    dtype: DType,
    func: def[width: Int, alignment: Int = 1](Coord) capturing[_] -> None,
](count: Int, ctx: DeviceContext) raises:
    """`_parallel_for` with an Apple-GPU float64 comptime guard."""
    comptime if dtype == DType.float64 and has_apple_gpu_accelerator():
        if ctx.api() != "cpu":
            raise Error("float64 is not supported on Apple GPU")
        elementwise[func, simd_width=1](Coord(count), ctx)
    else:
        _parallel_for[func](count, ctx)


def _copy_strided_kernel[
    dtype: DType
](
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    src_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    shape: IndexList[MAX_RANK],
    dst_strides: IndexList[MAX_RANK],
    src_strides: IndexList[MAX_RANK],
    total: Int,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while i < total:
        var rest = i
        var dst_off = 0
        var src_off = 0

        comptime for d in range(MAX_RANK - 1, 0, -1):
            var coord = rest % shape[d]
            rest = rest // shape[d]
            dst_off += coord * dst_strides[d]
            src_off += coord * src_strides[d]
        dst_off += rest * dst_strides[0]
        src_off += rest * src_strides[0]
        dst_ptr[dst_off] = src_ptr[src_off]
        i += gstride


# A transposed 2-D read is the one strided copy that the element-at-a-time
# kernel above handles catastrophically badly: consecutive threads touch
# addresses `src_ld` elements apart, so every 2-byte load pulls its own cache
# line and the copy runs at a few percent of HBM bandwidth.  Staging a square
# tile through LDS makes both the read and the write fully coalesced.  The
# tile edge is chosen so one LDS row is a 128-byte cache line; the +1 padding
# column removes the bank conflict on the transposing access.
comptime _T2D_LINE = 128


@always_inline
def _t2d_tile[dtype: DType]() -> Int:
    return _T2D_LINE // size_of[dtype]()


comptime _T2D_ROWS = 8

# HIP and CUDA both cap gridDim.y at 65535.
comptime _MAX_GRID_Y = 65535


# The scalar tile above moves one element per thread per access, so a wave of 64
# lanes issues a 128-byte request and both the read and the write walk memory in
# 128-byte runs separated by a whole matrix row -- 96 KB for the weight-gradient
# operands.  Measured on those, it sustains 2.1-2.4 TB/s of the ~4 TB/s the part
# gives a streaming copy.  Widening each access to 16 bytes makes the runs
# `_T2DV_TILE * sizeof(dtype)` = 256 bytes and cuts the request count eightfold,
# at the cost of needing the transpose to happen on the LDS side: the read stages
# a whole 16-byte run of the source's contiguous axis, so the LDS tile is
# contiguous along it and the write gathers across LDS rows.
#
# LDS is nowhere near binding here (two passes over the tile against a
# 128 B/cycle pipe is under a tenth of the HBM time), but a 32-way bank conflict
# would be, so the eight-element chunks of an LDS row are XOR-permuted by the row
# index.  That keeps the tile at exactly 32 KB -- padding would push it over and
# leave one workgroup per CU, where a memory-bound kernel needs several.
comptime _T2DV_THREADS = 256
comptime _T2DV_LANE_C = 8  # lanes spread across the source rows


@always_inline
def _t2dv_vec[dtype: DType]() -> Int:
    return 16 // size_of[dtype]()


def _transpose2d_vec_kernel[
    dtype: DType
](
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    src_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rows: Int,
    cols: Int,
    src_ld: Int,
    batch: Int,
    dst_bstride: Int,
    src_bstride: Int,
):
    """`dst[b, r, c] = src[b, c, r]`, 16 bytes per access and no LDS at all.

    One thread owns a `VEC x VEC` element block: it reads `VEC` 16-byte pieces,
    one from each of `VEC` source rows, transposes them in registers, and writes
    `VEC` 16-byte pieces, one to each of `VEC` destination rows.  Lanes are
    arranged 8 across the source rows by 8 down them, which makes both the reads
    and the writes 128-byte runs -- measured, that is within 2% of the bandwidth
    of 256-byte runs (3.57 against 3.62 TB/s) while 64-byte runs collapse to
    2.79.

    Requires `rows % VEC == 0`, `cols % VEC == 0`, `src_ld % VEC == 0` and both
    pointers 16-byte aligned, all checked by the caller.  A whole block is then
    inside the matrix or wholly outside it, so the edge costs one predicate.
    """
    comptime VEC = _t2dv_vec[dtype]()
    comptime LC = _T2DV_LANE_C  # lanes across the source rows
    comptime assert 64 % LC == 0, "the lane grid must divide a wave"
    comptime LR = 64 // LC

    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var lane = tid % 64
    var super = tid // 64
    var gstride = Int(grid_dim.x) * Int(block_dim.x) // 64

    var cb = ceildiv(cols, VEC * LC)  # source-row supertiles
    var rb = ceildiv(rows, VEC * LR)
    var supers = cb * rb

    var vals = stack_allocation[VEC * VEC, dtype]()

    var b = Int(block_idx.z)
    while b < batch:
        var dst_base = b * dst_bstride
        var src_base = b * src_bstride
        var q = super
        while q < supers:
            var c0 = ((q % cb) * LC + lane % LC) * VEC
            var r0 = ((q // cb) * LR + lane // LC) * VEC
            if c0 < cols and r0 < rows:
                comptime for j in range(VEC):
                    vals.store(
                        j * VEC,
                        src_ptr.load[width=VEC](
                            src_base + (c0 + j) * src_ld + r0
                        ),
                    )
                comptime for i in range(VEC):
                    var o = SIMD[dtype, VEC]()
                    comptime for j in range(VEC):
                        o[j] = vals[j * VEC + i]
                    dst_ptr.store(dst_base + (r0 + i) * cols + c0, o)
            q += gstride
        b += Int(grid_dim.z)


def _transpose2d_kernel[
    dtype: DType
](
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    src_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rows: Int,
    cols: Int,
    src_ld: Int,
    batch: Int,
    dst_bstride: Int,
    src_bstride: Int,
):
    """`dst[b, r, c] = src[b, c, r]` for `batch` independent matrices.

    Each matrix has a `rows x cols` row-major destination and a `cols x rows`
    source whose row stride is `src_ld`; `batch == 1` with zero batch strides is
    the plain 2-D case. Batch is carried on `block_idx.z` and grid-strided, like
    the row tiles, so the launch can clamp both dimensions.
    """
    comptime TILE = _t2d_tile[dtype]()
    var tile = stack_allocation[
        TILE * (TILE + 1), dtype, address_space=AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    var tx = tid % TILE
    var ty = tid // TILE
    var c0 = Int(block_idx.x) * TILE
    # Row tiles are grid-strided rather than one-to-one with block_idx.y, so
    # the launch can clamp the y dimension: gridDim.y is capped at 65535 on
    # both HIP and CUDA, and rows / TILE crosses that at 1.05M rows for 8-byte
    # elements. The generic kernel this fast path replaces was already clamped,
    # so without the stride a tall-thin transpose would fail to launch where it
    # previously worked. Both trip counts depend only on block indices, grid
    # dimensions and runtime scalars, so they are uniform across the block and
    # the barriers below stay outside divergent control flow.
    var row_tile_stride = Int(grid_dim.y) * TILE
    var batch_stride = Int(grid_dim.z)
    var b = Int(block_idx.z)
    while b < batch:
        var dst_base = b * dst_bstride
        var src_base = b * src_bstride
        var r0 = Int(block_idx.y) * TILE
        while r0 < rows:
            # Read src[c0 + y, r0 + tx]: consecutive tx are consecutive
            # addresses.
            var r = r0 + tx
            if r < rows:
                var y = ty
                while y < TILE:
                    var c = c0 + y
                    if c < cols:
                        tile[y * (TILE + 1) + tx] = src_ptr[
                            src_base + c * src_ld + r
                        ]
                    y += _T2D_ROWS
            barrier()
            # Write dst[r0 + y, c0 + tx]: consecutive tx are consecutive
            # addresses.
            var c = c0 + tx
            if c < cols:
                var y = ty
                while y < TILE:
                    var row = r0 + y
                    if row < rows:
                        dst_ptr[dst_base + row * cols + c] = tile[
                            tx * (TILE + 1) + y
                        ]
                    y += _T2D_ROWS
            # Every lane must finish reading the LDS tile before the next row
            # tile overwrites it.
            barrier()
            r0 += row_tile_stride
        b += batch_stride


@always_inline
def _copy_strided[
    dtype: DType
](
    dst_addr: Int,
    src_addr: Int,
    shape: IndexList[MAX_RANK],
    dst_strides: IndexList[MAX_RANK],
    src_strides: IndexList[MAX_RANK],
    ctx: DeviceContext,
) raises:
    """dst[coords] = src[coords] over a rank-8-padded strided index space
    (0-stride broadcast reads included). Layout-only: dispatch on element
    *size*, not dtype."""
    var dst_ptr = _make_ptr[dtype](dst_addr)
    var src_ptr = _make_ptr[dtype](src_addr)
    var total = 1
    for i in range(MAX_RANK):
        total *= shape[i]
    if total == 0:
        return

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(dst_ptr, src_ptr, shape, dst_strides, src_strides)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var rest = Int(idx[0].value())
            var dst_off = 0
            var src_off = 0

            comptime for d in range(MAX_RANK - 1, 0, -1):
                var coord = rest % shape[d]
                rest = rest // shape[d]
                dst_off += coord * dst_strides[d]
                src_off += coord * src_strides[d]
            dst_off += rest * dst_strides[0]
            src_off += rest * src_strides[0]
            dst_ptr[dst_off] = src_ptr[src_off]

        elementwise[func, simd_width=1](Coord(total), ctx)
    else:
        comptime if has_accelerator():
            comptime TILE = _t2d_tile[dtype]()
            var rows = shape[MAX_RANK - 2]
            var cols = shape[MAX_RANK - 1]
            # Transposed read into a contiguous destination, optionally batched:
            # the innermost two dims are a (rows, cols) row-major destination
            # whose source is a (cols, rows) matrix read down its columns, and at
            # most one leading dim -- the batch -- may be non-trivial. Batching
            # matters because the SDPA backward transposes
            # [batch*heads, seq, head_dim] four times per layer; without it those
            # fall to the generic strided copy and run at a fraction of
            # bandwidth.
            var batch = shape[MAX_RANK - 3]
            var outer_trivial = True
            for d in range(MAX_RANK - 3):
                if shape[d] != 1:
                    outer_trivial = False
            if (
                outer_trivial
                and rows > 1
                and cols > 1
                and total >= 1024
                and dst_strides[MAX_RANK - 1] == 1
                and dst_strides[MAX_RANK - 2] == cols
                and src_strides[MAX_RANK - 2] == 1
                and src_strides[MAX_RANK - 1] >= rows
                # A batch of one leaves the strides unconstrained; anything more
                # needs both operands to repeat their matrix at a fixed pitch,
                # and the destination's must be exactly one dense matrix so the
                # writes stay contiguous.
                and (
                    batch == 1
                    or (
                        dst_strides[MAX_RANK - 3] == rows * cols
                        and src_strides[MAX_RANK - 3]
                        >= cols * src_strides[MAX_RANK - 1]
                    )
                )
            ):
                # Wide regime: 16 bytes per access on both sides, which needs
                # every run this kernel touches to be a whole number of vectors
                # and both bases 16-byte aligned.  Those are properties of the
                # strides and the allocator, not of a particular shape, so it
                # serves every shape they admit; the scalar tile below serves the
                # rest.
                comptime VEC = _t2dv_vec[dtype]()
                comptime VBLK = VEC * _T2DV_LANE_C
                if (
                    size_of[dtype]() >= 2
                    and rows % VEC == 0
                    and cols % VEC == 0
                    and src_strides[MAX_RANK - 1] % VEC == 0
                    and dst_addr % 16 == 0
                    and src_addr % 16 == 0
                    and (batch == 1 or src_strides[MAX_RANK - 3] % VEC == 0)
                ):
                    _enqueue_cached[_transpose2d_vec_kernel[dtype]](
                        ctx,
                        String(t"transpose2d_vec_{dtype}"),
                        # One wave per VBLK x VBLK region, unclamped.  A clamp
                        # makes the grid-stride loop give some waves one region
                        # and some two, and the makespan is the larger; measured
                        # it is worth 0.6-2.4% at the weight-gradient shapes, so
                        # it is small, but there is nothing to trade it against
                        # -- the kernel holds no LDS and its state is per-region.
                        max(
                            1,
                            ceildiv(
                                ceildiv(cols, VBLK) * ceildiv(rows, VBLK) * 64,
                                _T2DV_THREADS,
                            ),
                        ),
                        1,
                        min(batch, _MAX_GRID_Y),
                        _T2DV_THREADS,
                        dst_ptr.as_unsafe_any_origin(),
                        src_ptr.as_unsafe_any_origin().as_immutable(),
                        rows,
                        cols,
                        src_strides[MAX_RANK - 1],
                        batch,
                        dst_strides[MAX_RANK - 3] if batch > 1 else 0,
                        src_strides[MAX_RANK - 3] if batch > 1 else 0,
                    )
                    return
                _enqueue_cached[_transpose2d_kernel[dtype]](
                    ctx,
                    String(t"transpose2d_{dtype}"),
                    ceildiv(cols, TILE),
                    # gridDim.y and .z are both capped at 65535; the kernel
                    # grid-strides row tiles and batch, so clamping here only
                    # costs extra iterations.
                    min(ceildiv(rows, TILE), _MAX_GRID_Y),
                    min(batch, _MAX_GRID_Y),
                    TILE * _T2D_ROWS,
                    dst_ptr.as_unsafe_any_origin(),
                    src_ptr.as_unsafe_any_origin().as_immutable(),
                    rows,
                    cols,
                    src_strides[MAX_RANK - 1],
                    batch,
                    dst_strides[MAX_RANK - 3] if batch > 1 else 0,
                    src_strides[MAX_RANK - 3] if batch > 1 else 0,
                )
                return
            _enqueue_cached[_copy_strided_kernel[dtype]](
                ctx,
                String(t"copy_strided_{dtype}"),
                _gs_blocks(total),
                1,
                1,
                GS_THREADS,
                dst_ptr.as_unsafe_any_origin(),
                src_ptr.as_unsafe_any_origin().as_immutable(),
                shape,
                dst_strides,
                src_strides,
                total,
            )
        else:
            raise Error("no GPU accelerator available at compile time")


@always_inline
def _scratch_copy(
    src_addr: Int,
    shape: IndexList[MAX_RANK],
    strides: IndexList[MAX_RANK],
    rank: Int,
    numel: Int,
    itemsize: Int,
    ctx: DeviceContext,
) raises -> DeviceBuffer[DType.uint8]:
    """Materialize the logical order of (shape, strides) into a fresh
    contiguous scratch buffer — the Mojo-side temporary of
    docs/tensor_spec_design.md §4.7. Python never sees a wrapper for it;
    the caller keeps the returned buffer alive until its kernel launch is
    enqueued (`_ = buf^`); the stream-ordered free rides the same queue."""
    var nbytes = numel * itemsize
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(nbytes, 1))
    var addr = Int(buf.unsafe_ptr())
    if numel > 0:
        var dst_strides = _row_major8(shape, rank)
        if itemsize == 4:
            _copy_strided[DType.uint32](
                addr, src_addr, shape, dst_strides, strides, ctx
            )
        elif itemsize == 2:
            _copy_strided[DType.uint16](
                addr, src_addr, shape, dst_strides, strides, ctx
            )
        elif itemsize == 8:
            _copy_strided[DType.uint64](
                addr, src_addr, shape, dst_strides, strides, ctx
            )
        elif itemsize == 1:
            _copy_strided[DType.uint8](
                addr, src_addr, shape, dst_strides, strides, ctx
            )
        else:
            raise Error("mojo spec temp: unsupported element size ", itemsize)
    return buf^


@always_inline
def _scratch_contig(
    a: TensorSpec, ctx: DeviceContext
) raises -> DeviceBuffer[DType.uint8]:
    """`_scratch_copy` over a spec's own logical layout."""
    return _scratch_copy(
        a.ptr, a.shape, a.strides, a.rank, a.numel, a.itemsize, ctx
    )


@always_inline
def _reduce_spec_geom(
    a: TensorSpec,
    rdims_t: PyObjectPtr,
    keepdim_o: PyObjectPtr,
    mut rows: Int,
    mut cols: Int,
    mut out_rank: Int,
    mut oshape: IndexList[MAX_RANK],
    mut pshape: IndexList[MAX_RANK],
    mut pstrides: IndexList[MAX_RANK],
    mut needs_copy: Bool,
) raises:
    """Geometry for a reduction spec op over arbitrary (sorted, normalized)
    reduce dims — Python only parses the dim spec.

    (pshape, pstrides) is the permuted logical layout — kept dims ascending,
    then reduce dims ascending, trailing-aligned — i.e. the layout the
    Python classic path used to build with permute+materialize. When the
    input is contiguous with trailing reduce dims (the hot path) it is
    already exactly that layout and `needs_copy` stays False; otherwise the
    caller materializes it via `_scratch_copy` inside the call. The output
    shape is leading-padded for `out_rank`: keepdim puts 1s at the original
    reduce positions; otherwise the kept dims pack the trailing slots.
    """
    var n = _raw_tuple_len(rdims_t)
    if n > a.rank:
        raise Error("mojo spec reduce: more reduce dims than rank")
    var is_red = IndexList[MAX_RANK](0)
    for k in range(n):
        var d = _raw_tuple_int(rdims_t, k)
        if d < 0 or d >= a.rank:
            raise Error("mojo spec reduce: reduce dim out of range")
        is_red[MAX_RANK - a.rank + d] = 1

    pshape = IndexList[MAX_RANK](1)
    pstrides = IndexList[MAX_RANK](0)
    var w = MAX_RANK - a.rank
    rows = 1
    for i in range(MAX_RANK - a.rank, MAX_RANK):
        if is_red[i] == 0:
            pshape[w] = a.shape[i]
            pstrides[w] = a.strides[i]
            rows *= a.shape[i]
            w += 1
    cols = 1
    for i in range(MAX_RANK - a.rank, MAX_RANK):
        if is_red[i] == 1:
            pshape[w] = a.shape[i]
            pstrides[w] = a.strides[i]
            cols *= a.shape[i]
            w += 1

    needs_copy = not a.contig
    for k in range(n):
        if _raw_tuple_int(rdims_t, k) != a.rank - n + k:
            needs_copy = True

    oshape = IndexList[MAX_RANK](1)
    if _raw_int(keepdim_o) != 0:
        out_rank = a.rank
        for i in range(MAX_RANK - a.rank, MAX_RANK):
            if is_red[i] == 0:
                oshape[i] = a.shape[i]
    else:
        out_rank = a.rank - n
        var w2 = MAX_RANK - out_rank
        for i in range(MAX_RANK - a.rank, MAX_RANK):
            if is_red[i] == 0:
                oshape[w2] = a.shape[i]
                w2 += 1
