# ===----------------------------------------------------------------------=== #
# The owning device-memory holder for mojo_device eager tensors, plus the
# host-transfer and strided-copy primitives that replace the Python
# `max.driver.Buffer`.
#
# Design (see docs/strided_owning_tensors_design.md): a `TensorHolder` is a
# *pure ownership token* — it owns one `DeviceBuffer[DType.uint8]` (byte-typed
# = dtype-erased) allocated on MAX's stream via `enqueue_create_buffer`. All
# layout metadata (shape / strides / storage_offset / dtype) lives on the
# Python `TorchMojoTensor` wrapper; views share the *same* holder object and
# CPython's refcount keeps the allocation alive until the last view dies, at
# which point the holder's destructor releases the AsyncRT buffer — a
# stream-ordered (fire-and-forget) free that is safe because alloc, kernels
# and free all ride the device's one default stream.
#
# Kernels never see the holder: Python passes raw data pointers (with the
# storage offset already applied) as ints. The holder/spec struct definitions
# live in `op_utils` (single shared source, compiled into every kernel module
# that registers them); this module registers them for Python and owns
# `make_spec`, the one Python-facing spec constructor.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr, Py_ssize_t
from std.python.bindings import PythonModuleBuilder

from std.sys.info import has_apple_gpu_accelerator, size_of
from std.utils import IndexList

from op_utils import (
    GS_THREADS,
    MAX_RANK,
    TensorHolder,
    TensorSpec,
    _copy_strided,
    _enqueue_cached,
    _fill_bits,
    _fill_bits_dtype,
    _fill_layout,
    _get_ctx,
    _gs_blocks,
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _raw_ret_none,
    _raw_tuple_int,
    _spec_unsupported,
)

# Every dtype a mojo_device tensor can hold (read_scalar / StridedFill
# dispatch over this at compile time).
comptime ALL_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.bool,
]


struct _PinnedHostTransfer(Movable, Writable):
    """Owns one page-locked staging allocation until Python drops it.

    For H2D, host-side memcpy into this MAX-owned pinned allocation is
    synchronous only with the calling CPU and the following DMA stays
    asynchronous. For D2H, Python exposes this allocation directly as a CPU
    tensor through DLPack, avoiding both a pageable destination and a host-side
    copy. Python retains asynchronous owners behind stream events.
    """

    var buf: HostBuffer[DType.uint8]

    def __init__(out self, buf: HostBuffer[DType.uint8]):
        self.buf = buf

    def write_to(self, mut writer: Some[Writer]):
        writer.write("PinnedHostTransfer(nbytes=", len(self.buf), ")")


def _stage_pageable_h2d(
    ctx: DeviceContext,
    dst: DeviceBuffer[DType.uint8],
    host_ptr: Int,
    nbytes: Int,
) raises -> PythonObject:
    """Copy pageable CPU bytes to pinned memory, then enqueue pinned H2D."""
    var staging = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    memcpy(
        dest=staging.unsafe_ptr(),
        src=_u8_ptr(host_ptr),
        count=nbytes,
    )
    # Construct the Python owner before enqueue: if wrapping/allocation fails,
    # there is no live DMA yet. Keep an extra reference through the exception
    # handler so even a return-path failure cannot release pinned storage before
    # the conservative stream drain.
    var owner = PythonObject(alloc=_PinnedHostTransfer(staging^))
    var owner_guard = PythonObject(copy=owner)
    var transfer = owner.downcast_value_ptr[_PinnedHostTransfer]()
    try:
        dst.enqueue_copy_from(transfer[].buf)
        return owner^
    except e:
        ctx.synchronize()
        _ = owner_guard
        raise e^


@always_inline
def _u8_ptr(
    addr: Int,
) -> UnsafePointer[Scalar[DType.uint8], MutUntrackedOrigin]:
    return _make_ptr[DType.uint8](addr)


@always_inline
def _wrap_raw(
    ctx: DeviceContext, addr: Int, nbytes: Int
) -> DeviceBuffer[DType.uint8]:
    """A non-owning DeviceBuffer view over raw device memory, for copies."""
    return DeviceBuffer[DType.uint8](ctx, _u8_ptr(addr), nbytes, owning=False)


# ---------------------------------------------------------------------------
# Allocation + host transfers. All return/accept raw addresses as Python
# ints; the holder is only ever returned as the ownership token.
# ---------------------------------------------------------------------------


def alloc(ctx_ptr: PythonObject, nbytes: PythonObject) raises -> PythonObject:
    """Allocate owning device memory. Returns (holder, data_ptr)."""
    var ctx = _get_ctx(ctx_ptr)
    var n = Int(py=nbytes)
    # Zero-sized tensors still get a real 1-byte allocation so every tensor
    # carries a valid pointer (empty driver.Buffers had sentinel pointers
    # that didn't survive round-trips).
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(n, 1))
    var addr = Int(buf.unsafe_ptr())
    var holder = PythonObject(alloc=TensorHolder(buf=buf^, nbytes=n))
    return Python.tuple(holder^, PythonObject(addr))


def alloc_from_host(
    ctx_ptr: PythonObject, host_ptr: PythonObject, nbytes: PythonObject
) raises -> PythonObject:
    """Allocate + H2D copy. Returns (holder, data_ptr, transfer_owner).

    On GPU, ``transfer_owner`` owns an exact-size MAX pinned-host staging
    allocation. Python records an event immediately after this call and retains
    that owner until the DMA completes. This keeps pageable PyTorch CPU tensors
    independent of torch-cuda while avoiding a default-stream drain.
    """
    var ctx = _get_ctx(ctx_ptr)
    var n = Int(py=nbytes)
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(n, 1))
    var transfer_owner = Python.none()
    if n > 0:
        # CPU copies run on a worker pool that is not ordered with later kernel
        # launches. Preserve the established blocking behavior on that device.
        if ctx.api() == "cpu":
            buf.enqueue_copy_from(_u8_ptr(Int(py=host_ptr)))
            ctx.synchronize()
        else:
            transfer_owner = _stage_pageable_h2d(ctx, buf, Int(py=host_ptr), n)
    var addr = Int(buf.unsafe_ptr())
    var transfer_guard = PythonObject(copy=transfer_owner)
    var holder_guard = Python.none()
    try:
        var holder = PythonObject(alloc=TensorHolder(buf=buf^, nbytes=n))
        holder_guard = PythonObject(copy=holder)
        return Python.tuple(holder^, PythonObject(addr), transfer_owner^)
    except e:
        # H2D may already be queued. The guards above retain both allocations
        # until this drain finishes, even if tuple construction consumed its
        # arguments before failing.
        ctx.synchronize()
        _ = holder_guard
        _ = transfer_guard
        raise e^


def copy_from_host(
    ctx_ptr: PythonObject,
    dev_ptr: PythonObject,
    host_ptr: PythonObject,
    nbytes: PythonObject,
) raises -> PythonObject:
    """Enqueue H2D and return a pinned transfer owner for Python to retain."""
    var n = Int(py=nbytes)
    if n == 0:
        return Python.none()
    var ctx = _get_ctx(ctx_ptr)
    var dst = _wrap_raw(ctx, Int(py=dev_ptr), n)
    if ctx.api() == "cpu":
        dst.enqueue_copy_from(_u8_ptr(Int(py=host_ptr)))
        ctx.synchronize()
        return Python.none()
    return _stage_pageable_h2d(ctx, dst, Int(py=host_ptr), n)


def copy_to_host(
    ctx_ptr: PythonObject,
    dev_ptr: PythonObject,
    host_ptr: PythonObject,
    nbytes: PythonObject,
) raises:
    """Blocking D2H into caller-owned ordinary host memory."""
    var n = Int(py=nbytes)
    if n == 0:
        return
    var ctx = _get_ctx(ctx_ptr)
    var src = _wrap_raw(ctx, Int(py=dev_ptr), n)
    src.enqueue_copy_to(_u8_ptr(Int(py=host_ptr)))
    ctx.synchronize()


def copy_to_pinned_host(
    ctx_ptr: PythonObject,
    dev_ptr: PythonObject,
    nbytes: PythonObject,
) raises -> PythonObject:
    """Enqueue D2H into a new pinned buffer; return ``(owner, data_ptr)``.

    The returned owner must survive until a stream event recorded behind the
    DMA is ready. Python only calls this for a non-blocking GPU transfer.
    """
    var n = Int(py=nbytes)
    var ctx = _get_ctx(ctx_ptr)
    var host = ctx.enqueue_create_host_buffer[DType.uint8](max(n, 1))
    # Allocate/wrap the Python owner before enqueue so a wrapping failure has
    # no outstanding DMA. Access the HostBuffer through that stable owner.
    var owner = PythonObject(alloc=_PinnedHostTransfer(host^))
    var owner_guard = PythonObject(copy=owner)
    var transfer = owner.downcast_value_ptr[_PinnedHostTransfer]()
    var addr = Int(transfer[].buf.unsafe_ptr())
    if n == 0:
        return Python.tuple(owner^, PythonObject(addr))

    var src = _wrap_raw(ctx, Int(py=dev_ptr), n)
    try:
        src.enqueue_copy_to(transfer[].buf)
        return Python.tuple(owner^, PythonObject(addr))
    except e:
        # ``owner_guard`` remains live even if tuple construction stole
        # ``owner`` before failing.
        ctx.synchronize()
        _ = owner_guard
        raise e^


def _d2d_copy_vec4_kernel(
    dst_ptr: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    src_ptr: UnsafePointer[Scalar[DType.uint32], ImmutAnyOrigin],
    vec_count_arg: Int64,
    tail_words_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var vec_count = Int(vec_count_arg)
    var tail_words = Int(tail_words_arg)
    # 16-byte vector moves over the u32-viewed buffers plus a scalar-word
    # tail; the host guarantees 16-byte-aligned bases and nbytes % 4 == 0.
    # Wider (32-byte) vectors measured slightly faster but intermittently
    # wedge this Metal driver's queue; 16 bytes is what every stable kernel
    # in this package uses.  The 4-deep unrolled body issues four
    # independent loads before the stores, which is what this GPU needs to
    # stream at full rate (sequential-per-thread access measured ~2x the
    # one-chunk-per-thread grid-stride loop).
    var gid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    if gid < tail_words:
        dst_ptr[vec_count * 4 + gid] = src_ptr[vec_count * 4 + gid]
    # Each thread owns a group of 16 consecutive 16-byte chunks — one
    # sequential 256-byte stream per thread, the length the strided-permute
    # rowloop kernel measured streaming at ~2x the one-chunk-per-thread
    # rate on this GPU.
    var groups = vec_count // 16
    var g = gid
    while g < groups:
        var b = g * 16
        for j in range(16):
            dst_ptr.store[width=4, alignment=16](
                (b + j) * 4, src_ptr.load[width=4, alignment=16]((b + j) * 4)
            )
        g += gstride
    var c = groups * 16 + gid
    if c < vec_count:
        dst_ptr.store[width=4, alignment=16](
            c * 4, src_ptr.load[width=4, alignment=16](c * 4)
        )


def copy_d2d(
    ctx_ptr: PythonObject,
    dst_ptr: PythonObject,
    src_ptr: PythonObject,
    nbytes: PythonObject,
) raises:
    """Device-to-device copy on one context.

    Stream-ordered with no sync on GPU. The CPU device runs copies on a
    worker pool that is NOT ordered with kernel execution, so there the
    copy must complete before returning or a later kernel writing the same
    buffer can be overwritten by it (seen as select_scatter flakes under
    parallel test load)."""
    _copy_d2d_raw(
        _get_ctx(ctx_ptr), Int(py=dst_ptr), Int(py=src_ptr), Int(py=nbytes)
    )


def _copy_d2d_raw(
    ctx: DeviceContext, dst_addr: Int, src_addr: Int, n: Int
) raises:
    """The copy_d2d body over raw ints (shared by both entry points)."""
    if n == 0:
        return
    comptime if has_apple_gpu_accelerator():
        if (
            ctx.api() != "cpu"
            and n % 4 == 0
            and (dst_addr | src_addr) % 16 == 0
        ):
            var words = n // 4
            var vec_count = words // 4
            _enqueue_cached[_d2d_copy_vec4_kernel](
                ctx,
                "th_d2d_copy_vec4",
                _gs_blocks(max(vec_count // 16, 1)),
                1,
                1,
                GS_THREADS,
                _make_ptr[DType.uint32](dst_addr).as_unsafe_any_origin(),
                _make_ptr[DType.uint32](src_addr)
                .as_unsafe_any_origin()
                .as_immutable(),
                Int64(vec_count),
                Int64(words - vec_count * 4),
            )
            return
    var dst = _wrap_raw(ctx, dst_addr, n)
    var src = _wrap_raw(ctx, src_addr, n)
    dst.enqueue_copy_from(src)
    if ctx.api() == "cpu":
        ctx.synchronize()


def _copy_d2d_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _copy_d2d_raw(
            _raw_ctx(args[0]),
            _raw_int(args[1]),
            _raw_int(args[2]),
            _raw_int(args[3]),
        )
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def synchronize(ctx_ptr: PythonObject) raises:
    _get_ctx(ctx_ptr).synchronize()


def read_scalar(
    ctx_ptr: PythonObject, dev_ptr: PythonObject, dtype_value: PythonObject
) raises -> PythonObject:
    """Read one element at dev_ptr as a Python bool/int/float (syncs)."""
    var ctx = _get_ctx(ctx_ptr)
    var dtype = DType._from_ui8(UInt8(Int(py=dtype_value))._mlir_value)
    var staging = InlineArray[UInt8, 16](fill=0)

    comptime for dt in ALL_DTYPES:
        if dtype == dt:
            var src = _wrap_raw(ctx, Int(py=dev_ptr), size_of[dt]())
            src.enqueue_copy_to(staging.unsafe_ptr())
            ctx.synchronize()
            var val = staging.unsafe_ptr().bitcast[Scalar[dt]]()[0]
            comptime if dt == DType.bool:
                return PythonObject(Bool(val))
            elif dt.is_floating_point():
                return PythonObject(Float64(val.cast[DType.float64]()))
            else:
                return PythonObject(Int(val))
    raise Error("read_scalar: unsupported dtype ", dtype)


# ---------------------------------------------------------------------------
# CopyStrided: dst[coords] = src[coords] over an arbitrary-rank-<=8 index
# space, with independent element strides on both sides (0-stride broadcast
# reads included). This is the shared materialize primitive: .contiguous(),
# copy_ into strided destinations, expand materialization — everything that
# moves elements between two layouts of the same dtype.
#
# Layout-only, so it dispatches on element *size*, not dtype.
#
# Raw METH_FASTCALL args:
#   (dst_ptr, src_ptr, shape8, dst_strides8, src_strides8, itemsize, ctx_ptr)
# shape8/dst_strides8/src_strides8 are int tuples padded to MAX_RANK
# leading entries (size 1 / stride 0). Pointers are element-aligned ints
# with any storage offset already applied.
# ---------------------------------------------------------------------------


def _copy_strided_go(
    dst_ptr: PyObjectPtr,
    src_ptr: PyObjectPtr,
    shape_t: PyObjectPtr,
    dst_strides_t: PyObjectPtr,
    src_strides_t: PyObjectPtr,
    itemsize_o: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var dst_addr = _raw_int(dst_ptr)
    var src_addr = _raw_int(src_ptr)
    var shape = IndexList[MAX_RANK](1)
    var dst_strides = IndexList[MAX_RANK](0)
    var src_strides = IndexList[MAX_RANK](0)
    for i in range(MAX_RANK):
        shape[i] = _raw_tuple_int(shape_t, i)
        dst_strides[i] = _raw_tuple_int(dst_strides_t, i)
        src_strides[i] = _raw_tuple_int(src_strides_t, i)
    var itemsize = _raw_int(itemsize_o)
    var ctx = _raw_ctx(ctx_ptr)

    if itemsize == 4:
        _copy_strided[DType.uint32](
            dst_addr, src_addr, shape, dst_strides, src_strides, ctx
        )
    elif itemsize == 2:
        _copy_strided[DType.uint16](
            dst_addr, src_addr, shape, dst_strides, src_strides, ctx
        )
    elif itemsize == 8:
        _copy_strided[DType.uint64](
            dst_addr, src_addr, shape, dst_strides, src_strides, ctx
        )
    elif itemsize == 1:
        _copy_strided[DType.uint8](
            dst_addr, src_addr, shape, dst_strides, src_strides, ctx
        )
    else:
        raise Error("CopyStrided: unsupported element size ", itemsize)


def _copy_strided_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _copy_strided_go(
            args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        )
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


# ---------------------------------------------------------------------------
# StridedFill: dst[coords] = value over a strided rank-<=8 destination.
# Needs the real dtype (the Float64 value is cast once). Raw args:
#   (dst_ptr, value_f64, shape8, dst_strides8, dtype_value, ctx_ptr)
# ---------------------------------------------------------------------------


@always_inline
def _strided_fill[
    dtype: DType
](
    dst_addr: Int,
    value: Float64,
    shape: IndexList[MAX_RANK],
    dst_strides: IndexList[MAX_RANK],
    ctx: DeviceContext,
) raises:
    """`dst[coords] = value`, collapsing the layout first (`_fill_layout`).

    The Apple float64 guard is kept at the *tensor's* dtype even though the
    store itself now goes through a same-width unsigned type that Metal
    accepts: declining exactly what used to be declined keeps the caller's
    error path unchanged.
    """
    comptime if dtype == DType.float64 and has_apple_gpu_accelerator():
        if ctx.api() != "cpu":
            raise Error("float64 is not supported on Apple GPU")
    comptime BITS = _fill_bits_dtype[dtype]()
    _fill_layout[BITS](
        dst_addr, _fill_bits[dtype, BITS](value), shape, dst_strides, ctx
    )


def _strided_fill_go(
    dst_ptr: PyObjectPtr,
    value_o: PyObjectPtr,
    shape_t: PyObjectPtr,
    dst_strides_t: PyObjectPtr,
    dtype_o: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var dst_addr = _raw_int(dst_ptr)
    var value = _raw_f64(value_o)
    var shape = IndexList[MAX_RANK](1)
    var dst_strides = IndexList[MAX_RANK](0)
    for i in range(MAX_RANK):
        shape[i] = _raw_tuple_int(shape_t, i)
        dst_strides[i] = _raw_tuple_int(dst_strides_t, i)
    var dtype = DType._from_ui8(UInt8(_raw_int(dtype_o))._mlir_value)
    var ctx = _raw_ctx(ctx_ptr)

    # bool's exact 0/1 byte is `_fill_bits`' job, not this dispatcher's.
    var handled = False
    comptime for dt in ALL_DTYPES:
        if dtype == dt:
            _strided_fill[dt](dst_addr, value, shape, dst_strides, ctx)
            handled = True
    if not handled:
        raise Error("StridedFill: unsupported dtype ", dtype)


def _strided_fill_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _strided_fill_go(args[0], args[1], args[2], args[3], args[4], args[5])
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


# ===========================================================================
# TensorSpec constructor (`make_spec`, the one Python-facing spec
# constructor). The shared infrastructure (TensorSpec/TensorHolder structs,
# _spec_ptr, _spec_unsupported) lives in `op_utils`; spec ops
# live next to their kernels (see docs/tensor_spec_design.md §3).
# ===========================================================================


def _make_spec_go(
    ptr_o: PyObjectPtr,
    rank_o: PyObjectPtr,
    shape_t: PyObjectPtr,
    strides_t: PyObjectPtr,
    offset_o: PyObjectPtr,
    dtype_o: PyObjectPtr,
    itemsize_o: PyObjectPtr,
    numel_o: PyObjectPtr,
    contig_o: PyObjectPtr,
    ctx_o: PyObjectPtr,
) raises -> PyObjectPtr:
    var shape = IndexList[MAX_RANK](1)
    var strides = IndexList[MAX_RANK](0)
    for i in range(MAX_RANK):
        shape[i] = _raw_tuple_int(shape_t, i)
        strides[i] = _raw_tuple_int(strides_t, i)
    var obj = PythonObject(
        alloc=TensorSpec(
            ptr=_raw_int(ptr_o),
            rank=_raw_int(rank_o),
            shape=shape,
            strides=strides,
            offset=_raw_int(offset_o),
            dtype=_raw_dtype_int(dtype_o),
            itemsize=_raw_int(itemsize_o),
            numel=_raw_int(numel_o),
            contig=_raw_int(contig_o) != 0,
            ctx_ptr=_raw_int(ctx_o),
        )
    )
    return obj^.steal_data()


def _make_spec_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        return _make_spec_go(
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
    except e:
        return _spec_unsupported(e)


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_tensor_holder() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("tensor_holder")
        _ = m.add_type[_PinnedHostTransfer]("PinnedHostTransfer")
        _ = (
            m.add_type[TensorHolder]("TensorHolder")
            .def_method[TensorHolder.data_ptr]("data_ptr")
            .def_method[TensorHolder.get_nbytes]("get_nbytes")
        )
        _ = m.add_type[TensorSpec]("TensorSpec")
        m.def_py_c_function(
            _make_spec_dispatcher,
            "make_spec",
            docstring=(
                "(ptr, rank, shape8, strides8, offset, dtype, itemsize,"
                " numel, contig, ctx_ptr) -> TensorSpec"
            ),
        )
        m.def_function[alloc]("alloc")
        m.def_function[alloc_from_host]("alloc_from_host")
        m.def_function[copy_from_host]("copy_from_host")
        m.def_function[copy_to_host]("copy_to_host")
        m.def_function[copy_to_pinned_host]("copy_to_pinned_host")
        m.def_function[copy_d2d]("copy_d2d")
        m.def_py_c_function(
            _copy_d2d_dispatcher,
            "CopyD2D",
            docstring=(
                "(ctx_ptr, dst_ptr, src_ptr, nbytes); raw fastcall"
                " device-to-device copy"
            ),
        )
        m.def_function[synchronize]("synchronize")
        m.def_function[read_scalar]("read_scalar")
        m.def_py_c_function(
            _copy_strided_dispatcher,
            "CopyStrided",
            docstring=(
                "dst[coords] = src[coords] over rank-8-padded strided"
                " layouts (element-size dispatch)"
            ),
        )
        m.def_py_c_function(
            _strided_fill_dispatcher,
            "StridedFill",
            docstring=(
                "dst[coords] = value over a rank-8-padded strided layout"
                " (dtype dispatch)"
            ),
        )
        return m.finalize()
    except e:
        abort(t"failed to create tensor_holder python module: {e}")
