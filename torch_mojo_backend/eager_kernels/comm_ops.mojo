# ===----------------------------------------------------------------------=== #
# Eager multi-GPU collectives for the mojo device, built on Modular's
# pure-Mojo P2P comm kernels (modular repo: max/kernels/src/comm/).
#
# One call launches the collective on EVERY participating GPU: the comm
# kernels are single-process by design — each GPU's kernel instance receives
# peer pointers into the other GPUs' buffers, so all instances must be
# enqueued together from the address space that owns them all. Launches are
# dispatched through `_launch_device_collective`, which places each enqueue
# on the AsyncRT worker with affinity for that device, with the GIL
# released.
#
# Python passes raw data pointers, per-device Signal-buffer pointers and
# DeviceContext pointers (from `Accelerator(i)._device_context_ptr()`), so
# every instance rides its device's default stream and stays ordered with
# the eager kernels and stream-ordered frees on that device.
#
# The Signal buffers are allocated and zeroed on the Python side
# (`torch_mojo_backend/distributed.py`) and must be sized
# `signal_header_bytes() + world_size * payload_nbytes` for the 2-stage
# bandwidth-bound path.
# ===----------------------------------------------------------------------=== #

from std.collections import InlineArray, List, Optional
from std.memory import OpaquePointer, UnsafePointer
from std.os import abort
from std.gpu.host import DeviceContext, DeviceContextList, DeviceEvent
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from std.sys import size_of

from comm import MAX_GPUS, Signal
from comm.allreduce import allreduce, elementwise_epilogue_type
from comm.device_collective import _launch_device_collective
from comm.sync import is_p2p_enabled
from layout import Coord, TileTensor, row_major

# Grad-carrying dtypes DDP buckets can hold.
comptime COMM_DTYPES = [DType.float32, DType.bfloat16, DType.float16]


def signal_header_bytes() raises -> PythonObject:
    """Byte size of the comm Signal header (Python mirrors this to size
    signal buffers as header + world_size * payload bytes)."""
    return PythonObject(size_of[Signal]())


@parameter
def _all_reduce_impl[
    dtype: DType, ngpus: Int
](
    in_ptrs: PythonObject,
    out_ptrs_obj: PythonObject,
    sig_ptrs: PythonObject,
    ctx_ptrs: PythonObject,
    numel: Int,
    average: Bool,
) raises:
    # Phase 1 — everything that can raise (Python indexing) goes through
    # Lists; InlineArrays with uninitialized slots must not exist while a
    # raise can unwind (destroying uninitialized slots is UB).
    var in_addrs = List[Int]()
    var out_addrs = List[Int]()
    var sig_addrs = List[Int]()
    var ctx_l = List[DeviceContext]()
    for i in range(ngpus):
        in_addrs.append(Int(py=in_ptrs[i]))
        out_addrs.append(Int(py=out_ptrs_obj[i]))
        sig_addrs.append(Int(py=sig_ptrs[i]))
        ctx_l.append(
            DeviceContext(
                OpaquePointer[MutUntrackedOrigin](
                    unsafe_from_address=Int(py=ctx_ptrs[i])
                )
            )
        )

    # Phase 2 — non-raising fills of the fixed-size arrays the kernels need.
    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )
    comptime InTile = TileTensor[dtype, type_of(row_major(0)), ImmutAnyOrigin]
    var in_tiles = InlineArray[InTile, ngpus](uninitialized=True)
    var out_ptrs = InlineArray[
        UnsafePointer[Scalar[dtype], MutAnyOrigin], ngpus
    ](uninitialized=True)
    var ctx_array = InlineArray[DeviceContext, ngpus](uninitialized=True)
    for i in range(ngpus):
        rank_sigs[i] = UnsafePointer[Signal, MutAnyOrigin](
            unsafe_from_address=sig_addrs[i]
        )
        in_tiles[i] = TileTensor(
            UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
                unsafe_from_address=in_addrs[i]
            ),
            row_major(numel),
        )
        out_ptrs[i] = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=out_addrs[i]
        )
        # init_pointee_move prevents DeviceContext.__del__ from dropping a
        # refcount that assigning into the uninitialized slot would trigger.
        (ctx_array.unsafe_ptr() + i).init_pointee_move(
            DeviceContext(copy=ctx_l[i])
        )
    var dev_ctxs = DeviceContextList[ngpus](ctx_array^)
    var inv = 1.0 / Float64(ngpus)

    @always_inline
    def launch[
        index: Int
    ]() raises {
        read in_tiles,
        read rank_sigs,
        read out_ptrs,
        read dev_ctxs,
        read numel,
        read inv,
        read average,
    }:
        var out_tile = TileTensor(out_ptrs[index], row_major(numel))

        @always_inline
        @parameter
        @__copy_capture(out_tile, inv)
        def avg_lambda[
            _dtype: DType, _width: SIMDSize, *, _alignment: Int
        ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
            out_tile.store[width=_width, alignment=_alignment](
                coords, (val * inv.cast[_dtype]()).cast[dtype]()
            )

        if average:
            allreduce[
                ngpus=ngpus,
                output_lambda=Optional[elementwise_epilogue_type](avg_lambda),
            ](in_tiles, out_tile, rank_sigs, dev_ctxs[index])
        else:
            allreduce[ngpus=ngpus](
                in_tiles, out_tile, rank_sigs, dev_ctxs[index]
            )

    # Release the GIL during the blocking TaskGroup wait so other Python
    # threads keep dispatching while the per-device enqueues run.
    with GILReleased(Python()):
        _launch_device_collective[ngpus](
            launch, DeviceContextList[ngpus](copy=dev_ctxs)
        )


def all_reduce(
    in_ptrs: PythonObject,
    out_ptrs: PythonObject,
    sig_ptrs: PythonObject,
    ctx_ptrs: PythonObject,
    numel: PythonObject,
    dtype_value: PythonObject,
    ngpus: PythonObject,
    average: PythonObject,
) raises -> PythonObject:
    """Allreduce across `ngpus` GPUs: out[i] = sum over ranks of in[r].

    All argument tuples are indexed by rank. `in_ptrs`/`out_ptrs` are raw
    element-aligned device addresses of contiguous same-shape tensors;
    outputs must not alias inputs (the latency-bound path writes outputs
    while peers still read inputs). `sig_ptrs` point to zero-initialized
    Signal buffers; `ctx_ptrs` are DeviceContext addresses. With
    `average`, each element is scaled by 1/ngpus in the store epilogue.
    """
    var ngpus_v = Int(py=ngpus)
    var numel_v = Int(py=numel)
    var average_v = Bool(py=average)
    var dtype = DType._from_ui8(UInt8(Int(py=dtype_value))._mlir_value)

    comptime for dt in COMM_DTYPES:
        if dtype == dt:
            comptime for n in range(2, MAX_GPUS + 1):
                if ngpus_v == n:
                    _all_reduce_impl[dt, n](
                        in_ptrs,
                        out_ptrs,
                        sig_ptrs,
                        ctx_ptrs,
                        numel_v,
                        average_v,
                    )
                    return Python.none()
            raise Error(
                t"mojo all_reduce: ngpus={ngpus_v} must be in [2, {MAX_GPUS}]"
            )
    raise Error(t"mojo all_reduce: unsupported dtype {dtype}")


# ---------------------------------------------------------------------------
# Async collectives on persistent per-device comm streams (M3).
#
# NCCL-style scheduling: the gradient-ready point is marked with an event on
# each device's default stream, the comm stream waits on it, the collective
# kernel launches on the comm stream, and a done event is recorded behind it.
# The default stream keeps executing later compute; consumers order against
# the done events (host wait from a watcher thread, or a GPU-side
# default-stream wait).
#
# The comm stream is the default stream of a SECOND owning DeviceContext for
# the same device: MAX device contexts share the CUDA primary context, so
# buffers, peer access, and events interop with the driver's context, and
# ctx.id() still reports the device id (which the comm kernels use as the
# rank). This routes the unmodified public `allreduce` onto a side stream;
# a raised-priority comm stream needs either priority on context default
# streams or capture-carrying stream launches upstream (follow-up).
# ---------------------------------------------------------------------------


struct CommStream(Movable, Writable):
    """A persistent comm 'stream' for one GPU: a second owning
    DeviceContext whose default stream carries the collectives."""

    var ctx: DeviceContext

    def __init__(out self, ctx: DeviceContext):
        self.ctx = ctx

    def write_to(self, mut writer: Some[Writer]):
        writer.write("CommStream")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("CommStream()")


struct CommWork(Movable, Writable):
    """Done events of one in-flight collective, one per rank."""

    var events: List[DeviceEvent]

    def __init__(out self, var events: List[DeviceEvent]):
        self.events = events^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("CommWork(ranks=", len(self.events), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def comm_stream_create(device_id: PythonObject) raises -> PythonObject:
    """A comm stream for GPU ``device_id`` (a secondary device context)."""
    var ctx = DeviceContext(device_id=Int(py=device_id))
    return PythonObject(alloc=CommStream(ctx^))


def _get_ctx_from(ctx_ptr: PythonObject) raises -> DeviceContext:
    return DeviceContext(
        OpaquePointer[MutUntrackedOrigin](unsafe_from_address=Int(py=ctx_ptr))
    )


def work_host_wait(work_obj: PythonObject) raises:
    """Block the calling host thread (GIL released) until the collective is
    complete on every device."""
    var work = work_obj.downcast_value_ptr[CommWork]()
    with GILReleased(Python()):
        for i in range(len(work[].events)):
            work[].events[i].synchronize()


def work_enqueue_main_stream_waits(
    work_obj: PythonObject, ctx_ptrs: PythonObject
) raises:
    """Make each device's default stream wait (GPU-side) for its done event."""
    var work = work_obj.downcast_value_ptr[CommWork]()
    for i in range(len(work[].events)):
        var ctx = _get_ctx_from(ctx_ptrs[i])
        ctx.stream().enqueue_wait_for(work[].events[i])


def work_enqueue_main_stream_wait(
    work_obj: PythonObject, index: PythonObject, ctx_ptr: PythonObject
) raises:
    """Make ONE device's default stream wait (GPU-side) for its done event.

    Rank threads call this for their own device, as late as possible (after
    the rest of backward's compute has been enqueued), so only work enqueued
    afterwards orders behind the collective.
    """
    var work = work_obj.downcast_value_ptr[CommWork]()
    var ctx = _get_ctx_from(ctx_ptr)
    ctx.stream().enqueue_wait_for(work[].events[Int(py=index)])


@parameter
def _all_reduce_async_impl[
    dtype: DType, ngpus: Int
](
    stream_objs: PythonObject,
    in_ptrs: PythonObject,
    out_ptrs_obj: PythonObject,
    sig_ptrs: PythonObject,
    ctx_ptrs: PythonObject,
    numel: Int,
    average: Bool,
    max_blocks: Int,
) raises -> PythonObject:
    # Phase 1 — everything that can raise (Python indexing, downcasts,
    # event creation) goes through Lists, which tolerate partial
    # construction. InlineArrays with uninitialized slots must not exist
    # while a raise can unwind: destroying uninitialized slots is UB.
    var in_addrs = List[Int]()
    var out_addrs = List[Int]()
    var sig_addrs = List[Int]()
    var comm_ctx_l = List[DeviceContext]()
    var main_ctx_l = List[DeviceContext]()
    var done_events = List[DeviceEvent]()
    for i in range(ngpus):
        in_addrs.append(Int(py=in_ptrs[i]))
        out_addrs.append(Int(py=out_ptrs_obj[i]))
        sig_addrs.append(Int(py=sig_ptrs[i]))
        var stream_holder = stream_objs[i].downcast_value_ptr[CommStream]()
        comm_ctx_l.append(DeviceContext(copy=stream_holder[].ctx))
        main_ctx_l.append(
            DeviceContext(
                OpaquePointer[MutUntrackedOrigin](
                    unsafe_from_address=Int(py=ctx_ptrs[i])
                )
            )
        )
        done_events.append(comm_ctx_l[i].create_event())

    # Phase 2 — non-raising fills of the fixed-size arrays the kernels need.
    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )
    comptime InTile = TileTensor[dtype, type_of(row_major(0)), ImmutAnyOrigin]
    var in_tiles = InlineArray[InTile, ngpus](uninitialized=True)
    var out_ptrs = InlineArray[
        UnsafePointer[Scalar[dtype], MutAnyOrigin], ngpus
    ](uninitialized=True)
    var comm_ctxs = InlineArray[DeviceContext, ngpus](uninitialized=True)
    var main_ctxs = InlineArray[DeviceContext, ngpus](uninitialized=True)
    for i in range(ngpus):
        rank_sigs[i] = UnsafePointer[Signal, MutAnyOrigin](
            unsafe_from_address=sig_addrs[i]
        )
        in_tiles[i] = TileTensor(
            UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
                unsafe_from_address=in_addrs[i]
            ),
            row_major(numel),
        )
        out_ptrs[i] = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=out_addrs[i]
        )
        (comm_ctxs.unsafe_ptr() + i).init_pointee_move(
            DeviceContext(copy=comm_ctx_l[i])
        )
        (main_ctxs.unsafe_ptr() + i).init_pointee_move(
            DeviceContext(copy=main_ctx_l[i])
        )
    var comm_ctx_list = DeviceContextList[ngpus](comm_ctxs^)

    var inv = 1.0 / Float64(ngpus)
    # The grid cap trades isolated collective speed for SM co-residency
    # with compute kernels — without raised stream priority, a full-grid
    # collective time-slices against fat compute kernels and extends both.
    var blocks = Optional[Int]()
    if max_blocks > 0:
        blocks = Optional[Int](max_blocks)

    @always_inline
    def launch[
        index: Int
    ]() raises {
        read in_tiles,
        read rank_sigs,
        read out_ptrs,
        read comm_ctx_list,
        read main_ctxs,
        read done_events,
        read numel,
        read average,
        read inv,
        read blocks,
    }:
        var comm_ctx = comm_ctx_list[index]
        ref main_ctx = main_ctxs[index]
        # Mark the gradient-ready point on the main (default) stream; the
        # comm stream starts only after work enqueued so far, while later
        # main-stream kernels proceed unblocked.
        var ready = main_ctx.create_event()
        main_ctx.stream().record_event(ready)
        comm_ctx.stream().enqueue_wait_for(ready)

        var out_tile = TileTensor(out_ptrs[index], row_major(numel))

        @always_inline
        @parameter
        @__copy_capture(out_tile, inv)
        def avg_lambda[
            _dtype: DType, _width: SIMDSize, *, _alignment: Int
        ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
            out_tile.store[width=_width, alignment=_alignment](
                coords, (val * inv.cast[_dtype]()).cast[dtype]()
            )

        if average:
            allreduce[
                ngpus=ngpus,
                output_lambda=Optional[elementwise_epilogue_type](avg_lambda),
            ](
                in_tiles,
                out_tile,
                rank_sigs,
                comm_ctx,
                _max_num_blocks=blocks,
            )
        else:
            allreduce[ngpus=ngpus](
                in_tiles, out_tile, rank_sigs, comm_ctx, _max_num_blocks=blocks
            )
        comm_ctx.stream().record_event(done_events[index])

    # Release the GIL during the blocking TaskGroup wait so other Python
    # threads keep dispatching while the per-device enqueues run.
    with GILReleased(Python()):
        _launch_device_collective[ngpus](
            launch, DeviceContextList[ngpus](copy=comm_ctx_list)
        )
    return PythonObject(alloc=CommWork(done_events^))


def all_reduce_async(
    stream_objs: PythonObject,
    in_ptrs: PythonObject,
    out_ptrs: PythonObject,
    sig_ptrs: PythonObject,
    ctx_ptrs: PythonObject,
    numel: PythonObject,
    dtype_value: PythonObject,
    average_and_max_blocks: PythonObject,
) raises -> PythonObject:
    """Allreduce on the per-device comm streams; returns a CommWork.

    Same contract as `all_reduce` (rank-indexed tuples, contiguous
    same-shape tensors, outputs must not alias inputs, device ids must be
    0..n-1 in order: the kernels derive each instance's rank from
    ctx.id()), plus ``stream_objs``: one CommStream per rank from
    `comm_stream_create`; the world size is the tuple length. Requires
    P2P. Signal buffers must be dedicated to the comm-stream channel:
    sync and async collectives sharing counters would mispair the barrier.
    """
    if not is_p2p_enabled():
        raise Error("all_reduce_async requires P2P access between GPUs")
    var ngpus_v = len(in_ptrs)
    var numel_v = Int(py=numel)
    # (average, max_blocks): packed to stay within def_function's arity.
    var average_v = Bool(py=average_and_max_blocks[0])
    var max_blocks_v = Int(py=average_and_max_blocks[1])
    var dtype = DType._from_ui8(UInt8(Int(py=dtype_value))._mlir_value)

    comptime for dt in COMM_DTYPES:
        if dtype == dt:
            comptime for n in range(2, MAX_GPUS + 1):
                if ngpus_v == n:
                    return _all_reduce_async_impl[dt, n](
                        stream_objs,
                        in_ptrs,
                        out_ptrs,
                        sig_ptrs,
                        ctx_ptrs,
                        numel_v,
                        average_v,
                        max_blocks_v,
                    )
            raise Error(
                t"mojo all_reduce_async: ngpus={ngpus_v} must be in"
                t" [2, {MAX_GPUS}]"
            )
    raise Error(t"mojo all_reduce_async: unsupported dtype {dtype}")


@export
def PyInit_comm_ops() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("comm_ops")
        _ = m.add_type[CommStream]("CommStream")
        _ = m.add_type[CommWork]("CommWork")
        m.def_function[signal_header_bytes]("signal_header_bytes")
        m.def_function[all_reduce]("all_reduce")
        m.def_function[comm_stream_create]("comm_stream_create")
        m.def_function[all_reduce_async]("all_reduce_async")
        m.def_function[work_host_wait]("work_host_wait")
        m.def_function[work_enqueue_main_stream_waits](
            "work_enqueue_main_stream_waits"
        )
        m.def_function[work_enqueue_main_stream_wait](
            "work_enqueue_main_stream_wait"
        )
        return m.finalize()
    except e:
        abort(t"failed to create comm_ops python module: {e}")
