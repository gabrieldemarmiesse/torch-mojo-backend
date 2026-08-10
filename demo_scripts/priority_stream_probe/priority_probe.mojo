# ===----------------------------------------------------------------------=== #
# Priority-stream probe: can Modular's pure-Mojo allreduce run on a
# raised-priority DeviceStream today, from a --emit shared-lib Python
# extension?
#
# The known blocker (upstream_issues/modular-1-...): kernels whose comptime
# parameter is a capturing @parameter closure cannot go through the split
# compile_function + DeviceStream.enqueue_function path in shared-lib builds.
# The public comm.allreduce ALWAYS passes a capturing output_lambda (even the
# SUM default captures the output TileTensor on the host).
#
# The dodge probed here: a thin wrapper KERNEL that takes the same runtime
# arguments as the internal _allreduce_{1,2}stage_kernel and constructs the
# store epilogue INSIDE device code from the `result` runtime argument. The
# wrapper itself has no capturing comptime parameters, so the split path has
# no host-side captures to materialize.
# ===----------------------------------------------------------------------=== #

from std.collections import InlineArray, List
from std.math import ceildiv
from std.memory import OpaquePointer, UnsafePointer
from std.os import abort
from std.gpu import global_idx
from std.gpu.host import (
    DeviceContext,
    DeviceEvent,
    DeviceStream,
    get_gpu_target,
)
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import simd_width_of
from std.time import perf_counter_ns

from comm import MAX_GPUS, Signal
from comm.allreduce import _allreduce_1stage_kernel, _allreduce_2stage_kernel
from comm.sync import MAX_NUM_BLOCKS_UPPER_BOUND, is_p2p_enabled
from layout import Coord, TileTensor, row_major

comptime FlatLayout = type_of(row_major(0))
comptime BLOCK_SIZE = 256


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _sum_allreduce_wrapper[
    dtype: DType,
    ngpus: Int,
    *,
    use_2stage: Bool,
](
    result: TileTensor[dtype, FlatLayout, MutAnyOrigin],
    src_tensors: InlineArray[
        TileTensor[dtype, FlatLayout, ImmutAnyOrigin], ngpus
    ],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    num_elements: Int,
    my_rank: Int,
):
    """SUM allreduce with the store epilogue built inside the kernel from the
    `result` runtime argument: no capturing comptime parameters on the entry
    point, so it survives compile_function + DeviceStream.enqueue_function in
    a shared-lib build (the device-internal closure is the same pattern the
    2-stage kernel already uses for its reduce-scatter epilogue)."""

    @always_inline
    @parameter
    @__copy_capture(result)
    def store_epilogue[
        _dtype: DType, _width: SIMDSize, *, _alignment: Int
    ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
        result.store[width=_width, alignment=_alignment](
            coords, val.cast[dtype]()
        )

    comptime if use_2stage:
        _allreduce_2stage_kernel[
            dtype,
            ngpus,
            FlatLayout,
            FlatLayout,
            BLOCK_SIZE=BLOCK_SIZE,
            output_lambda=store_epilogue,
        ](result, src_tensors, rank_sigs, num_elements, my_rank)
    else:
        _allreduce_1stage_kernel[
            dtype,
            ngpus,
            FlatLayout,
            FlatLayout,
            BLOCK_SIZE=BLOCK_SIZE,
            output_lambda=store_epilogue,
        ](result, src_tensors, rank_sigs, num_elements, my_rank)


def _iota_kernel(
    out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    numel: Int,
):
    """Trivial capture-free kernel for the minimal mechanism claim."""
    var idx = Int(global_idx.x)
    if idx < numel:
        out_ptr[idx] = Scalar[DType.float32](idx)


def _spin_kernel(
    out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    iters: Int,
):
    """FMA spin loop; each block runs ~iters FMAs. Launch with many blocks so
    the kernel executes in multiple waves (priority preemption acts at block
    boundaries). The store is data-dependent and never taken, defeating DCE
    without racing on out_ptr."""
    var acc = Float32(global_idx.x) * Float32(1e-9)
    for _ in range(iters):
        acc = acc * Float32(0.9999999) + Float32(1e-7)
    if acc == Float32(-123456.0):
        out_ptr[0] = acc


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


def _ctx_from(ctx_ptr: PythonObject) raises -> DeviceContext:
    return DeviceContext(
        OpaquePointer[MutUntrackedOrigin](unsafe_from_address=Int(py=ctx_ptr))
    )


def stream_priority_range(ctx_ptr: PythonObject) raises -> PythonObject:
    """(least, greatest) stream priority for the device of `ctx_ptr`."""
    var ctx = _ctx_from(ctx_ptr)
    var r = ctx.stream_priority_range()
    return Python.tuple(PythonObject(r.least), PythonObject(r.greatest))


def probe_simple(
    ctx_ptr: PythonObject,
    out_ptr: PythonObject,
    numel: PythonObject,
    priority: PythonObject,
) raises -> PythonObject:
    """Minimal claim: compile_function + create_stream(priority=..) +
    DeviceStream.enqueue_function of a capture-free kernel."""
    var ctx = _ctx_from(ctx_ptr)
    var stream = ctx.create_stream(priority=Int(py=priority))
    var f = ctx.compile_function[_iota_kernel]()
    var ready = ctx.create_event()
    ctx.stream().record_event(ready)
    stream.enqueue_wait_for(ready)
    var out = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=Int(py=out_ptr)
    )
    var n = Int(py=numel)
    stream.enqueue_function(
        f, out, n, grid_dim=ceildiv(n, BLOCK_SIZE), block_dim=BLOCK_SIZE
    )
    var done = ctx.create_event()
    stream.record_event(done)
    ctx.stream().enqueue_wait_for(done)
    done.synchronize()
    return Python.none()


@parameter
def _enqueue_allreduce[
    dtype: DType, ngpus: Int, use_2stage: Bool
](
    stream: DeviceStream,
    ctx: DeviceContext,
    out_tile: TileTensor[dtype, FlatLayout, MutAnyOrigin],
    in_tiles: InlineArray[TileTensor[dtype, FlatLayout, ImmutAnyOrigin], ngpus],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    numel: Int,
    my_rank: Int,
    grid_size: Int,
) raises:
    var f = ctx.compile_function[
        _sum_allreduce_wrapper[dtype, ngpus, use_2stage=use_2stage]
    ]()
    stream.enqueue_function(
        f,
        out_tile,
        in_tiles,
        rank_sigs,
        numel,
        my_rank,
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
    )


@parameter
def _priority_all_reduce_impl[
    dtype: DType, ngpus: Int, use_2stage: Bool
](
    in_ptrs: PythonObject,
    out_ptrs_obj: PythonObject,
    sig_ptrs: PythonObject,
    ctx_ptrs: PythonObject,
    numel: Int,
    priority: Int,
    spin_blocks: Int,
    spin_iters: Int,
) raises -> PythonObject:
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime if use_2stage:
        if numel % simd_width != 0:
            raise Error(
                "2-stage requires numel to be a multiple of simd width"
            )

    # Phase 1 — everything that can raise goes through Lists (comm_ops.mojo
    # two-phase pattern: InlineArrays with uninitialized slots must not exist
    # while a raise can unwind).
    var in_addrs = List[Int]()
    var out_addrs = List[Int]()
    var sig_addrs = List[Int]()
    var ctx_l = List[DeviceContext]()
    for i in range(ngpus):
        in_addrs.append(Int(py=in_ptrs[i]))
        out_addrs.append(Int(py=out_ptrs_obj[i]))
        sig_addrs.append(Int(py=sig_ptrs[i]))
        ctx_l.append(_ctx_from(ctx_ptrs[i]))

    # Phase 2 — non-raising fills of the fixed-size arrays the kernels need.
    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )
    comptime InTile = TileTensor[dtype, FlatLayout, ImmutAnyOrigin]
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
        (ctx_array.unsafe_ptr() + i).init_pointee_move(
            DeviceContext(copy=ctx_l[i])
        )

    # Grid size: identical on every rank (the cross-GPU barrier is per-block).
    var grid_size: Int
    comptime if use_2stage:
        grid_size = min(
            MAX_NUM_BLOCKS_UPPER_BOUND,
            max(1, ceildiv(numel // (simd_width * ngpus), BLOCK_SIZE)),
        )
    else:
        var num_simd_vecs = numel // simd_width
        var tail = numel - num_simd_vecs * simd_width
        grid_size = min(
            MAX_NUM_BLOCKS_UPPER_BOUND,
            max(
                max(
                    ceildiv(num_simd_vecs, BLOCK_SIZE),
                    ceildiv(tail, BLOCK_SIZE),
                ),
                1,
            ),
        )

    var streams = List[DeviceStream]()
    var comm_done = List[DeviceEvent]()
    var spin_done = List[DeviceEvent]()

    for i in range(ngpus):
        ref ctx = ctx_array[i]
        var stream = ctx.create_stream(priority=priority)
        # Gradient-ready point: recorded BEFORE the spin kernel is enqueued,
        # so the comm stream does not wait for the compute we are trying to
        # overlap with.
        var ready = ctx.create_event()
        ctx.stream().record_event(ready)
        if spin_iters > 0:
            ctx.enqueue_function[_spin_kernel](
                out_ptrs[i],
                spin_iters,
                grid_dim=spin_blocks,
                block_dim=BLOCK_SIZE,
            )
        var sd = ctx.create_event()
        ctx.stream().record_event(sd)
        stream.enqueue_wait_for(ready)
        _enqueue_allreduce[dtype, ngpus, use_2stage](
            stream,
            ctx,
            TileTensor(out_ptrs[i], row_major(numel)),
            in_tiles,
            rank_sigs,
            numel,
            i,
            grid_size,
        )
        var cd = ctx.create_event()
        stream.record_event(cd)
        # Order later default-stream work (and torch-side copies) behind the
        # collective.
        ctx.stream().enqueue_wait_for(cd)
        streams.append(stream)
        comm_done.append(cd)
        spin_done.append(sd)

    # Host timing: how long until the collective is done everywhere vs until
    # the spin compute is done everywhere.
    var t0 = perf_counter_ns()
    for i in range(ngpus):
        comm_done[i].synchronize()
    var t_comm_ms = Float64(perf_counter_ns() - t0) / 1e6
    for i in range(ngpus):
        spin_done[i].synchronize()
    var t_all_ms = Float64(perf_counter_ns() - t0) / 1e6
    _ = streams^
    return Python.tuple(PythonObject(t_comm_ms), PythonObject(t_all_ms))


def priority_all_reduce(
    in_ptrs: PythonObject,
    out_ptrs: PythonObject,
    sig_ptrs: PythonObject,
    ctx_ptrs: PythonObject,
    numel: PythonObject,
    config: PythonObject,
) raises -> PythonObject:
    """SUM allreduce (float32, 2 GPUs) on per-device side streams created
    with create_stream(priority=...).

    config = (use_2stage, priority, spin_blocks, spin_iters). With
    spin_iters > 0 a many-wave FMA kernel is enqueued on each default stream
    AFTER the ready event, so the collective must share the GPU with it;
    returns (ms until collective done on all ranks, ms until spin also done).
    """
    if not is_p2p_enabled():
        raise Error("priority probe requires P2P access between GPUs")
    var numel_v = Int(py=numel)
    var use_2stage = Bool(py=config[0])
    var priority = Int(py=config[1])
    var spin_blocks = Int(py=config[2])
    var spin_iters = Int(py=config[3])
    if len(in_ptrs) != 2:
        raise Error("probe is fixed at ngpus=2")
    if use_2stage:
        return _priority_all_reduce_impl[DType.float32, 2, True](
            in_ptrs,
            out_ptrs,
            sig_ptrs,
            ctx_ptrs,
            numel_v,
            priority,
            spin_blocks,
            spin_iters,
        )
    return _priority_all_reduce_impl[DType.float32, 2, False](
        in_ptrs,
        out_ptrs,
        sig_ptrs,
        ctx_ptrs,
        numel_v,
        priority,
        spin_blocks,
        spin_iters,
    )


@export
def PyInit_priority_probe() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("priority_probe")
        m.def_function[stream_priority_range]("stream_priority_range")
        m.def_function[probe_simple]("probe_simple")
        m.def_function[priority_all_reduce]("priority_all_reduce")
        return m.finalize()
    except e:
        abort(t"failed to create priority_probe python module: {e}")
