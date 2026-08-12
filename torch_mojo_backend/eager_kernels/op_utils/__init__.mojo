# ===----------------------------------------------------------------------=== #
# Shared helpers for the fast eager-mode kernel modules.
#
# Mirrors `max._interpreter_ops.op_utils`: unwrap `max.driver.Buffer` Python
# objects into raw typed pointers, and rebuild the MAX DeviceContext from the
# pointer that `device._device_context_ptr()` hands us on the Python side.
# ===----------------------------------------------------------------------=== #

from max.algorithm import elementwise
from std.builtin.device_passable import DevicePassable
from std.ffi import _get_global_or_null, external_call
from max.gpu.sync import barrier
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from std.math import ceildiv, sqrt
from std.memory import OpaquePointer, alloc, bitcast, stack_allocation
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr, Py_ssize_t
from std.sys import llvm_intrinsic
from std.sys.info import (
    has_accelerator,
    has_apple_gpu_accelerator,
    is_nvidia_gpu,
    simd_width_of,
    size_of,
)
from std.utils import IndexList
from std.utils.coord import Coord


# The floating-point dtypes the fast kernels specialize for. Dispatchers loop
# over this at compile time (`comptime for dt in FLOAT_DTYPES`) to pick the
# runtime dtype, which unrolls into the same `if dtype == ...` chain without
# repeating the call site once per dtype.
comptime FLOAT_DTYPES = [DType.float32, DType.float16, DType.bfloat16]


# ===========================================================================
# Correctly rounded square root
# ===========================================================================
#
# `std.math.sqrt` is not IEEE-754 on NVIDIA. Its NVIDIA arm routes every float
# dtype through `_sqrt_nvvm`, i.e. `llvm.nvvm.sqrt.approx.ftz.f`
# (`mojo/stdlib/std/math/math.mojo`), and that is a property of the stdlib,
# not of any fast-math flag we could turn off. PTX documents `sqrt.approx` at
# up to 2 ulp, and `.ftz` flushes denormals to zero on input AND output, so a
# value whose true root is denormal comes back as exactly 0. Neither is a
# precision preference: a zeroed AdamW denominator is a wrong answer, and a
# 1-2 ulp drift means no eager op that returns a square root can be compared
# bit-for-bit against ATen on CPU or CUDA.
#
# `llvm.sqrt` lowers to `sqrt.rn.f32` / `sqrt.rn.f64` on NVPTX -- correctly
# rounded and denormal preserving -- so the override is a plain intrinsic
# swap; no inline PTX is needed. Every other target already reaches the right
# instruction through `std.math.sqrt` itself, which is why the fast path stays
# gated on `is_nvidia_gpu()`: AMD expands `llvm.sqrt` to `v_sqrt_f32` plus the
# denormal rescale and the +-1 ulp fma correction (verified in the emitted
# gfx942 assembly), and Apple uses `llvm.air.sqrt`.
@always_inline
def ieee_sqrt[
    dtype: DType, width: SIMDSize, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Elementwise square root that is correctly rounded on every backend.

    Use this, not `std.math.sqrt`, wherever the root reaches a user-visible
    result. `std.math.sqrt` remains the right call only where the value feeds
    a heuristic that never leaves the kernel.

    Parameters:
        dtype: Element type of the input and output vector.
        width: SIMD width of the input and output vector.

    Args:
        x: Vector to take the square root of.

    Returns:
        The elementwise square root of `x`.
    """
    comptime if is_nvidia_gpu() and dtype.is_floating_point():
        comptime if dtype in (DType.float16, DType.bfloat16):
            # Widening is exact, and f32 carries at least 2p+2 bits for both
            # 16-bit formats (24 >= 2*11+2 for f16, 24 >= 2*8+2 for bf16), so
            # rounding a correctly rounded f32 root back down is itself
            # correctly rounded -- the classic no-double-rounding bound.
            return llvm_intrinsic[
                "llvm.sqrt", SIMD[DType.float32, width], has_side_effect=False
            ](x.cast[DType.float32]()).cast[dtype]()
        else:
            return llvm_intrinsic[
                "llvm.sqrt", SIMD[dtype, width], has_side_effect=False
            ](x)
    else:
        return sqrt(x)


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
def _device_sm_count(ctx: DeviceContext) -> Int:
    """SMs / CUs of the device actually in hand.

    The compile-time `default_device_info.sm_count` describes the
    ARCHITECTURE the variant was built for, which is the right fallback but
    not the right answer: two cards of one architecture differ here (an H100
    PCIe has 114, an H100 SXM 132, and the table reports 132 for both). Any
    grid derived from the SM count wants this, not the table.
    """
    comptime fallback = ctx.default_device_info.sm_count
    var count = 0
    try:
        count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    except:
        count = fallback
    return count if count > 0 else fallback


# ===========================================================================
# Shifted single-pass moments: the shared core of every mean+variance pass
# ===========================================================================
#
# One read of a contiguous run of elements yields the pair
#
#   s = sum(x - K)      q = sum((x - K)^2)
#
# about an assumed mean K, from which `mean = K + s/n` and the second central
# moment `M2 = q - s^2/n` follow. Shards of the same run merge by PLAIN
# ADDITION as long as they all use the same K, and an empty shard contributes
# the additive identity (0, 0) — which is why this and not Welford is what the
# split reductions here use: no per-element division, no count bookkeeping in
# the merge.
#
# The shift is what makes one pass safe. With K = 0 the two terms of `M2`
# cancel catastrophically on data with a large mean (float32 loses every
# significant digit on x ~ 1e6 + noise); shifting by anything near the mean
# leaves `q` the same order as the answer. Callers use the run's FIRST element,
# which is one broadcast load away and identical in every shard.
#
# Choosing K from the data is also what makes it fallible, so every caller
# must pair the scan with `_moment_cancels` and a second read about the
# now-known accurate mean when it answers true — see that function.
#
# Consumers: `reduction_ops._var_moments` (aten.var.correction, over an
# arbitrary (outer, reduce, inner) view) and
# `normalization_forward_kernels` (layer / group / batch norm forward, whose
# statistics are exactly this pair).


# Cancellation budget for `M2 = q - s^2/n`: re-read a slice about its accurate
# mean once `q / M2` exceeds this, i.e. once more than 4 of float32's 24
# significand bits have been eaten by the subtraction. Not fitted to any card
# — it is a property of the float32 format, and the two ends of the trade are:
#
#   * how often the second read is paid. `q/M2 = 1 + (K - mean)^2 / var`, so
#     tripping 16 needs the slice's FIRST element to sit 3.9 standard
#     deviations from its mean. Uniform data (every benchmarked case) tops out
#     at q/M2 = 4 and never re-passes; Gaussian data re-passes one slice in
#     ~1e4.
#   * the error left when it does not trip. Bounded by 16x the accumulator's
#     own relative error (~4e-7 measured at n = 2^22), i.e. ~6e-6 — still an
#     order of magnitude better than torch's own worst measured case (2.2e-4,
#     a 2048x2048 dim=0 reduction of N(1e4,1)).
comptime MOMENT_CANCEL_RATIO = Float32(16)


@always_inline
def _vec16_phase[dtype: DType](addr: Int) -> Int:
    """Element offset of `addr` inside its own 16-byte block, or -1 when `addr`
    is not even element-aligned.

    A 128-bit access is legal only at an address that is a multiple of 16, so
    this — and not the column index — is what decides where a row's vectorized
    body may start: element index `j` of a buffer sits at a 16-byte boundary iff
    `(phase + j) % V == 0`, which collapses to `j % V == 0` only for a buffer
    whose own base is 16-byte aligned.

    Shared by the log-softmax row pass, the variance moments, the generic
    reduction skeleton and the normalization forward kernels; all of them walk
    a runtime sub-range of a buffer whose base phase they do not control.
    """
    comptime esize = size_of[dtype]()
    if addr % esize != 0:
        return -1
    return (addr % 16) // esize


@always_inline
def _moment_cancels(s: Float32, q: Float32, n: Int) -> Bool:
    """Did `M2 = q - s^2/n` lose too much of the significand to be trusted?

    `q >= s^2/n` always (Cauchy-Schwarz), so the subtraction is pure
    cancellation and destroys about `log2(q / M2)` bits. `q / M2` equals
    `1 + (K - mean)^2 / var`, so it is a direct measure of how bad the assumed
    mean K was: it stays near 1 for any K within a standard deviation of the
    mean and explodes when K is an outlier — which is exactly the case a
    single-pass shifted formulation cannot survive on its own (a slice whose
    FIRST element is 1e6 while the rest are N(0,1) reaches q/M2 = 4e6, i.e.
    22 of float32's 24 bits gone and a 37% error). The caller answers a true
    here by re-reading the slice about the now-known accurate mean, which
    leaves no cancellation at all.

    A `q` of exactly 0 (constant slice) reports false and stays exact; a nan
    reports true and re-passes to the same nan.
    """
    return not ((q - s * s / Float32(n)) * MOMENT_CANCEL_RATIO >= q)


@always_inline
def _moment_partition[
    dtype: DType
](
    base_addr: Int,
    start: Int,
    n: Int,
    mut head: Int,
    mut n_vec: Int,
    mut vec_start: Int,
    mut tail_start: Int,
):
    """Split `[start, start + n)` of a buffer based at `base_addr` into a
    scalar head, a 16-byte-aligned vector body and a scalar tail.

    The split is on the ADDRESS, not the column index: the range start has
    whatever 16-byte phase the buffer's own base and `start` give it. Nothing
    here depends on the values, so a caller that re-scans the same range about
    a corrected shift passes the same partition back in.
    """
    comptime V = 16 // size_of[dtype]()
    var phase = _vec16_phase[dtype](base_addr)
    head = n
    n_vec = 0
    if phase >= 0:
        head = (V - (phase + start) % V) % V
        if head > n:
            head = n
        n_vec = (n - head) // V
    vec_start = start + head
    tail_start = head + n_vec * V


@always_inline
def _moments_scan_contig[
    dtype: DType, //, V: Int, vec_align: Int, threads: Int
](
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    start: Int,
    n: Int,
    head: Int,
    n_vec: Int,
    vec_start: Int,
    tail_start: Int,
    tid: Int,
    shift: Float32,
    mut s_t: Float32,
    mut q_t: Float32,
):
    """One thread's share of a contiguous run, as moments about `shift`.

    16-byte vector body plus the scalar head and tail the buffer's own
    alignment phase leaves over (`_moment_partition` computes the split). The
    partition arguments are shift-independent so a caller that has to re-scan
    about a corrected shift passes the same ones back in.
    """
    s_t = Float32(0)
    q_t = Float32(0)
    if n <= 0:
        return
    var s_vec = SIMD[DType.float32, V](0)
    var q_vec = SIMD[DType.float32, V](0)
    var v = tid
    while v < n_vec:
        var d = (
            in_ptr.load[width=V, alignment=vec_align](vec_start + v * V).cast[
                DType.float32
            ]()
            - shift
        )
        s_vec += d
        q_vec = d.fma(d, q_vec)
        v += threads
    s_t = s_vec.reduce_add()
    q_t = q_vec.reduce_add()

    var jh = tid
    while jh < head:
        var d = in_ptr[start + jh].cast[DType.float32]() - shift
        s_t += d
        q_t += d * d
        jh += threads
    var jt = tail_start + tid
    while jt < n:
        var d = in_ptr[start + jt].cast[DType.float32]() - shift
        s_t += d
        q_t += d * d
        jt += threads


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


@always_inline
def _runtime_sm_count(ctx: DeviceContext) -> Int:
    """SMs / CUs of the device actually in hand.

    The compile-time `default_device_info` describes the ARCHITECTURE the
    variant was built for, which is the right fallback but not the right
    answer: two cards of one architecture differ here (H100 PCIe 114 vs SXM
    132, and harvested parts differ again), and every split-reduction grid in
    this package is derived from it.
    """
    comptime fallback = ctx.default_device_info.sm_count
    var count = 0
    try:
        count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    except:
        count = fallback
    return count if count > 0 else fallback


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
# boundary call: input checks, geometry, and the kernel launch. The output is
# always allocated by Python and handed in as a trailing spec, so a spec op
# writes into it and returns None — there is no allocating return ABI.
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
def _check_into_sized(
    a: TensorSpec, dst: TensorSpec, expected_numel: Int, expected_dtype: DType
) raises:
    """Validate an Into-ABI output whose element count the op computes.

    Reductions and shape-changing ops pass the count they derived (rows,
    m*n, ...); same-shape ops go through ``_check_into``.
    (``dst``, not ``out``: that name is Mojo's result-argument keyword.)
    """
    if (
        dst.numel != expected_numel
        or not dst.contig
        or dst.ctx_ptr != a.ctx_ptr
    ):
        raise Error("mojo spec into: output buffer mismatch")
    if dst.dtype != expected_dtype:
        raise Error("mojo spec into: output dtype mismatch")


@always_inline
def _check_into(a: TensorSpec, dst: TensorSpec, expected_dtype: DType) raises:
    """Validate an Into-ABI output against its input's buffer geometry.

    The shared form of every bridge's preallocated-output check: same element
    count, contiguous, same device, and exactly the dtype Python inferred.
    """
    _check_into_sized(a, dst, a.numel, expected_dtype)


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


# ===========================================================================
# Arity-generic METH_FASTCALL dispatchers
# ===========================================================================
#
# Every spec entry point shares one skeleton: check the argument count, hand
# the raw PyObjectPtr arguments to a `_go` function (which does its own
# conversion and validation), return None, and translate any Error into
# Python's NotImplementedError. Only the arity and the target differ, so the
# target is a compile-time function parameter — the same idiom
# `_enqueue_cached` uses — and one dispatcher per observed arity replaces the
# per-op copies. `what` names the op family in the arity error.


def _spec_dispatcher2[
    go: def(PyObjectPtr, PyObjectPtr) raises thin -> None,
    what: StaticString = "spec op",
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 2:
            raise Error(what, " expects exactly 2 arguments")
        go(args[0], args[1])
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def _spec_dispatcher3[
    go: def(PyObjectPtr, PyObjectPtr, PyObjectPtr) raises thin -> None,
    what: StaticString = "spec op",
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 3:
            raise Error(what, " expects exactly 3 arguments")
        go(args[0], args[1], args[2])
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def _spec_dispatcher4[
    go: def(
        PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr
    ) raises thin -> None,
    what: StaticString = "spec op",
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 4:
            raise Error(what, " expects exactly 4 arguments")
        go(args[0], args[1], args[2], args[3])
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def _spec_dispatcher5[
    go: def(
        PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr
    ) raises thin -> None,
    what: StaticString = "spec op",
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 5:
            raise Error(what, " expects exactly 5 arguments")
        go(args[0], args[1], args[2], args[3], args[4])
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def _spec_dispatcher6[
    go: def(
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
    ) raises thin -> None,
    what: StaticString = "spec op",
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 6:
            raise Error(what, " expects exactly 6 arguments")
        go(args[0], args[1], args[2], args[3], args[4], args[5])
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def _spec_dispatcher7[
    go: def(
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
    ) raises thin -> None,
    what: StaticString = "spec op",
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 7:
            raise Error(what, " expects exactly 7 arguments")
        go(args[0], args[1], args[2], args[3], args[4], args[5], args[6])
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def _spec_dispatcher8[
    go: def(
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
    ) raises thin -> None,
    what: StaticString = "spec op",
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 8:
            raise Error(what, " expects exactly 8 arguments")
        go(
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


def _spec_dispatcher9[
    go: def(
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
    ) raises thin -> None,
    what: StaticString = "spec op",
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 9:
            raise Error(what, " expects exactly 9 arguments")
        go(
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
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def _spec_dispatcher10[
    go: def(
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
    ) raises thin -> None,
    what: StaticString = "spec op",
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 10:
            raise Error(what, " expects exactly 10 arguments")
        go(
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


def _spec_dispatcher11[
    go: def(
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
    ) raises thin -> None,
    what: StaticString = "spec op",
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 11:
            raise Error(what, " expects exactly 11 arguments")
        go(
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
            args[10],
        )
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def _spec_dispatcher12[
    go: def(
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
    ) raises thin -> None,
    what: StaticString = "spec op",
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 12:
            raise Error(what, " expects exactly 12 arguments")
        go(
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
            args[10],
            args[11],
        )
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def _spec_dispatcher13[
    go: def(
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
    ) raises thin -> None,
    what: StaticString = "spec op",
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 13:
            raise Error(what, " expects exactly 13 arguments")
        go(
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
            args[10],
            args[11],
            args[12],
        )
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


def _spec_dispatcher15[
    go: def(
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
        PyObjectPtr,
    ) raises thin -> None,
    what: StaticString = "spec op",
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 15:
            raise Error(what, " expects exactly 15 arguments")
        go(
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
            args[10],
            args[11],
            args[12],
            args[13],
            args[14],
        )
        return _raw_ret_none()
    except e:
        return _spec_unsupported(e)


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
    total_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var total = Int(total_arg)
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
    rows_arg: Int64,
    cols_arg: Int64,
    src_ld_arg: Int64,
    batch_arg: Int64,
    dst_bstride_arg: Int64,
    src_bstride_arg: Int64,
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

    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var src_ld = Int(src_ld_arg)
    var batch = Int(batch_arg)
    var dst_bstride = Int(dst_bstride_arg)
    var src_bstride = Int(src_bstride_arg)

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
    rows_arg: Int64,
    cols_arg: Int64,
    src_ld_arg: Int64,
    batch_arg: Int64,
    dst_bstride_arg: Int64,
    src_bstride_arg: Int64,
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
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var src_ld = Int(src_ld_arg)
    var batch = Int(batch_arg)
    var dst_bstride = Int(dst_bstride_arg)
    var src_bstride = Int(src_bstride_arg)
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
                        Int64(rows),
                        Int64(cols),
                        Int64(src_strides[MAX_RANK - 1]),
                        Int64(batch),
                        Int64(dst_strides[MAX_RANK - 3] if batch > 1 else 0),
                        Int64(src_strides[MAX_RANK - 3] if batch > 1 else 0),
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
                    Int64(rows),
                    Int64(cols),
                    Int64(src_strides[MAX_RANK - 1]),
                    Int64(batch),
                    Int64(dst_strides[MAX_RANK - 3] if batch > 1 else 0),
                    Int64(src_strides[MAX_RANK - 3] if batch > 1 else 0),
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
                Int64(total),
            )
        else:
            raise Error("no GPU accelerator available at compile time")


# ===========================================================================
# Fill: dst[coords] = value over a contiguous run or a strided rank-<=8
# layout. Shared by the in-place `aten::fill_.Scalar` / `aten::zero_` bridge
# in tensor_holder and by the alloc+fill factories (zeros / ones / full /
# *_like / new_*) behind `FillSpec` in elementwise_ops.
#
# A fill reads nothing and writes one repeated bit pattern, which is what
# makes both of its optimizations legal:
#
#  * The kernel never needs the real dtype. The host casts the Float64 value
#    once and the device stores the resulting bits through the unsigned
#    integer type of the same width, so thirteen fillable dtypes compile to
#    four kernels per vector width instead of thirteen -- the same dispatch
#    on element *size* that `CopyStrided` above already uses.
#  * The destination is a SET of addresses, not an ordered traversal, so the
#    layout can be collapsed hard before any kernel runs (see
#    `_collapse_fill_layout`).
#
# What bounds how many regimes this family may have is COLD START, not
# runtime: every `_enqueue_cached` instantiation is compiled into
# tensor_holder.mojo, the ungated module `_ensure_tensor_holder` builds as a
# serial barrier ahead of every other extension. The twenty kernels here --
# four element widths x {16-byte, scalar} x {contiguous, rank-2 strided},
# plus the rank-8 scalar arm -- take that module's `mojo build` from 15 s to
# 30 s on this box. Anything that multiplies another axis into that product,
# a third vector width or a third rank, buys microseconds with seconds; the
# two places that turned one down say so where the choice is made.
# ===========================================================================


@always_inline
def _fill_bits_dtype[dtype: DType]() -> DType:
    """The unsigned integer dtype `dtype`'s bit pattern is stored through."""
    comptime if size_of[dtype]() == 1:
        return DType.uint8
    else:
        comptime if size_of[dtype]() == 2:
            return DType.uint16
        else:
            comptime if size_of[dtype]() == 4:
                return DType.uint32
            else:
                return DType.uint64


@always_inline
def _fill_bits[dtype: DType, bits: DType](value: Float64) -> Scalar[bits]:
    """`value` narrowed to `dtype`, reinterpreted as `bits`.

    torch's bool is one byte holding exactly 0 or 1 while Mojo's `Bool`
    scalar is an `i1`, so bool builds its byte directly instead of
    bitcasting; every other dtype is a pure reinterpretation of the value
    the old dtype-typed store would have written.
    """
    comptime if dtype == DType.bool:
        return Scalar[bits](1) if value != Float64(0) else Scalar[bits](0)
    else:
        return bitcast[bits](value.cast[dtype]())


@always_inline
def _collapse_fill_layout(
    shape: IndexList[MAX_RANK],
    strides: IndexList[MAX_RANK],
    mut out_shape: IndexList[MAX_RANK],
    mut out_strides: IndexList[MAX_RANK],
) -> Int:
    """The smallest layout that writes the same addresses as (shape, strides).

    Because a fill stores one constant, the order dimensions are visited in
    does not matter and an address written twice is written the same value
    twice. That buys three reductions no copy could make: dimensions of
    extent 1 and dimensions of stride 0 (`expand`/broadcast views, whose
    repeats are redundant) drop outright, what is left is sorted by
    descending stride, and adjacent dimensions whose strides already tile
    (`strides[i] == strides[i + 1] * shape[i + 1]`) merge into one. A
    contiguous tensor of any rank, a transposed view of a dense matrix and an
    expanded view all end up as a single contiguous run.

    Returns the collapsed rank, with slots `[0, rank)` holding it
    outermost-first and the remaining slots padded `(1, 0)` so the rank-8
    kernel can still be handed the result. A layout whose dimensions all drop
    is a single address and comes back as rank 1, `(1, 1)`.
    """
    var n = 0
    for i in range(MAX_RANK):
        if shape[i] > 1 and strides[i] != 0:
            out_shape[n] = shape[i]
            out_strides[n] = strides[i]
            n += 1
    if n == 0:
        out_shape[0] = 1
        out_strides[0] = 1
        return 1

    # Descending stride, insertion sort (at most 8 entries).
    for i in range(1, n):
        var extent = out_shape[i]
        var stride = out_strides[i]
        var j = i
        while j > 0 and out_strides[j - 1] < stride:
            out_shape[j] = out_shape[j - 1]
            out_strides[j] = out_strides[j - 1]
            j -= 1
        out_shape[j] = extent
        out_strides[j] = stride

    # Merge innermost outward into the suffix [w, n); a merged pair keeps the
    # inner stride and multiplies the extents.
    var w = n
    for idx in range(n - 1, -1, -1):
        if w < n and out_strides[idx] == out_strides[w] * out_shape[w]:
            out_shape[w] *= out_shape[idx]
        else:
            w -= 1
            out_shape[w] = out_shape[idx]
            out_strides[w] = out_strides[idx]

    var rank = n - w
    for i in range(rank):
        out_shape[i] = out_shape[w + i]
        out_strides[i] = out_strides[w + i]
    for i in range(rank, MAX_RANK):
        out_shape[i] = 1
        out_strides[i] = 0
    return rank


comptime FILL_THREADS = GS_THREADS


@always_inline
def _fill_blocks(slots: Int) -> Int:
    """Grid for a FILL_THREADS-wide fill: cover the slots exactly.

    A fill has no reuse to protect and reads nothing, so -- unlike the
    read+write traffic `_bw_blocks` sizes -- there is no cache-resident arm
    that prefers a few long-lived workgroups. Measured on an H100 PCIe (114
    SMs, 50 MB L2, 1395 MHz core clock pinned by the suite), ours/stock
    device time on `benchmarks/test_inplace.py::test_fill_`, sweeping a cap
    of N waves of `MULTIPROCESSOR_COUNT` blocks against this exact-cover
    grid:

      cap        16.8M bf16 (33 MB, L2-resident)   16.8M fp32 (67 MB, HBM)
       1 wave                0.964                        1.098
       2 waves               0.953                        1.096
       4 waves               0.950                        1.098
       8 waves               0.943                        1.090
      16 waves               0.901                        1.055
      64 waves               0.928                        1.004
      exact cover            0.934                        0.999

    The streaming arm is the one that matters (it is the one that was
    failing) and it wants every slot to have its own thread; the resident arm
    gives up 3.5% against its own best cap, from 0.901 to 0.934, which is far
    inside the 1.10 bar. Covering exactly is therefore the whole rule: no SM
    count, no cache-size constant, nothing fitted to this card and nothing
    that can silently retarget another one. Past `_BW_MAX_BLOCKS` the
    kernels' grid-stride loop takes over.
    """
    return max(1, min(ceildiv(slots, FILL_THREADS), _BW_MAX_BLOCKS))


# Named for the profiler: a fill dispatches on element WIDTH, not dtype, so
# the name carries the width in bits and how many elements one store covers
# (`fill_contig_16bit_v8` is one 16-byte store of eight 2-byte elements).
@__name(t"fill_contig_{8 * size_of[dtype]()}bit_v{VEC}")
def _fill_vec_kernel[
    dtype: DType, VEC: Int
](
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    value: Scalar[dtype],
    nvec_arg: Int64,
    size_arg: Int64,
):
    """`dst[i] = value` for `size` elements, VEC of them per thread."""
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var nvec = Int(nvec_arg)
    var size = Int(size_arg)
    comptime ALIGN = min(16, VEC * size_of[dtype]())
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    var wide = SIMD[dtype, VEC](value)
    var j = tid
    while j < nvec:
        dst_ptr.store[width=VEC, alignment=ALIGN](j * VEC, wide)
        j += gstride
    # The tail is at most VEC-1 elements and the grid is never narrower than
    # one FILL_THREADS-wide block, so the leading threads cover all of it.
    var t = nvec * VEC + tid
    if t < size:
        dst_ptr[t] = value


@__name(t"fill_strided_{8 * size_of[dtype]()}bit_r{RANK}_v{VEC}")
def _fill_strided_kernel[
    dtype: DType, RANK: Int, VEC: Int
](
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    value: Scalar[dtype],
    shape: IndexList[MAX_RANK],
    strides: IndexList[MAX_RANK],
    total_arg: Int64,
):
    """`dst[coords] = value` over the collapsed layout in slots [0, RANK).

    The layout is in units of VEC elements: `total` counts VEC-blocks, every
    stride is a VEC-block stride, and one thread stores one whole block. The
    launcher only rewrites a layout that way when the trailing dimension is
    contiguous and every stride tiles by VEC, so VEC=1 is the general arm.
    """
    var total = Int(total_arg)
    comptime ALIGN = min(16, VEC * size_of[dtype]())
    var wide = SIMD[dtype, VEC](value)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while i < total:
        var rest = i
        var off = 0

        comptime for d in range(RANK - 1, 0, -1):
            off += (rest % shape[d]) * strides[d]
            rest = rest // shape[d]
        off += rest * strides[0]
        dst_ptr.store[width=VEC, alignment=ALIGN](off * VEC, wide)
        i += gstride


@always_inline
def _fill_contig[
    dtype: DType
](dst_addr: Int, value: Scalar[dtype], size: Int, ctx: DeviceContext) raises:
    """`dst[i] = value` for `size` contiguous elements.

    The vector width is a compile-time regime picked at runtime: 16 bytes
    when the base address admits it, one element when it does not. The check
    has to be on the address itself, not on the element count -- a tensor can
    start at any element offset inside its storage (`x[1:]`), and an
    under-aligned wide store is a corruption, not a slowdown. Nothing is
    needed at the head, exactly because the base is what is being tested; the
    at-most-VEC-1 elements at the tail are stored one at a time by the
    leading threads of the same launch, so a ragged length never costs a
    second kernel.

    Unlike `_cast`'s ladder this stops at two rungs rather than stepping
    8, 4, 2. Each rung is one more kernel per element width in
    tensor_holder's cold-start build -- the build every other extension waits
    behind -- and it only ever serves a base address whose phase is a nonzero
    multiple of 4 bytes, which is both rare and already off the fast path.
    """
    if size <= 0:
        return
    var dst_ptr = _make_ptr[dtype](dst_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(dst_ptr, value)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            dst_ptr.store[width=width](
                Int(idx[0].value()), SIMD[dtype, width](value)
            )

        elementwise[func, simd_width=simd_width_of[dtype]()](Coord(size), ctx)
        return

    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:

        @always_inline
        @parameter
        def _try_fill[VEC: Int]() raises -> Bool:
            comptime ALIGN = min(16, VEC * size_of[dtype]())
            if dst_addr % ALIGN != 0:
                return False
            var nvec = size // VEC
            _enqueue_cached[_fill_vec_kernel[dtype, VEC]](
                ctx,
                String(t"fill_vec_{dtype}_v{VEC}"),
                _fill_blocks(nvec),
                1,
                1,
                FILL_THREADS,
                dst_ptr.as_unsafe_any_origin(),
                value,
                Int64(nvec),
                Int64(size),
            )
            return True

        # 16 bytes per store; wider is not a single instruction anywhere.
        comptime WIDEST = 16 // size_of[dtype]()
        if _try_fill[WIDEST]():
            return
        # VEC=1 needs no alignment, so this always launches.
        _ = _try_fill[1]()


@always_inline
def _fill_strided[
    dtype: DType, RANK: Int, VEC: Int
](
    dst_addr: Int,
    value: Scalar[dtype],
    shape: IndexList[MAX_RANK],
    strides: IndexList[MAX_RANK],
    total: Int,
    ctx: DeviceContext,
) raises:
    """`dst[coords] = value` over a COLLAPSED rank-`RANK` layout in VEC units.
    """
    var dst_ptr = _make_ptr[dtype](dst_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(dst_ptr, value, shape, strides)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var rest = Int(idx[0].value())
            var off = 0

            comptime for d in range(RANK - 1, 0, -1):
                off += (rest % shape[d]) * strides[d]
                rest = rest // shape[d]
            off += rest * strides[0]
            dst_ptr.store[width=VEC](off * VEC, SIMD[dtype, VEC](value))

        elementwise[func, simd_width=1](Coord(total), ctx)
        return

    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        _enqueue_cached[_fill_strided_kernel[dtype, RANK, VEC]](
            ctx,
            String(t"fill_strided_{dtype}_r{RANK}_v{VEC}"),
            _fill_blocks(total),
            1,
            1,
            FILL_THREADS,
            dst_ptr.as_unsafe_any_origin(),
            value,
            shape,
            strides,
            Int64(total),
        )


@always_inline
def _fill_strided_wide[
    dtype: DType, RANK: Int
](
    dst_addr: Int,
    value: Scalar[dtype],
    shape: IndexList[MAX_RANK],
    strides: IndexList[MAX_RANK],
    rank: Int,
    total: Int,
    ctx: DeviceContext,
) raises:
    """`_fill_strided` with the trailing contiguous run vectorized.

    A collapsed layout whose innermost dimension has stride 1 is a stack of
    contiguous runs -- a channel slice, a batch slice, any dense sub-block --
    and the scalar arm would write it one element at a time. When the base
    address is aligned, the run is a whole number of VEC-element blocks and
    every outer stride tiles by VEC, the whole layout can be re-expressed in
    units of VEC elements: the index space keeps its shape, each thread's one
    store becomes 16 bytes wide, and no head or tail appears anywhere. The
    ladder walks down to VEC=1, which imposes none of those conditions and is
    therefore the general arm the old kernel always was.
    """

    @always_inline
    @parameter
    def _try_wide[VEC: Int]() raises -> Bool:
        comptime ALIGN = min(16, VEC * size_of[dtype]())
        var vshape = shape
        var vstrides = strides
        comptime if VEC > 1:
            if dst_addr % ALIGN != 0:
                return False
            if strides[rank - 1] != 1 or shape[rank - 1] % VEC != 0:
                return False
            for d in range(rank - 1):
                if strides[d] % VEC != 0:
                    return False
                vstrides[d] = strides[d] // VEC
            vshape[rank - 1] = shape[rank - 1] // VEC
            vstrides[rank - 1] = 1
        _fill_strided[dtype, RANK, VEC](
            dst_addr, value, vshape, vstrides, total // VEC, ctx
        )
        return True

    # Two rungs, not the five `_fill_contig` has: this ladder is crossed with
    # RANK, so every rung costs one kernel instantiation per rank per element
    # width, and those show up directly in the cold-start build of
    # tensor_holder (the module every other extension waits on). A layout that
    # misses the 16-byte rung is one whose runs or strides are ragged, and a
    # half-width store on ragged runs is not worth doubling that build.
    comptime WIDEST = 16 // size_of[dtype]()
    if _try_wide[WIDEST]():
        return
    _ = _try_wide[1]()


@always_inline
def _fill_layout[
    dtype: DType
](
    dst_addr: Int,
    value: Scalar[dtype],
    shape: IndexList[MAX_RANK],
    strides: IndexList[MAX_RANK],
    ctx: DeviceContext,
) raises:
    """`dst[coords] = value` over a rank-8-padded strided layout.

    `dtype` is the *bit-pattern* dtype (`_fill_bits_dtype`), never the
    tensor's own; `value` is what `_fill_bits` made of the scalar.
    """
    var total = 1
    for i in range(MAX_RANK):
        total *= shape[i]
    if total == 0:
        return

    var cshape = IndexList[MAX_RANK](1)
    var cstrides = IndexList[MAX_RANK](0)
    var rank = _collapse_fill_layout(shape, strides, cshape, cstrides)
    var count = 1
    for i in range(rank):
        count *= cshape[i]

    if rank == 1 and cstrides[0] == 1:
        _fill_contig[dtype](dst_addr, value, count, ctx)
    elif rank <= 2:
        # A collapsed rank of 1 is a single non-contiguous run; it rides the
        # rank-2 instance, whose padded trailing slot divides by 1 and adds
        # nothing, rather than earning four kernels of its own.
        _fill_strided_wide[dtype, 2](
            dst_addr, value, cshape, cstrides, rank, count, ctx
        )
    else:
        # Ranks 3-8 share the rank-8 instance, whose padded trailing slots
        # divide by 1 exactly as the uncollapsed kernel always did, and they
        # stay scalar: a layout that survives the collapse with three or more
        # mutually incompatible stride groups is rare enough that widening it
        # is not worth another four kernels in tensor_holder's cold-start
        # build (see `_fill_strided_wide`).
        _fill_strided[dtype, MAX_RANK, 1](
            dst_addr, value, cshape, cstrides, count, ctx
        )


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
) raises:
    """Geometry for a reduction spec op over trailing reduce dims of a
    contiguous operand — Python parses the dim spec AND pre-materializes.

    Any other layout raises: the Python routes permute+materialize through
    the queued strided copy (aten_fast._reduce_ready_operand), so the
    transient is allocated by `_alloc` — metered by the run-ahead budget
    and covered by the allocation retry — instead of by an invisible
    Mojo-side scratch buffer. The output shape is leading-padded for
    `out_rank`: keepdim puts 1s at the original reduce positions;
    otherwise the kept dims pack the trailing slots.
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

    if not a.contig:
        raise Error(
            "mojo spec reduce: operand must be contiguous (Python"
            " pre-materializes)"
        )
    for k in range(n):
        if _raw_tuple_int(rdims_t, k) != a.rank - n + k:
            raise Error(
                "mojo spec reduce: reduce dims must be trailing (Python"
                " pre-materializes)"
            )

    rows = 1
    cols = 1
    for i in range(MAX_RANK - a.rank, MAX_RANK):
        if is_red[i] == 0:
            rows *= a.shape[i]
        else:
            cols *= a.shape[i]

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


@always_inline
def _adjacent_reduce_geom(
    a: TensorSpec,
    rdims_t: PyObjectPtr,
    mut outer: Int,
    mut reduce_n: Int,
    mut inner: Int,
) raises -> Bool:
    """Collapse a contiguous, adjacent, ascending reduce interval into runtime
    (outer, reduce, inner) dimensions; False when the layout is anything else.

    The one geometry every reduction in this package works in: element
    (o, r, i) of the operand sits at `(o * reduce + r) * inner + i` and output
    (o, i) at `o * inner + i`. `inner == 1` is the trailing case (a (rows,
    cols) buffer, a full reduction being rows == 1); `inner > 1` is the
    interior/leading case, which is exactly the reduction Python would
    otherwise have to materialize a transposed copy for. All three are runtime
    values -- no shape is ever baked in.
    """
    if not a.contig:
        return False
    var n = _raw_tuple_len(rdims_t)
    if n == 0 or n > a.rank:
        return False
    var first = _raw_tuple_int(rdims_t, 0)
    if first < 0 or first + n > a.rank:
        return False
    for k in range(n):
        if _raw_tuple_int(rdims_t, k) != first + k:
            return False
    outer = 1
    reduce_n = 1
    inner = 1
    for d in range(first):
        outer *= a.dim(d)
    for d in range(first, first + n):
        reduce_n *= a.dim(d)
    for d in range(first + n, a.rank):
        inner *= a.dim(d)
    return True
