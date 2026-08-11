# ===----------------------------------------------------------------------=== #
# Fast eager-mode NN kernels for mojo_device: batch norm (inference),
# layer norm, row softmax (with optional causal mask), spatial mean,
# max pool (with indices), embedding gather, and boolean all-reduce.
#
# Same architecture as elementwise_ops.mojo: Python-visible functions get raw
# integer pointers (tensor `._ptr`, offset pre-applied) plus dtype ints and
# the device's DeviceContext pointer, and enqueue work on MAX's own device
# queue (fire and forget, no sync).
#
# Most kernels here are written as a parallel-for over independent output
# elements or rows (`elementwise` with an inner sequential loop), so the same
# code runs on CPU and GPU with fully dynamic shapes. The row-reduction ops
# (layer norm, argmax/max) additionally have explicit GPU kernels that
# launch one thread block per row: their row counts are far too small
# (batch * seq_len) for a thread-per-row launch to fill the GPU. Row softmax
# instead delegates its GPU path to modular's nn.softmax grid-stride kernels.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from max.gpu.sync import barrier
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.host import DeviceContext
from max.gpu.primitives import block
from std.gpu.primitives import warp
from std.math import ceildiv, exp, floor
from std.memory import alloc, stack_allocation
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import (
    _accelerator_arch,
    has_accelerator,
    has_apple_gpu_accelerator,
    size_of,
)
from std.utils.coord import Coord
from std.utils.index import IndexList
from std.utils.numerics import min_finite, min_or_neg_inf
from std.utils.static_tuple import StaticTuple

from max.algorithm import elementwise
from max.algorithm.reduction import mean
from max.algorithm.reduction import max as reduce_max
from max.algorithm.reduction import min as reduce_min

from layout import TileTensor, row_major
from native_dropout_kernels import _philox4x32_10
from nn.softmax import softmax

from std.python._cpython import PyObjectPtr, Py_ssize_t

from op_utils import (
    FLOAT_DTYPES,
    MAX_RANK,
    _check_into,
    _check_into_sized,
    _enqueue_cached,
    _make_ptr,
    _parallel_for,
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _raw_ret_none,
    _raw_tuple_f64,
    _raw_tuple_int,
    _reduce_spec_geom,
    _spec_dispatcher2,
    _spec_dispatcher4,
    _spec_dispatcher5,
    _spec_dispatcher7,
    _spec_ptr,
    _spec_unsupported,
    ieee_sqrt,
)

from variant_gates import (
    _dtype_arg_on,
    _dtype_supported,
    _op_on,
    _register_call,
)


@always_inline
def _accum_dtype[dtype: DType]() -> DType:
    """float rows accumulate in float32 (matching torch); int rows in their own
    dtype. Used to pick the reduction accumulator handed to the stdlib library.
    """
    comptime if dtype.is_floating_point():
        return DType.float32
    else:
        return dtype


# The stdlib reduction library only beats a 256-thread block-per-row kernel when
# there are too few rows to saturate the device yet each row is huge (its
# two-phase multi-block tier); elsewhere its per-call buffer allocation and
# 128-thread blocks lose. Route only that regime to the library. Mirrors
# `reduction_ops._use_library_reduce`.
comptime LIB_MIN_COLS = 1 << 20  # 1,048,576
comptime LIB_MAX_ROWS = 128


@always_inline
def _use_library_reduce(rows: Int, cols: Int) -> Bool:
    return rows <= LIB_MAX_ROWS and cols >= LIB_MIN_COLS


# ---------------------------------------------------------------------------
# Batch norm, inference mode: out = (x - mean[c]) / sqrt(var[c] + eps) * g + b
# Input is NC... contiguous; `inner` is the product of the dims after C.
# ---------------------------------------------------------------------------


@always_inline
def _batch_norm[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    mean_addr: Int,
    var_addr: Int,
    gamma_addr: Int,
    beta_addr: Int,
    eps: Float32,
    channels: Int,
    inner: Int,
    total: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var mean_ptr = _make_ptr[dtype](mean_addr)
    var var_ptr = _make_ptr[dtype](var_addr)
    var gamma_ptr = _make_ptr[dtype](gamma_addr)
    var beta_ptr = _make_ptr[dtype](beta_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var c = (i // inner) % channels
        var m = mean_ptr[c].cast[DType.float32]()
        var v = var_ptr[c].cast[DType.float32]()
        var g = gamma_ptr[c].cast[DType.float32]()
        var b = beta_ptr[c].cast[DType.float32]()
        var scale = g / ieee_sqrt(v + eps)
        var a = in_ptr[i].cast[DType.float32]()
        out_ptr[i] = ((a - m) * scale + b).cast[dtype]()

    _parallel_for[func](total, ctx)


def _batch_norm_go(
    out_ptr_obj: PyObjectPtr,
    in_ptr_obj: PyObjectPtr,
    mean_ptr_obj: PyObjectPtr,
    var_ptr_obj: PyObjectPtr,
    gamma_ptr_obj: PyObjectPtr,
    beta_ptr_obj: PyObjectPtr,
    params: PyObjectPtr,  # (eps, channels, inner, total)
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var out_addr = _raw_int(out_ptr_obj)
    var in_addr = _raw_int(in_ptr_obj)
    var mean_addr = _raw_int(mean_ptr_obj)
    var var_addr = _raw_int(var_ptr_obj)
    var gamma_addr = _raw_int(gamma_ptr_obj)
    var beta_addr = _raw_int(beta_ptr_obj)
    var eps_val = Float32(_raw_tuple_f64(params, 0))
    var channels_val = _raw_tuple_int(params, 1)
    var inner_val = _raw_tuple_int(params, 2)
    var total = _raw_tuple_int(params, 3)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _batch_norm[dt](
                    out_addr,
                    in_addr,
                    mean_addr,
                    var_addr,
                    gamma_addr,
                    beta_addr,
                    eps_val,
                    channels_val,
                    inner_val,
                    total,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error("unsupported dtype for fast batch_norm: " + String(dtype))


# ---------------------------------------------------------------------------
# Layer norm over the last dim; also writes the per-row mean and rstd
# (float32), matching aten.native_layer_norm outputs. The CPU path is one
# parallel task per row. The GPU path launches one thread block per row —
# transformer decode calls this with few rows (batch * seq_len), so a
# thread-per-row launch would leave all but a warp of the GPU idle.
# ---------------------------------------------------------------------------

comptime ROWRED_THREADS = 256
# log2(ROWRED_THREADS): halving steps in the shared-memory reduction trees.
comptime ROWRED_STAGES = 8


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name(t"layer_norm_block_{dtype}")
def _layer_norm_block_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    mean_out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd_out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    beta_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    eps: Float32,
    cols_arg: Int64,
):
    """One block per row (grid.x = rows); lanes stride over the row and
    tree-reduce the sum and squared-deviation partials in shared memory —
    the same two-pass mean/variance the CPU path computes."""
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var cols = Int(cols_arg)
    var r = block_idx.x
    var tid = thread_idx.x
    var base = r * cols

    var red = stack_allocation[
        ROWRED_THREADS, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var bcast = stack_allocation[
        2, DType.float32, address_space=AddressSpace.SHARED
    ]()

    var s = Float32(0)
    for j in range(tid, cols, ROWRED_THREADS):
        s += in_ptr[base + j].cast[DType.float32]()
    red[tid] = s
    barrier()
    var stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            red[tid] += red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        bcast[0] = red[0] / Float32(cols)
    barrier()
    var mean = bcast[0]

    var vs = Float32(0)
    for j in range(tid, cols, ROWRED_THREADS):
        var d = in_ptr[base + j].cast[DType.float32]() - mean
        vs += d * d
    red[tid] = vs
    barrier()
    stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            red[tid] += red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        var rstd0 = 1.0 / ieee_sqrt(red[0] / Float32(cols) + eps)
        bcast[1] = rstd0
        mean_out_ptr[r] = mean
        rstd_out_ptr[r] = rstd0
    barrier()
    var rstd = bcast[1]

    for j in range(tid, cols, ROWRED_THREADS):
        var x = in_ptr[base + j].cast[DType.float32]()
        var g = gamma_ptr[j].cast[DType.float32]()
        var b = beta_ptr[j].cast[DType.float32]()
        out_ptr[base + j] = ((x - mean) * rstd * g + b).cast[dtype]()


@always_inline
def _layer_norm[
    dtype: DType
](
    out_addr: Int,
    mean_out_addr: Int,
    rstd_out_addr: Int,
    in_addr: Int,
    gamma_addr: Int,
    beta_addr: Int,
    eps: Float32,
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var mean_out_ptr = _make_ptr[DType.float32](mean_out_addr)
    var rstd_out_ptr = _make_ptr[DType.float32](rstd_out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var gamma_ptr = _make_ptr[dtype](gamma_addr)
    var beta_ptr = _make_ptr[dtype](beta_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(
            out_ptr, mean_out_ptr, rstd_out_ptr, in_ptr, gamma_ptr, beta_ptr
        )
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var r = Int(idx[0].value())
            var base = r * cols
            var total = Float32(0)
            for j in range(cols):
                total += in_ptr[base + j].cast[DType.float32]()
            var mean = total / Float32(cols)
            var var_sum = Float32(0)
            for j in range(cols):
                var d = in_ptr[base + j].cast[DType.float32]() - mean
                var_sum += d * d
            var rstd = 1.0 / ieee_sqrt(var_sum / Float32(cols) + eps)
            for j in range(cols):
                var x = in_ptr[base + j].cast[DType.float32]()
                var g = gamma_ptr[j].cast[DType.float32]()
                var b = beta_ptr[j].cast[DType.float32]()
                out_ptr[base + j] = ((x - mean) * rstd * g + b).cast[dtype]()
            mean_out_ptr[r] = mean
            rstd_out_ptr[r] = rstd

        _parallel_for[func](rows, ctx)
    else:
        comptime if has_accelerator():
            _enqueue_cached[_layer_norm_block_kernel[dtype]](
                ctx,
                String(t"layer_norm_block_{dtype}"),
                rows,
                1,
                1,
                ROWRED_THREADS,
                out_ptr.as_unsafe_any_origin(),
                mean_out_ptr.as_unsafe_any_origin(),
                rstd_out_ptr.as_unsafe_any_origin(),
                in_ptr.as_unsafe_any_origin().as_immutable(),
                gamma_ptr.as_unsafe_any_origin().as_immutable(),
                beta_ptr.as_unsafe_any_origin().as_immutable(),
                eps,
                Int64(cols),
            )
        else:
            raise Error("no GPU accelerator available at compile time")


def _layer_norm_go(
    out_ptr_obj: PyObjectPtr,
    mean_out_ptr_obj: PyObjectPtr,
    rstd_out_ptr_obj: PyObjectPtr,
    in_ptr_obj: PyObjectPtr,
    gamma_ptr_obj: PyObjectPtr,
    beta_ptr_obj: PyObjectPtr,
    params: PyObjectPtr,  # (eps, rows, cols)
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var out_addr = _raw_int(out_ptr_obj)
    var mean_out_addr = _raw_int(mean_out_ptr_obj)
    var rstd_out_addr = _raw_int(rstd_out_ptr_obj)
    var in_addr = _raw_int(in_ptr_obj)
    var gamma_addr = _raw_int(gamma_ptr_obj)
    var beta_addr = _raw_int(beta_ptr_obj)
    var eps_val = Float32(_raw_tuple_f64(params, 0))
    var rows_val = _raw_tuple_int(params, 1)
    var cols_val = _raw_tuple_int(params, 2)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _layer_norm[dt](
                    out_addr,
                    mean_out_addr,
                    rstd_out_addr,
                    in_addr,
                    gamma_addr,
                    beta_addr,
                    eps_val,
                    rows_val,
                    cols_val,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error("unsupported dtype for fast layer_norm: " + String(dtype))


# ---------------------------------------------------------------------------
# Row-wise softmax with optional scaling and causal masking, for attention.
# Input is (rows, cols) where rows = batch * q_len. With causal=1, row r
# (query index r % q_len) only attends to columns j <= r % q_len — the
# top-left-aligned tril(ones(L, S)) mask that torch's sdpa is_causal=True
# specifies; masked columns get probability 0.
#
# The CUDA/ROCm GPU path delegates to modular's `nn.softmax.softmax`, which
# runs an online single-pass kernel (2 input reads + 1 write) — less HBM
# traffic than a hand-written 4-pass block kernel — and a warp-shuffle kernel
# for short rows (cols <= WARP_SIZE: 32 NVIDIA, 64 AMD). `scale` and the
# causal mask are folded into the input lambda: the value is read and scaled
# in float32 (for scale == 1 the round-trip back to `dtype` is exact), and
# masked columns are fed as -inf so their softmax weight is exactly 0,
# matching the CPU branch below. Because
# `allowed = min(cols, r % q_len + 1) >= 1`, no causal row is ever fully
# masked. The CPU MAX device keeps the explicit per-row loop below.
#
# On Apple GPUs the library kernel is ~10x off the bandwidth roofline (2.65 ms
# for 12288x256 f32 on this machine's ~100 GB/s part), so Metal gets custom
# kernels instead:
#
#   * `_softmax_rows_warp_kernel`: one simdgroup (32 lanes) per row, the whole
#     row held in registers (`_APPLE_SM_MAX_VPT` SIMD[V] slots per lane), so the row
#     is read exactly once and only up to the causal boundary — shuffles for
#     the max/sum, no shared memory, no re-read. Handles rows up to
#     32 * _APPLE_SM_MAX_VPT * V elements; V = 16B vectors when the pointers are
#     16B-aligned and cols is a multiple of V, else scalar (V = 1).
#   * `_softmax_rows_block_kernel`: generic fallback for longer or unaligned
#     rows — one thread block per row, online (m, s) accumulation in one read
#     (see `_log_softmax_rows_block_kernel` for the min_finite-not-inf
#     rationale), block.max/block.sum combine, then an output pass that
#     re-reads via the cache and zero-fills past the causal boundary.
# ---------------------------------------------------------------------------

comptime _APPLE_SM_WARPS_PER_BLOCK = 8
comptime _APPLE_SM_MAX_VPT = 8  # per-lane register slots in the warp kernel
comptime _APPLE_SM_BIG_ROW_BYTES = 25_000  # 1024-thread blocks above this


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_APPLE_SM_WARPS_PER_BLOCK * WARP_SIZE)
    )
)
@__name(t"softmax_rows_warp_{dtype}_{V}")
def _softmax_rows_warp_kernel[
    dtype: DType, V: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
    scale: Float32,
    causal_arg: Int64,
    q_len_arg: Int64,
):
    # One warp per row; requires cols % V == 0 and
    # cols <= WARP_SIZE * _APPLE_SM_MAX_VPT * V (host-checked). Lanes at or past
    # the causal boundary carry min_finite (finite, so `exp` below stays in
    # range even when the row max is tiny) and are explicitly zeroed before
    # the sum, so no sentinel arithmetic can leak into the result.
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var causal = Int(causal_arg)
    var q_len = Int(q_len_arg)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * _APPLE_SM_WARPS_PER_BLOCK + Int(warp_id())
    var row_stride = Int(grid_dim.x) * _APPLE_SM_WARPS_PER_BLOCK
    var n_vec = cols // V
    while row < rows:
        var base = row * cols
        var allowed = cols
        if causal != 0:
            allowed = min(cols, row % q_len + 1)

        # Read pass: the row (up to the causal boundary) into registers,
        # scaled, with a per-lane running max.
        var vals = StaticTuple[SIMD[DType.float32, V], _APPLE_SM_MAX_VPT]()
        var m_t = min_finite[DType.float32]()
        comptime for k in range(_APPLE_SM_MAX_VPT):
            var j0 = (lane + k * WARP_SIZE) * V
            var x = SIMD[DType.float32, V](min_finite[DType.float32]())
            if j0 < allowed:  # implies j0 < cols, i.e. a full in-row vector
                var raw = (
                    in_ptr.load[width=V, alignment=V * size_of[dtype]()](
                        base + j0
                    ).cast[DType.float32]()
                    * scale
                )
                if j0 + V > allowed:
                    comptime for li in range(V):
                        if j0 + li < allowed:
                            x[li] = raw[li]
                else:
                    x = raw
            vals[k] = x
            m_t = max(m_t, x.reduce_max())
        var row_m = warp.max(m_t)

        # exp pass in registers; masked lanes forced to exactly 0.
        var s_t = Float32(0)
        comptime for k in range(_APPLE_SM_MAX_VPT):
            var j0 = (lane + k * WARP_SIZE) * V
            var e = SIMD[DType.float32, V](0)
            if j0 < allowed:
                e = exp(vals[k] - row_m)
                if j0 + V > allowed:
                    comptime for li in range(V):
                        if j0 + li >= allowed:
                            e[li] = 0
            vals[k] = e
            s_t += e.reduce_add()
        var inv = Float32(1) / warp.sum(s_t)

        # Write pass: probabilities, zeros past the causal boundary.
        comptime for k in range(_APPLE_SM_MAX_VPT):
            var v = lane + k * WARP_SIZE
            if v < n_vec:
                out_ptr.store[width=V, alignment=V * size_of[dtype]()](
                    base + v * V, (vals[k] * inv).cast[dtype]()
                )
        row += row_stride


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(threads))
)
@__name(t"softmax_rows_block_{dtype}_{threads}_{vec}")
def _softmax_rows_block_kernel[
    dtype: DType, threads: Int, vec: Bool
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
    scale: Float32,
    causal_arg: Int64,
    q_len_arg: Int64,
):
    # One thread block per row (grid-stride over rows), any rows/cols >= 1.
    # `vec = True` requires 16B-aligned base pointers (host-checked); rows
    # whose start is then still unaligned get a scalar head, exactly like
    # `_log_softmax_rows_block_kernel`.
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var causal = Int(causal_arg)
    var q_len = Int(q_len_arg)
    comptime V = 16 // size_of[dtype]()
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    while row < rows:
        var base = row * cols
        var allowed = cols
        if causal != 0:
            allowed = min(cols, row % q_len + 1)

        # ---- Pass 1: online (max, sum) over the allowed prefix, read once.
        var m_t = min_finite[DType.float32]()
        var s_t = Float32(0)

        comptime if vec:
            var head = (V - (base % V)) % V
            if head > cols:
                head = cols
            var n_vec_a = 0
            if allowed > head:
                n_vec_a = (allowed - head) // V

            var m_vec = SIMD[DType.float32, V](min_finite[DType.float32]())
            var s_vec = SIMD[DType.float32, V](0)
            var v = tid
            while v < n_vec_a:
                var x = (
                    in_ptr.load[width=V, alignment=16](
                        base + head + v * V
                    ).cast[DType.float32]()
                    * scale
                )
                var nm = max(m_vec, x)
                s_vec = s_vec * exp(m_vec - nm) + exp(x - nm)
                m_vec = nm
                v += threads
            m_t = m_vec.reduce_max()
            s_t = (s_vec * exp(m_vec - m_t)).reduce_add()

            # Scalar head plus the partial vector at the causal boundary.
            var jh = tid
            while jh < min(head, allowed):
                var x = in_ptr[base + jh].cast[DType.float32]() * scale
                var nm = max(m_t, x)
                s_t = s_t * exp(m_t - nm) + exp(x - nm)
                m_t = nm
                jh += threads
            var jt = head + n_vec_a * V + tid
            while jt < allowed:
                var x = in_ptr[base + jt].cast[DType.float32]() * scale
                var nm = max(m_t, x)
                s_t = s_t * exp(m_t - nm) + exp(x - nm)
                m_t = nm
                jt += threads
        else:
            var j = tid
            while j < allowed:
                var x = in_ptr[base + j].cast[DType.float32]() * scale
                var nm = max(m_t, x)
                s_t = s_t * exp(m_t - nm) + exp(x - nm)
                m_t = nm
                j += threads

        var block_m = block.max[block_size=threads, broadcast=True](m_t)
        var block_s = block.sum[block_size=threads, broadcast=True](
            s_t * exp(m_t - block_m)
        )
        var inv = Float32(1) / block_s

        # ---- Pass 2: write probabilities, zeros past the boundary. The
        # allowed prefix is re-read through the cache.
        comptime if vec:
            var head = (V - (base % V)) % V
            if head > cols:
                head = cols
            var n_vec_c = (cols - head) // V
            var jh = tid
            while jh < head:
                var y = Scalar[dtype](0)
                if jh < allowed:
                    var x = in_ptr[base + jh].cast[DType.float32]() * scale
                    y = (exp(x - block_m) * inv).cast[dtype]()
                out_ptr[base + jh] = y
                jh += threads
            var v = tid
            while v < n_vec_c:
                var j0 = head + v * V
                var y = SIMD[dtype, V](0)
                if j0 < allowed:
                    var x = (
                        in_ptr.load[width=V, alignment=16](base + j0).cast[
                            DType.float32
                        ]()
                        * scale
                    )
                    var e = exp(x - block_m) * inv
                    if j0 + V > allowed:
                        comptime for li in range(V):
                            if j0 + li >= allowed:
                                e[li] = 0
                    y = e.cast[dtype]()
                out_ptr.store[width=V, alignment=16](base + j0, y)
                v += threads
            var jt = head + n_vec_c * V + tid
            while jt < cols:
                var y = Scalar[dtype](0)
                if jt < allowed:
                    var x = in_ptr[base + jt].cast[DType.float32]() * scale
                    y = (exp(x - block_m) * inv).cast[dtype]()
                out_ptr[base + jt] = y
                jt += threads
        else:
            var j = tid
            while j < allowed:
                var x = in_ptr[base + j].cast[DType.float32]() * scale
                out_ptr[base + j] = (exp(x - block_m) * inv).cast[dtype]()
                j += threads
            var jz = allowed + tid
            while jz < cols:
                out_ptr[base + jz] = Scalar[dtype](0)
                jz += threads

        row += Int(grid_dim.x)


@always_inline
def _softmax_rows_apple[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    rows: Int,
    cols: Int,
    scale: Float32,
    causal: Int,
    q_len: Int,
    ctx: DeviceContext,
) raises:
    """Regime dispatch for the Apple-GPU row-softmax kernels."""
    comptime V = 16 // size_of[dtype]()
    var mout = out_ptr.as_unsafe_any_origin()
    var min_ = in_ptr.as_unsafe_any_origin().as_immutable()
    var aligned = Int(out_ptr) % 16 == 0 and Int(in_ptr) % 16 == 0
    var warp_grid = min(
        (rows + _APPLE_SM_WARPS_PER_BLOCK - 1) // _APPLE_SM_WARPS_PER_BLOCK,
        32768,
    )
    if aligned and cols % V == 0 and cols <= WARP_SIZE * _APPLE_SM_MAX_VPT * V:
        _enqueue_cached[_softmax_rows_warp_kernel[dtype, V]](
            ctx,
            String(t"softmax_rows_warp_{dtype}_{V}"),
            warp_grid,
            1,
            1,
            _APPLE_SM_WARPS_PER_BLOCK * WARP_SIZE,
            mout,
            min_,
            Int64(rows),
            Int64(cols),
            scale,
            Int64(causal),
            Int64(q_len),
        )
    elif cols <= WARP_SIZE * _APPLE_SM_MAX_VPT:
        _enqueue_cached[_softmax_rows_warp_kernel[dtype, 1]](
            ctx,
            String(t"softmax_rows_warp_{dtype}_1"),
            warp_grid,
            1,
            1,
            _APPLE_SM_WARPS_PER_BLOCK * WARP_SIZE,
            mout,
            min_,
            Int64(rows),
            Int64(cols),
            scale,
            Int64(causal),
            Int64(q_len),
        )
    else:
        var block_grid = min(rows, 32768)
        if cols * size_of[dtype]() > _APPLE_SM_BIG_ROW_BYTES:
            if aligned:
                _enqueue_cached[_softmax_rows_block_kernel[dtype, 1024, True]](
                    ctx,
                    String(t"softmax_rows_block_{dtype}_1024_v"),
                    block_grid,
                    1,
                    1,
                    1024,
                    mout,
                    min_,
                    Int64(rows),
                    Int64(cols),
                    scale,
                    Int64(causal),
                    Int64(q_len),
                )
            else:
                _enqueue_cached[_softmax_rows_block_kernel[dtype, 1024, False]](
                    ctx,
                    String(t"softmax_rows_block_{dtype}_1024_s"),
                    block_grid,
                    1,
                    1,
                    1024,
                    mout,
                    min_,
                    Int64(rows),
                    Int64(cols),
                    scale,
                    Int64(causal),
                    Int64(q_len),
                )
        else:
            if aligned:
                _enqueue_cached[_softmax_rows_block_kernel[dtype, 256, True]](
                    ctx,
                    String(t"softmax_rows_block_{dtype}_256_v"),
                    block_grid,
                    1,
                    1,
                    256,
                    mout,
                    min_,
                    Int64(rows),
                    Int64(cols),
                    scale,
                    Int64(causal),
                    Int64(q_len),
                )
            else:
                _enqueue_cached[_softmax_rows_block_kernel[dtype, 256, False]](
                    ctx,
                    String(t"softmax_rows_block_{dtype}_256_s"),
                    block_grid,
                    1,
                    1,
                    256,
                    mout,
                    min_,
                    Int64(rows),
                    Int64(cols),
                    scale,
                    Int64(causal),
                    Int64(q_len),
                )


# ---------------------------------------------------------------------------
# Fused causal softmax + native dropout for the Apple SDPA forward: one
# launch produces the pre-dropout probabilities (saved for backward), the
# dropped/rescaled probabilities (consumed by the value BMM), and the bool
# keep-mask. The probabilities never make a DRAM round-trip between softmax
# and dropout, saving one full read of the (rows, cols) matrix plus a launch
# versus SoftmaxRows + NativeDropoutF32.
#
# RNG contract: identical to native_dropout_kernels (Philox4x32-10, element
# i belongs to block (base_offset + i // 4), lane i % 4, keep iff
# u32 < floor(Float32(1 - p) * 2^32)). cols % 4 == 0 and 16B-aligned
# pointers (host-checked) make each vec4 exactly one Philox block, so the
# mask and dropped values are byte-identical to the unfused path — dropout
# is applied across the whole row, including the zeros past the causal
# boundary, exactly like composed dropout.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_APPLE_SM_WARPS_PER_BLOCK * WARP_SIZE)
    )
)
@__name("softmax_rows_dropout_warp_f32")
def _softmax_rows_dropout_warp_kernel(
    probs_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    pdrop_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    mask_ptr: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
    scale: Float32,
    causal_arg: Int64,
    q_len_arg: Int64,
    seed: UInt64,
    base_offset: UInt64,
    threshold: UInt64,
    keep_scale: Float32,
):
    # Same structure as `_softmax_rows_warp_kernel[float32, 4]` (see there
    # for the masking rationale), plus the Philox dropout epilogue.
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var causal = Int(causal_arg)
    var q_len = Int(q_len_arg)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * _APPLE_SM_WARPS_PER_BLOCK + Int(warp_id())
    var row_stride = Int(grid_dim.x) * _APPLE_SM_WARPS_PER_BLOCK
    var n_vec = cols // 4
    while row < rows:
        var base = row * cols
        var allowed = cols
        if causal != 0:
            allowed = min(cols, row % q_len + 1)

        var vals = StaticTuple[SIMD[DType.float32, 4], _APPLE_SM_MAX_VPT]()
        var m_t = min_finite[DType.float32]()
        comptime for k in range(_APPLE_SM_MAX_VPT):
            var j0 = (lane + k * WARP_SIZE) * 4
            var x = SIMD[DType.float32, 4](min_finite[DType.float32]())
            if j0 < allowed:
                var raw = in_ptr.load[width=4, alignment=16](base + j0) * scale
                if j0 + 4 > allowed:
                    comptime for li in range(4):
                        if j0 + li < allowed:
                            x[li] = raw[li]
                else:
                    x = raw
            vals[k] = x
            m_t = max(m_t, x.reduce_max())
        var row_m = warp.max(m_t)

        var s_t = Float32(0)
        comptime for k in range(_APPLE_SM_MAX_VPT):
            var j0 = (lane + k * WARP_SIZE) * 4
            var e = SIMD[DType.float32, 4](0)
            if j0 < allowed:
                e = exp(vals[k] - row_m)
                if j0 + 4 > allowed:
                    comptime for li in range(4):
                        if j0 + li >= allowed:
                            e[li] = 0
            vals[k] = e
            s_t += e.reduce_add()
        var inv = Float32(1) / warp.sum(s_t)

        comptime for k in range(_APPLE_SM_MAX_VPT):
            var v = lane + k * WARP_SIZE
            if v < n_vec:
                var y = vals[k] * inv
                probs_ptr.store[width=4, alignment=16](base + v * 4, y)
                var rnd = _philox4x32_10(
                    base_offset + UInt64(base // 4 + v), seed
                )
                var keep_bits = (
                    rnd.cast[DType.uint64]() - SIMD[DType.uint64, 4](threshold)
                ) >> 63
                pdrop_ptr.store[width=4, alignment=16](
                    base + v * 4,
                    y * keep_bits.cast[DType.float32]() * keep_scale,
                )
                mask_ptr.bitcast[Scalar[DType.uint8]]().store[alignment=4](
                    base + v * 4, keep_bits.cast[DType.uint8]()
                )
        row += row_stride


def _softmax_rows_dropout_go(
    probs_ptr_obj: PyObjectPtr,
    pdrop_ptr_obj: PyObjectPtr,
    mask_ptr_obj: PyObjectPtr,
    in_ptr_obj: PyObjectPtr,
    rows_obj: PyObjectPtr,
    cols_obj: PyObjectPtr,
    scale_obj: PyObjectPtr,
    causal_obj: PyObjectPtr,
    q_len_obj: PyObjectPtr,
    p_obj: PyObjectPtr,
    seed_lo_obj: PyObjectPtr,
    seed_hi_obj: PyObjectPtr,
    offset_lo_obj: PyObjectPtr,
    offset_hi_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var probs_addr = _raw_int(probs_ptr_obj)
    var pdrop_addr = _raw_int(pdrop_ptr_obj)
    var mask_addr = _raw_int(mask_ptr_obj)
    var in_addr = _raw_int(in_ptr_obj)
    var rows = _raw_int(rows_obj)
    var cols = _raw_int(cols_obj)
    var scale = Float32(_raw_f64(scale_obj))
    var causal = _raw_int(causal_obj)
    var q_len = _raw_int(q_len_obj)
    var p = _raw_f64(p_obj)
    var seed = UInt64(_raw_int(seed_lo_obj)) | (
        UInt64(_raw_int(seed_hi_obj)) << 32
    )
    var base_offset = UInt64(_raw_int(offset_lo_obj)) | (
        UInt64(_raw_int(offset_hi_obj)) << 32
    )
    var ctx = _raw_ctx(device_context_ptr)

    enqueue_softmax_rows_dropout_f32(
        probs_addr,
        pdrop_addr,
        mask_addr,
        in_addr,
        rows,
        cols,
        scale,
        causal,
        q_len,
        p,
        seed,
        base_offset,
        ctx,
    )


@always_inline
def enqueue_softmax_rows_dropout_f32(
    probs_addr: Int,
    pdrop_addr: Int,
    mask_addr: Int,
    in_addr: Int,
    rows: Int,
    cols: Int,
    scale: Float32,
    causal: Int,
    q_len: Int,
    p: Float64,
    seed: UInt64,
    base_offset: UInt64,
    ctx: DeviceContext,
) raises:
    comptime if has_apple_gpu_accelerator():
        # The Python caller gates on all of this; re-checked here because the
        # dispatcher cannot report failure.
        if (
            ctx.api() == "cpu"
            or not (p > 0.0 and p < 1.0)
            or cols % 4 != 0
            or cols > WARP_SIZE * _APPLE_SM_MAX_VPT * 4
            or probs_addr % 16 != 0
            or pdrop_addr % 16 != 0
            or mask_addr % 4 != 0
            or in_addr % 16 != 0
        ):
            raise Error("SoftmaxRowsDropoutF32: unsupported configuration")
        # Same threshold arithmetic as native_dropout_kernels (Float64
        # subtraction, one narrowing, all 32 random bits compared).
        var keep_f32 = Float32(1.0 - p)
        var keep_scale = Float32(1.0) / keep_f32
        var threshold = (Float64(keep_f32) * 4294967296.0).cast[DType.uint64]()
        var warp_grid = min(
            (rows + _APPLE_SM_WARPS_PER_BLOCK - 1) // _APPLE_SM_WARPS_PER_BLOCK,
            32768,
        )
        _enqueue_cached[_softmax_rows_dropout_warp_kernel](
            ctx,
            String("softmax_rows_dropout_warp_f32"),
            warp_grid,
            1,
            1,
            _APPLE_SM_WARPS_PER_BLOCK * WARP_SIZE,
            _make_ptr[DType.float32](probs_addr).as_unsafe_any_origin(),
            _make_ptr[DType.float32](pdrop_addr).as_unsafe_any_origin(),
            _make_ptr[DType.bool](mask_addr).as_unsafe_any_origin(),
            _make_ptr[DType.float32](in_addr)
            .as_unsafe_any_origin()
            .as_immutable(),
            Int64(rows),
            Int64(cols),
            scale,
            Int64(causal),
            Int64(q_len),
            seed,
            base_offset,
            threshold,
            keep_scale,
        )
    else:
        raise Error("SoftmaxRowsDropoutF32 is Apple-GPU only")


# One warp per row, so a causal row's shorter extent costs proportionally less
# and the hardware balances the very uneven per-row work across blocks.
comptime _SM_WARPS_PER_BLOCK = 8 if WARP_SIZE <= 32 else 4
comptime _SM_BLOCK = _SM_WARPS_PER_BLOCK * WARP_SIZE
comptime _SM_MAX_GRID = 1 << 20
comptime _SM_VECTOR_BYTES = 16


@always_inline
def _softmax_warp_rows[
    dtype: DType, causal: Bool, VEC: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rows: Int,
    cols: Int,
    scale: Float32,
    q_len: Int,
):
    """Row softmax with the causal mask folded into the row extent.

    A causal row's masked columns are fed `-inf` by the reference kernel and
    come out exactly zero, so the extent `min(cols, query + 1)` carries all the
    information: this kernel reads only that prefix and writes the zeros
    directly.  The reduction is an online single-read max+sum in float32; the
    store pass re-reads the row from L1/L2.
    """
    comptime F32 = DType.float32
    comptime ALIGN = VEC * size_of[dtype]()
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * _SM_WARPS_PER_BLOCK + Int(warp_id())
    var row_stride = Int(grid_dim.x) * _SM_WARPS_PER_BLOCK

    while row < rows:
        var limit = cols
        comptime if causal:
            limit = min(cols, row % q_len + 1)
        var base = row * cols
        var vec_limit = (limit // VEC) * VEC

        # Pass 1: online max and sum over the live prefix.  The running max
        # starts at the most negative *finite* float, not at -inf: a lane or a
        # vector slot with no live column keeps the sentinel, and the rescaling
        # below subtracts two maxima, so -inf would give `-inf - -inf = NaN` and
        # poison the whole row through the warp reduction.  With a finite
        # sentinel the same expression is `exp(0) = 1` times a zero sum.
        var run_max = SIMD[F32, VEC](Float32.MIN_FINITE)
        var run_sum = SIMD[F32, VEC](0.0)
        var col = lane * VEC
        while col < vec_limit:
            var x = (
                in_ptr.load[width=VEC, alignment=ALIGN](base + col).cast[F32]()
                * scale
            )
            var new_max = max(run_max, x)
            run_sum = run_sum * exp(run_max - new_max) + exp(x - new_max)
            run_max = new_max
            col += WARP_SIZE * VEC
        # Ragged remainder: fewer than VEC columns, at most one per lane.
        var tail = vec_limit + lane
        var tail_max = Float32.MIN_FINITE
        var tail_sum = Float32(0.0)
        if tail < limit:
            tail_max = in_ptr[base + tail].cast[F32]() * scale
            tail_sum = 1.0

        # Lane-local then warp-wide combination of the (max, sum) pairs.
        var lane_max = max(run_max.reduce_max(), tail_max)
        var lane_sum = (run_sum * exp(run_max - lane_max)).reduce_add() + (
            tail_sum * exp(tail_max - lane_max)
        )
        # The row bound is warp-uniform, so every lane reaches the shuffles.
        var row_max = warp.max(lane_max)
        var inv = 1.0 / warp.sum(lane_sum * exp(lane_max - row_max))

        # Pass 2: the probabilities.  The row is L1/L2-resident from pass 1.
        col = lane * VEC
        while col < vec_limit:
            var x = (
                in_ptr.load[width=VEC, alignment=ALIGN](base + col).cast[F32]()
                * scale
            )
            out_ptr.store[width=VEC, alignment=ALIGN](
                base + col, (exp(x - row_max) * inv).cast[dtype]()
            )
            col += WARP_SIZE * VEC
        if tail < limit:
            out_ptr[base + tail] = (
                exp(in_ptr[base + tail].cast[F32]() * scale - row_max) * inv
            ).cast[dtype]()

        comptime if causal:
            var zero_head = min(cols, ceildiv(limit, VEC) * VEC)
            if limit + lane < zero_head:
                out_ptr[base + limit + lane] = Scalar[dtype](0)
            col = zero_head + lane * VEC
            while col + VEC <= cols:
                out_ptr.store[width=VEC, alignment=ALIGN](
                    base + col, SIMD[dtype, VEC](Scalar[dtype](0))
                )
                col += WARP_SIZE * VEC
            # `cols` and `zero_head` are both multiples of VEC in the wide
            # regime; VEC == 1 makes `zero_head == limit`, so nothing is left.
        row += row_stride


@__name(t"softmax_rows_warp_{dtype}_c{causal}_v{VEC}")
def _softmax_warp_kernel[
    dtype: DType, causal: Bool, VEC: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
    scale: Float32,
    q_len_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var q_len = Int(q_len_arg)
    _softmax_warp_rows[dtype, causal, VEC](
        out_ptr, in_ptr, rows, cols, scale, q_len
    )


@always_inline
def _enqueue_softmax_warp[
    dtype: DType, causal: Bool, VEC: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rows: Int,
    cols: Int,
    scale: Float32,
    q_len: Int,
    ctx: DeviceContext,
) raises:
    _enqueue_cached[_softmax_warp_kernel[dtype, causal, VEC]](
        ctx,
        String(t"softmax_rows_warp_{dtype}_c{causal}_v{VEC}"),
        min(ceildiv(rows, _SM_WARPS_PER_BLOCK), _SM_MAX_GRID),
        1,
        1,
        _SM_BLOCK,
        out_ptr,
        in_ptr,
        Int64(rows),
        Int64(cols),
        scale,
        Int64(q_len),
    )


@always_inline
def _softmax_rows[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    rows: Int,
    cols: Int,
    scale: Float32,
    causal: Int,
    q_len: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var r = Int(idx[0].value())
            var base = r * cols
            var allowed = cols
            if causal != 0:
                allowed = min(cols, r % q_len + 1)
            var m = Float32.MIN
            for j in range(allowed):
                var x = in_ptr[base + j].cast[DType.float32]() * scale
                if x > m:
                    m = x
            var denom = Float32(0)
            for j in range(allowed):
                var x = in_ptr[base + j].cast[DType.float32]() * scale
                denom += exp(x - m)
            for j in range(cols):
                if j < allowed:
                    var x = in_ptr[base + j].cast[DType.float32]() * scale
                    out_ptr[base + j] = (exp(x - m) / denom).cast[dtype]()
                else:
                    out_ptr[base + j] = Scalar[dtype](0)

        _parallel_for[func](rows, ctx)
    else:
        comptime if has_apple_gpu_accelerator():
            _softmax_rows_apple[dtype](
                out_ptr, in_ptr, rows, cols, scale, causal, q_len, ctx
            )
        elif has_accelerator():
            # Causal regime: half the score matrix is masked and the reference
            # kernel still reads, exponentiates and writes all of it.  A
            # warp-per-row kernel whose extent is the live prefix does the same
            # arithmetic over half the bytes.  Rows must be long enough for a
            # warp to vectorize and short enough that one warp per row is not
            # itself the bottleneck; both are runtime comparisons.
            comptime WIDE = _SM_VECTOR_BYTES // size_of[dtype]()
            if causal != 0 and cols <= 8192 and rows >= 256:
                var wide_ok = (
                    cols % WIDE == 0
                    and Int(out_ptr) % _SM_VECTOR_BYTES == 0
                    and Int(in_ptr) % _SM_VECTOR_BYTES == 0
                )
                if wide_ok:
                    _enqueue_softmax_warp[dtype, True, WIDE](
                        out_ptr.as_unsafe_any_origin(),
                        in_ptr.as_unsafe_any_origin().as_immutable(),
                        rows,
                        cols,
                        scale,
                        q_len,
                        ctx,
                    )
                else:
                    _enqueue_softmax_warp[dtype, True, 1](
                        out_ptr.as_unsafe_any_origin(),
                        in_ptr.as_unsafe_any_origin().as_immutable(),
                        rows,
                        cols,
                        scale,
                        q_len,
                        ctx,
                    )
                return

            @parameter
            @always_inline
            @__copy_capture(in_ptr)
            def input_fn[
                _simd_width: Int
            ](coords: Coord) -> SIMD[dtype, _simd_width]:
                var r = Int(coords[0].value())
                var c = Int(coords[1].value())
                var v = (
                    in_ptr.load[width=_simd_width](r * cols + c).cast[
                        DType.float32
                    ]()
                    * scale
                )
                if causal != 0:
                    var allowed = min(cols, r % q_len + 1)

                    @parameter
                    for lane in range(_simd_width):
                        if c + lane >= allowed:
                            v[lane] = min_or_neg_inf[DType.float32]()
                # Known trade-off: for float16 input with scale > 1 this
                # f32 -> dtype round-trip can overflow to +inf where the old
                # all-f32 kernel didn't (unreachable with the default
                # 1/sqrt(head_dim) scales).
                return v.cast[dtype]()

            softmax[dtype, 1, 2, input_fn, target="gpu"](
                Coord(rows, cols),
                TileTensor(out_ptr, row_major(rows, cols)),
                1,
                ctx,
            )
        else:
            raise Error("no GPU accelerator available at compile time")


def _softmax_rows_go(
    out_ptr_obj: PyObjectPtr,
    in_ptr_obj: PyObjectPtr,
    rows: PyObjectPtr,
    cols: PyObjectPtr,
    scale: PyObjectPtr,
    causal: PyObjectPtr,
    q_len: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var out_addr = _raw_int(out_ptr_obj)
    var in_addr = _raw_int(in_ptr_obj)
    var rows_val = _raw_int(rows)
    var cols_val = _raw_int(cols)
    var scale_val = Float32(_raw_f64(scale))
    var causal_val = _raw_int(causal)
    var q_len_val = _raw_int(q_len)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _softmax_rows[dt](
                    out_addr,
                    in_addr,
                    rows_val,
                    cols_val,
                    scale_val,
                    causal_val,
                    q_len_val,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error("unsupported dtype for fast softmax: " + String(dtype))


# ---------------------------------------------------------------------------
# Fused single-query attention (decode step): out = softmax(scale * q @ K^T)
# @ V for q_len == 1, one thread block per (batch * head). Replaces the
# bmm + softmax + bmm chain, whose m=1 GEMMs read K one row per thread
# (uncoalesced) and which costs three kernel launches plus two scratch
# buffers per call. The one-thread-block-per-row launch (with the ATTN_MAX_KV
# / ATTN_MAX_HD shared-memory caps below) is GPU only; on the CPU MAX device
# `_attn_decode_cpu` below computes the identical math with plain per-row
# loops and no size caps.
# ---------------------------------------------------------------------------

comptime ATTN_THREADS = 256
comptime ATTN_MAX_KV = 4096
comptime ATTN_MAX_HD = 256
comptime APPLE_ATTN_THREADS = 64
comptime APPLE_ATTN_MAX_KV = 256
comptime APPLE_ATTN_MAX_HD = 64


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(THREADS))
)
@__name(t"attn_decode_{dtype}_{THREADS}_{MAX_KV}_{MAX_HD}_{RED_STAGES}")
def _attn_decode_kernel[
    dtype: DType,
    THREADS: Int = ATTN_THREADS,
    MAX_KV: Int = ATTN_MAX_KV,
    MAX_HD: Int = ATTN_MAX_HD,
    RED_STAGES: Int = 8,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    q_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    kv_len_arg: Int64,
    head_dim_arg: Int64,
    scale: Float32,
    heads_arg: Int64,
    q_b_stride_arg: Int64,
    q_h_stride_arg: Int64,
    k_b_stride_arg: Int64,
    k_h_stride_arg: Int64,
    k_s_stride_arg: Int64,
    v_b_stride_arg: Int64,
    v_h_stride_arg: Int64,
    v_s_stride_arg: Int64,
):
    """out is (BH, 1, head_dim) contiguous. Q/K/V have unit head-dimension
    stride and explicit batch/head/sequence strides; this consumes both the
    fused-QKV transpose views and padded strided K/V storage without a gather.
    Scores for the block's row are staged in shared memory (hence the
    ATTN_MAX_KV cap), softmax uses f32 max/sum shared-memory tree reductions,
    and the V pass has lane d accumulate output element d so V reads coalesce
    across lanes."""
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var kv_len = Int(kv_len_arg)
    var head_dim = Int(head_dim_arg)
    var heads = Int(heads_arg)
    var q_b_stride = Int(q_b_stride_arg)
    var q_h_stride = Int(q_h_stride_arg)
    var k_b_stride = Int(k_b_stride_arg)
    var k_h_stride = Int(k_h_stride_arg)
    var k_s_stride = Int(k_s_stride_arg)
    var v_b_stride = Int(v_b_stride_arg)
    var v_h_stride = Int(v_h_stride_arg)
    var v_s_stride = Int(v_s_stride_arg)
    comptime vec_align = 4 * size_of[dtype]()
    var bh = block_idx.x
    var tid = thread_idx.x
    var out_base = bh * head_dim
    var batch = bh // heads
    var head = bh % heads
    var q_base = batch * q_b_stride + head * q_h_stride
    var k_base = batch * k_b_stride + head * k_h_stride
    var v_base = batch * v_b_stride + head * v_h_stride

    var q_smem = stack_allocation[
        MAX_HD, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var s_smem = stack_allocation[
        MAX_KV, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var red = stack_allocation[
        THREADS, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var bcast = stack_allocation[
        2, DType.float32, address_space=AddressSpace.SHARED
    ]()

    for d in range(tid, head_dim, THREADS):
        q_smem[d] = q_ptr[q_base + d].cast[DType.float32]()
    barrier()

    var m = Float32.MIN
    for j in range(tid, kv_len, THREADS):
        var krow = k_base + j * k_s_stride
        var acc = Float32(0)
        for d in range(0, head_dim, 4):
            var k4 = k_ptr.load[width=4, alignment=vec_align](krow + d).cast[
                DType.float32
            ]()
            var q4 = q_smem.load[width=4, alignment=16](d)
            acc += (q4 * k4).reduce_add()
        var s = acc * scale
        s_smem[j] = s
        if s > m:
            m = s
    red[tid] = m
    barrier()
    var stride = THREADS // 2
    for _ in range(RED_STAGES):
        if tid < stride:
            if red[tid + stride] > red[tid]:
                red[tid] = red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        bcast[0] = red[0]
    barrier()
    m = bcast[0]

    var s = Float32(0)
    for j in range(tid, kv_len, THREADS):
        var e = exp(s_smem[j] - m)
        s_smem[j] = e
        s += e
    red[tid] = s
    barrier()
    stride = THREADS // 2
    for _ in range(RED_STAGES):
        if tid < stride:
            red[tid] += red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        bcast[1] = red[0]
    barrier()
    var inv_denom = 1.0 / bcast[1]

    # GPT-2's D=64 matches an AMD wavefront. Partition the V reduction over
    # all four wavefronts so the bandwidth-heavy pass uses the whole block,
    # then combine four partial vectors in the now-free reduction scratch.
    # The Apple specialization instantiates this kernel with 64 threads, so
    # keep the four-wavefront reduction entirely out of non-gfx942 builds.
    comptime if _accelerator_arch() == "amdgpu:gfx942":
        if THREADS == 256 and head_dim == 64:
            var lane = tid % 64
            var wave = tid // 64
            var acc = Float32(0)
            for j in range(wave, kv_len, 4):
                acc += (
                    s_smem[j]
                    * v_ptr[v_base + j * v_s_stride + lane].cast[
                        DType.float32
                    ]()
                )
            red[tid] = acc
            barrier()
            if wave == 0:
                acc = (
                    red[lane]
                    + red[64 + lane]
                    + red[128 + lane]
                    + red[192 + lane]
                )
                out_ptr[out_base + lane] = (acc * inv_denom).cast[dtype]()
            return

    for d in range(tid, head_dim, THREADS):
        var acc = Float32(0)
        for j in range(kv_len):
            acc += (
                s_smem[j]
                * v_ptr[v_base + j * v_s_stride + d].cast[DType.float32]()
            )
        out_ptr[out_base + d] = (acc * inv_denom).cast[dtype]()


@always_inline
def _attn_decode_cpu[
    dtype: DType
](
    out_addr: Int,
    q_addr: Int,
    k_addr: Int,
    v_addr: Int,
    bh: Int,
    kv_len: Int,
    head_dim: Int,
    scale: Float32,
    heads: Int,
    q_b_stride: Int,
    q_h_stride: Int,
    k_b_stride: Int,
    k_h_stride: Int,
    k_s_stride: Int,
    v_b_stride: Int,
    v_h_stride: Int,
    v_s_stride: Int,
    ctx: DeviceContext,
) raises:
    """Same math as `_attn_decode_kernel`, one sequential task per (batch *
    head) row: a max pass over the scores followed by a fused exp-sum /
    weighted-V pass. `acc_out` is a per-row float32 scratch buffer (dynamic
    size, so it cannot live in `stack_allocation` like the GPU version's
    shared memory) that accumulates the output in the same precision the
    GPU kernel uses before the final per-dtype cast.
    """
    var out_ptr = _make_ptr[dtype](out_addr)
    var q_ptr = _make_ptr[dtype](q_addr)
    var k_ptr = _make_ptr[dtype](k_addr)
    var v_ptr = _make_ptr[dtype](v_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, q_ptr, k_ptr, v_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var out_base = i * head_dim
        var batch = i // heads
        var head = i % heads
        var q_base = batch * q_b_stride + head * q_h_stride
        var k_base = batch * k_b_stride + head * k_h_stride
        var v_base = batch * v_b_stride + head * v_h_stride

        var m = Float32.MIN
        for j in range(kv_len):
            var krow = k_base + j * k_s_stride
            var dot = Float32(0)
            for d in range(head_dim):
                dot += (
                    q_ptr[q_base + d].cast[DType.float32]()
                    * k_ptr[krow + d].cast[DType.float32]()
                )
            var s = dot * scale
            if s > m:
                m = s

        var acc_out = alloc[Float32](head_dim)
        for d in range(head_dim):
            acc_out[d] = Float32(0)
        var denom = Float32(0)
        for j in range(kv_len):
            var krow = k_base + j * k_s_stride
            var dot = Float32(0)
            for d in range(head_dim):
                dot += (
                    q_ptr[q_base + d].cast[DType.float32]()
                    * k_ptr[krow + d].cast[DType.float32]()
                )
            var s = dot * scale
            var p = exp(s - m)
            denom += p
            var vrow = v_base + j * v_s_stride
            for d in range(head_dim):
                acc_out[d] += p * v_ptr[vrow + d].cast[DType.float32]()

        for d in range(head_dim):
            out_ptr[out_base + d] = (acc_out[d] / denom).cast[dtype]()
        acc_out.free()

    # CPU-only launch: `func` uses a host `alloc()`/`free()` for its per-row
    # scratch, and `_parallel_for` would also compile a `target="gpu"`
    # instantiation of it. That device instantiation pulls host malloc/free
    # into the GPU binary -- a no-op on NVIDIA (device malloc exists) but a
    # link failure on AMDGPU ("undefined symbol: malloc"). This kernel is only
    # ever reached with a CPU context (see `_attn_decode`), so emit only the
    # CPU form and the GPU instantiation is never generated on any platform.
    elementwise[func, simd_width=1](Coord(bh), ctx)


@always_inline
def _attn_decode[
    dtype: DType
](
    out_addr: Int,
    q_addr: Int,
    k_addr: Int,
    v_addr: Int,
    bh: Int,
    kv_len: Int,
    head_dim: Int,
    scale: Float32,
    heads: Int,
    q_b_stride: Int,
    q_h_stride: Int,
    k_b_stride: Int,
    k_h_stride: Int,
    k_s_stride: Int,
    v_b_stride: Int,
    v_h_stride: Int,
    v_s_stride: Int,
    ctx: DeviceContext,
) raises:
    if ctx.api() == "cpu":
        _attn_decode_cpu[dtype](
            out_addr,
            q_addr,
            k_addr,
            v_addr,
            bh,
            kv_len,
            head_dim,
            scale,
            heads,
            q_b_stride,
            q_h_stride,
            k_b_stride,
            k_h_stride,
            k_s_stride,
            v_b_stride,
            v_h_stride,
            v_s_stride,
            ctx,
        )
        return
    comptime if has_accelerator():
        comptime if has_apple_gpu_accelerator():
            if kv_len <= APPLE_ATTN_MAX_KV and head_dim <= APPLE_ATTN_MAX_HD:
                _enqueue_cached[
                    _attn_decode_kernel[
                        dtype,
                        APPLE_ATTN_THREADS,
                        APPLE_ATTN_MAX_KV,
                        APPLE_ATTN_MAX_HD,
                        6,
                    ]
                ](
                    ctx,
                    String(t"attn_decode_apple_short_{dtype}"),
                    bh,
                    1,
                    1,
                    APPLE_ATTN_THREADS,
                    _make_ptr[dtype](out_addr).as_unsafe_any_origin(),
                    _make_ptr[dtype](q_addr)
                    .as_unsafe_any_origin()
                    .as_immutable(),
                    _make_ptr[dtype](k_addr)
                    .as_unsafe_any_origin()
                    .as_immutable(),
                    _make_ptr[dtype](v_addr)
                    .as_unsafe_any_origin()
                    .as_immutable(),
                    Int64(kv_len),
                    Int64(head_dim),
                    scale,
                    Int64(heads),
                    Int64(q_b_stride),
                    Int64(q_h_stride),
                    Int64(k_b_stride),
                    Int64(k_h_stride),
                    Int64(k_s_stride),
                    Int64(v_b_stride),
                    Int64(v_h_stride),
                    Int64(v_s_stride),
                )
                return
        _enqueue_cached[_attn_decode_kernel[dtype]](
            ctx,
            String(t"attn_decode_{dtype}"),
            bh,
            1,
            1,
            ATTN_THREADS,
            _make_ptr[dtype](out_addr).as_unsafe_any_origin(),
            _make_ptr[dtype](q_addr).as_unsafe_any_origin().as_immutable(),
            _make_ptr[dtype](k_addr).as_unsafe_any_origin().as_immutable(),
            _make_ptr[dtype](v_addr).as_unsafe_any_origin().as_immutable(),
            Int64(kv_len),
            Int64(head_dim),
            scale,
            Int64(heads),
            Int64(q_b_stride),
            Int64(q_h_stride),
            Int64(k_b_stride),
            Int64(k_h_stride),
            Int64(k_s_stride),
            Int64(v_b_stride),
            Int64(v_h_stride),
            Int64(v_s_stride),
        )
    else:
        raise Error("no GPU accelerator available at compile time")


# ---------------------------------------------------------------------------
# Mean over the trailing dims: input viewed as (rows, cols), out has `rows`
# elements. Covers aten.mean.dim over the last dims (e.g. global avg pool).
# ---------------------------------------------------------------------------


@always_inline
def _mean_rows[
    dtype: DType
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    """Row-wise mean over the trailing axis, backed by the stdlib reduction
    library (`std.algorithm.reduction.mean`). Floats accumulate in float32
    (the closures cast raw<->f32); the library applies the 1/cols scaling.
    The library picks the GPU launch tier, so full reductions (rows == 1) fan
    out across the device instead of the single sequential thread the old
    per-row kernel used."""
    comptime acc_dtype = DType.float32
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @parameter
    @__copy_capture(in_ptr)
    def input_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[acc_dtype, width]:
        var flat = coords[0] * cols + coords[1]
        return in_ptr.load[width=width](flat).cast[acc_dtype]()

    @always_inline
    @parameter
    @__copy_capture(out_ptr)
    def output_fn[
        width: SIMDLength, rank: Int
    ](coords: IndexList[rank], val: SIMD[acc_dtype, width]):
        out_ptr[coords[0]] = val[0].cast[dtype]()

    var in_shape = IndexList[2](rows, cols)
    var out_shape = IndexList[2](rows, 1)

    @always_inline
    @parameter
    def run[target: StaticString]() raises:
        mean[acc_dtype, input_fn, output_fn, target=target, reduce_dim=1](
            Coord(in_shape), Coord(out_shape), ctx
        )

    if ctx.api() == "cpu":
        run["cpu"]()
    else:
        comptime if has_accelerator():
            run["gpu"]()
        else:
            raise Error("no GPU accelerator available at compile time")


# ---------------------------------------------------------------------------
# Max pool 2D over NCHW contiguous input, with indices (torch semantics:
# index of the max within the flattened H*W input plane, int64).
# `planes` is N * C; one parallel task per output element.
# ---------------------------------------------------------------------------


@always_inline
def _max_pool2d[
    dtype: DType
](
    out_addr: Int,
    idx_addr: Int,
    in_addr: Int,
    in_h: Int,
    in_w: Int,
    out_h: Int,
    out_w: Int,
    kh: Int,
    kw: Int,
    stride_h: Int,
    stride_w: Int,
    pad_h: Int,
    pad_w: Int,
    dil_h: Int,
    dil_w: Int,
    planes: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var idx_ptr = _make_ptr[DType.int64](idx_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, idx_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var ow = i % out_w
        var oh = (i // out_w) % out_h
        var plane = i // (out_w * out_h)
        var in_base = plane * in_h * in_w
        var best = min_or_neg_inf[dtype]()
        var best_idx = 0
        for fh in range(kh):
            var ih = oh * stride_h - pad_h + fh * dil_h
            if ih < 0 or ih >= in_h:
                continue
            for fw in range(kw):
                var iw = ow * stride_w - pad_w + fw * dil_w
                if iw < 0 or iw >= in_w:
                    continue
                var v = in_ptr[in_base + ih * in_w + iw]
                if v > best:
                    best = v
                    best_idx = ih * in_w + iw
        out_ptr[i] = best
        idx_ptr[i] = Int64(best_idx)

    _parallel_for[func](planes * out_h * out_w, ctx)


def _max_pool2d_go(
    out_ptr_obj: PyObjectPtr,
    idx_ptr_obj: PyObjectPtr,
    in_ptr_obj: PyObjectPtr,
    params: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var out_addr = _raw_int(out_ptr_obj)
    var idx_addr = _raw_int(idx_ptr_obj)
    var in_addr = _raw_int(in_ptr_obj)
    var in_h = _raw_tuple_int(params, 0)
    var in_w = _raw_tuple_int(params, 1)
    var out_h = _raw_tuple_int(params, 2)
    var out_w = _raw_tuple_int(params, 3)
    var kh = _raw_tuple_int(params, 4)
    var kw = _raw_tuple_int(params, 5)
    var stride_h = _raw_tuple_int(params, 6)
    var stride_w = _raw_tuple_int(params, 7)
    var pad_h = _raw_tuple_int(params, 8)
    var pad_w = _raw_tuple_int(params, 9)
    var dil_h = _raw_tuple_int(params, 10)
    var dil_w = _raw_tuple_int(params, 11)
    var planes = _raw_tuple_int(params, 12)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _max_pool2d[dt](
                    out_addr,
                    idx_addr,
                    in_addr,
                    in_h,
                    in_w,
                    out_h,
                    out_w,
                    kh,
                    kw,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    dil_h,
                    dil_w,
                    planes,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error("unsupported dtype for fast max_pool2d: " + String(dtype))


# ---------------------------------------------------------------------------
# Embedding lookup: out[i] = weight[indices[i // row_len] * row_len +
# i % row_len]. This is gather along dim 0 of a 2D weight table.
# ---------------------------------------------------------------------------


@always_inline
def _gather0[
    dtype: DType, idx_dtype: DType
](
    out_addr: Int,
    weight_addr: Int,
    indices_addr: Int,
    num_indices: Int,
    row_len: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var weight_ptr = _make_ptr[dtype](weight_addr)
    var indices_ptr = _make_ptr[idx_dtype](indices_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, weight_ptr, indices_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var row = Int(indices_ptr[i // row_len])
        out_ptr[i] = weight_ptr[row * row_len + i % row_len]

    _parallel_for[func](num_indices * row_len, ctx)


@always_inline
def _gather0_data_dispatch[
    idx_dtype: DType
](
    dtype: DType,
    out_addr: Int,
    weight_addr: Int,
    indices_addr: Int,
    num_indices: Int,
    row_len: Int,
    ctx: DeviceContext,
) raises:
    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _gather0[dt, idx_dtype](
                    out_addr,
                    weight_addr,
                    indices_addr,
                    num_indices,
                    row_len,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error("unsupported dtype for fast embedding: " + String(dtype))


def _gather0_go(
    out_ptr_obj: PyObjectPtr,
    weight_ptr_obj: PyObjectPtr,
    indices_ptr_obj: PyObjectPtr,
    idx_dtype_obj: PyObjectPtr,
    num_indices: PyObjectPtr,
    row_len: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var idx_dtype = _raw_dtype_int(idx_dtype_obj)
    var out_addr = _raw_int(out_ptr_obj)
    var weight_addr = _raw_int(weight_ptr_obj)
    var indices_addr = _raw_int(indices_ptr_obj)
    var num_indices_val = _raw_int(num_indices)
    var row_len_val = _raw_int(row_len)
    var ctx = _raw_ctx(device_context_ptr)

    comptime if _dtype_arg_on[1, DType.int64]():
        if idx_dtype != DType.int64:
            raise Error("embedding specialization/index dtype mismatch")
        _gather0_data_dispatch[DType.int64](
            dtype,
            out_addr,
            weight_addr,
            indices_addr,
            num_indices_val,
            row_len_val,
            ctx,
        )
    elif _dtype_arg_on[1, DType.int32]():
        if idx_dtype != DType.int32:
            raise Error("embedding specialization/index dtype mismatch")
        _gather0_data_dispatch[DType.int32](
            dtype,
            out_addr,
            weight_addr,
            indices_addr,
            num_indices_val,
            row_len_val,
            ctx,
        )
    else:
        raise Error(
            "unsupported index dtype for fast embedding: " + String(idx_dtype)
        )


# ---------------------------------------------------------------------------
# Full any()/all() over a bool tensor -> scalar bool. Viewed as one row of
# `size` bools reduced over the trailing axis (any = max / all = min over bool).
# aten caps this path at < 4.2M elements, so it is almost always the small
# regime; a single 256-thread block (strided scan + shared-memory AND/OR tree,
# no per-call allocation) handles it, which matters because the HF sdpa mask
# check runs this every decode step. Only the rare huge case (>= 2^20) is handed
# to the stdlib library's two-phase multi-block tier.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name(t"bool_full_block_{is_all}")
def _bool_full_block_kernel[
    is_all: Bool
](
    out_ptr: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[DType.bool], ImmutAnyOrigin],
    size_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var size = Int(size_arg)
    var tid = Int(thread_idx.x)
    var acc = is_all
    for j in range(tid, size, ROWRED_THREADS):
        comptime if is_all:
            if not in_ptr[j]:
                acc = False
        else:
            if in_ptr[j]:
                acc = True
    var red = stack_allocation[
        ROWRED_THREADS, DType.bool, address_space=AddressSpace.SHARED
    ]()
    red[tid] = acc
    barrier()
    comptime for stage in range(ROWRED_STAGES):
        comptime half = ROWRED_THREADS >> (stage + 1)
        if tid < half:
            comptime if is_all:
                red[tid] = red[tid] and red[tid + half]
            else:
                red[tid] = red[tid] or red[tid + half]
        barrier()
    if tid == 0:
        out_ptr[0] = red[0]


@always_inline
def _bool_full_reduce[
    is_all: Bool
](out_addr: Int, in_addr: Int, size: Int, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[DType.bool](out_addr)
    var in_ptr = _make_ptr[DType.bool](in_addr)

    @always_inline
    @parameter
    @__copy_capture(in_ptr)
    def input_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[DType.bool, width]:
        return in_ptr.load[width=width](coords[0] * size + coords[1])

    @always_inline
    @parameter
    @__copy_capture(out_ptr)
    def output_fn[
        width: SIMDLength, rank: Int
    ](coords: IndexList[rank], val: SIMD[DType.bool, width]):
        out_ptr[coords[0]] = val[0]

    var shape = IndexList[2](1, size)

    @always_inline
    @parameter
    def run[target: StaticString]() raises:
        comptime if is_all:
            reduce_min[
                DType.bool, input_fn, output_fn, target=target, reduce_dim=1
            ](Coord(shape), ctx)
        else:
            reduce_max[
                DType.bool, input_fn, output_fn, target=target, reduce_dim=1
            ](Coord(shape), ctx)

    if ctx.api() == "cpu":
        run["cpu"]()
    else:
        comptime if has_accelerator():
            if _use_library_reduce(1, size):
                run["gpu"]()
            else:
                _enqueue_cached[_bool_full_block_kernel[is_all]](
                    ctx,
                    String(t"bool_full_block_{is_all}"),
                    1,
                    1,
                    1,
                    ROWRED_THREADS,
                    out_ptr.as_unsafe_any_origin(),
                    in_ptr.as_unsafe_any_origin().as_immutable(),
                    Int64(size),
                )
        else:
            raise Error("no GPU accelerator available at compile time")


def _all_bool(
    out_addr: Int, in_addr: Int, size: Int, ctx: DeviceContext
) raises:
    """Full all() over a bool tensor -> scalar bool (AND). Single-block for the
    common small case, stdlib library for the rare huge case (`_bool_full_reduce`).
    """
    _bool_full_reduce[is_all=True](out_addr, in_addr, size, ctx)


def _all_bool_go(
    out_ptr_obj: PyObjectPtr,
    in_ptr_obj: PyObjectPtr,
    size: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    _all_bool(
        _raw_int(out_ptr_obj),
        _raw_int(in_ptr_obj),
        _raw_int(size),
        _raw_ctx(device_context_ptr),
    )


# ---------------------------------------------------------------------------
# any() over a bool tensor -> scalar bool: same block reduction as all().
# ---------------------------------------------------------------------------


@always_inline
def _any_bool(
    out_addr: Int, in_addr: Int, size: Int, ctx: DeviceContext
) raises:
    """Full any() over a bool tensor -> scalar bool (OR). Single-block for the
    common small case, stdlib library for the rare huge case (`_bool_full_reduce`).
    """
    _bool_full_reduce[is_all=False](out_addr, in_addr, size, ctx)


def _any_bool_go(
    out_ptr_obj: PyObjectPtr,
    in_ptr_obj: PyObjectPtr,
    size: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    _any_bool(
        _raw_int(out_ptr_obj),
        _raw_int(in_ptr_obj),
        _raw_int(size),
        _raw_ctx(device_context_ptr),
    )


# ---------------------------------------------------------------------------
# Row-wise argmax: input viewed as (rows, cols), out is `rows` int64
# indices (first occurrence wins, matching torch). Covers argmax over the
# vocab dim in greedy decoding, where rows=1 and cols can be > 50000 —
# a single sequential task per row would leave the GPU almost idle, so the
# GPU path launches one thread block per row and reduces across the row in
# parallel. CPU keeps the original single-task-per-row scalar scan.
# ---------------------------------------------------------------------------

comptime ARGMAX_THREADS = 512
# log2(ARGMAX_THREADS): number of halving steps in the shared-memory
# reduction tree below.
comptime ARGMAX_STAGES = 9


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ARGMAX_THREADS))
)
@__name(t"argmax_rows_block_{dtype}")
def _argmax_rows_block_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols_arg: Int64,
):
    """One block per row (grid.x = rows); ARGMAX_THREADS lanes each stride
    over the row picking their own best (value, index) with strict `>` (so
    ties keep the lane's earliest index), then a shared-memory tree
    reduction combines lanes with an explicit lower-index tiebreak on equal
    values. Together these preserve torch's first-occurrence-wins argmax
    semantics regardless of how work is split across lanes.
    """
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var cols = Int(cols_arg)
    var r = block_idx.x
    var tid = thread_idx.x
    var base = r * cols

    var best_val = min_or_neg_inf[dtype]()
    var best_idx = Int64(-1)
    for j in range(tid, cols, ARGMAX_THREADS):
        var v = in_ptr[base + j]
        if v > best_val:
            best_val = v
            best_idx = Int64(j)

    var val_smem = stack_allocation[
        ARGMAX_THREADS, dtype, address_space=AddressSpace.SHARED
    ]()
    var idx_smem = stack_allocation[
        ARGMAX_THREADS, DType.int64, address_space=AddressSpace.SHARED
    ]()
    val_smem[tid] = best_val
    idx_smem[tid] = best_idx
    barrier()

    var stride = ARGMAX_THREADS // 2
    for _ in range(ARGMAX_STAGES):
        if tid < stride:
            var other_val = val_smem[tid + stride]
            var other_idx = idx_smem[tid + stride]
            var cur_val = val_smem[tid]
            var cur_idx = idx_smem[tid]
            if other_val > cur_val or (
                other_val == cur_val
                and other_idx != Int64(-1)
                and (cur_idx == Int64(-1) or other_idx < cur_idx)
            ):
                val_smem[tid] = other_val
                idx_smem[tid] = other_idx
        barrier()
        stride //= 2

    if tid == 0:
        out_ptr[r] = idx_smem[0]


@always_inline
def _argmax_rows[
    dtype: DType
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[DType.int64](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var r = Int(idx[0].value())
            var base = r * cols
            var best = in_ptr[base]
            var best_idx = 0
            for j in range(1, cols):
                var v = in_ptr[base + j]
                if v > best:
                    best = v
                    best_idx = j
            out_ptr[r] = Int64(best_idx)

        _parallel_for[func](rows, ctx)
    else:
        comptime if has_accelerator():
            var out_p = out_ptr.as_unsafe_any_origin()
            var in_p = in_ptr.as_unsafe_any_origin().as_immutable()
            _enqueue_cached[_argmax_rows_block_kernel[dtype]](
                ctx,
                String(t"argmax_rows_{dtype}"),
                rows,
                1,
                1,
                ARGMAX_THREADS,
                out_p,
                in_p,
                Int64(cols),
            )
        else:
            raise Error("no GPU accelerator available at compile time")


# ---------------------------------------------------------------------------
# Row-wise max reduction (values only): input viewed as (rows, cols), out
# has `rows` elements of the same dtype. rows=1 covers aten.max() (no dim).
# Routes the under-saturated, huge-col regime to the stdlib library (two-phase
# fan-out) and everything else to a 256-thread block-per-row kernel; the decode
# loop calls this every step with a tiny (1, batch) input, where the library's
# per-call scratch allocation would be pure overhead.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name(t"max_rows_block_{dtype}")
def _max_rows_block_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cols_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var cols = Int(cols_arg)
    comptime acc_dtype = _accum_dtype[dtype]()
    var r = block_idx.x
    var tid = thread_idx.x
    var base = r * cols

    var acc = min_or_neg_inf[acc_dtype]()
    for j in range(tid, cols, ROWRED_THREADS):
        var v = in_ptr[base + j].cast[acc_dtype]()
        if v > acc:
            acc = v

    var red = stack_allocation[
        ROWRED_THREADS, acc_dtype, address_space=AddressSpace.SHARED
    ]()
    red[tid] = acc
    barrier()
    var stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            if red[tid + stride] > red[tid]:
                red[tid] = red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        out_ptr[r] = red[0].cast[dtype]()


@always_inline
def _max_rows[
    dtype: DType
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    """Row-wise max (values only). rows == 1 covers aten.max() (full reduce).
    Floats reduce in float32 (max is exact under the cast)."""
    comptime acc_dtype = _accum_dtype[dtype]()
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @parameter
    @__copy_capture(in_ptr)
    def input_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[acc_dtype, width]:
        var flat = coords[0] * cols + coords[1]
        return in_ptr.load[width=width](flat).cast[acc_dtype]()

    @always_inline
    @parameter
    @__copy_capture(out_ptr)
    def output_fn[
        width: SIMDLength, rank: Int
    ](coords: IndexList[rank], val: SIMD[acc_dtype, width]):
        out_ptr[coords[0]] = val[0].cast[dtype]()

    var shape = IndexList[2](rows, cols)

    @always_inline
    @parameter
    def run[target: StaticString]() raises:
        reduce_max[acc_dtype, input_fn, output_fn, target=target, reduce_dim=1](
            Coord(shape), ctx
        )

    if ctx.api() == "cpu":
        run["cpu"]()
    else:
        comptime if has_accelerator():
            if _use_library_reduce(rows, cols):
                run["gpu"]()
            else:
                _enqueue_cached[_max_rows_block_kernel[dtype]](
                    ctx,
                    String(t"max_rows_{dtype}"),
                    rows,
                    1,
                    1,
                    ROWRED_THREADS,
                    out_ptr.as_unsafe_any_origin(),
                    in_ptr.as_unsafe_any_origin().as_immutable(),
                    Int64(cols),
                )
        else:
            raise Error("no GPU accelerator available at compile time")


# ---------------------------------------------------------------------------
# Row-wise cumulative sum along the last dim: input viewed as (rows, cols).
# One sequential task per row — used on the small int tensors of the
# generation loop (position ids from attention-mask cumsum).
# ---------------------------------------------------------------------------


@always_inline
def _cumsum_rows[
    dtype: DType
](out_addr: Int, in_addr: Int, rows: Int, cols: Int, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var r = Int(idx[0].value())
        var base = r * cols
        var total = Scalar[dtype](0)
        for j in range(cols):
            total += in_ptr[base + j]
            out_ptr[base + j] = total

    _parallel_for[func](rows, ctx)


# ---------------------------------------------------------------------------
# Average pool 2D over NCHW contiguous input (torch semantics). The window at
# output (oh, ow) covers input rows [oh*sh - ph, ...) intersected with the real
# input; the divisor honors count_include_pad / divisor_override exactly as
# aten's cpu_avg_pool2d does. ceil_mode is handled Python-side (only False is
# passed here). One parallel task per output element (CPU and GPU).
# ---------------------------------------------------------------------------


@always_inline
def _avg_pool2d[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    in_h: Int,
    in_w: Int,
    out_h: Int,
    out_w: Int,
    kh: Int,
    kw: Int,
    stride_h: Int,
    stride_w: Int,
    pad_h: Int,
    pad_w: Int,
    count_include_pad: Int,
    divisor_override: Int,
    planes: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var ow = i % out_w
        var oh = (i // out_w) % out_h
        var plane = i // (out_w * out_h)
        var in_base = plane * in_h * in_w

        # Window in (possibly padded) coordinates; pool_size uses the padded
        # extent (before clamping to the real input), matching torch.
        var ih0 = oh * stride_h - pad_h
        var iw0 = ow * stride_w - pad_w
        var ih1 = min(ih0 + kh, in_h + pad_h)
        var iw1 = min(iw0 + kw, in_w + pad_w)
        var pool_size = (ih1 - ih0) * (iw1 - iw0)
        ih0 = max(ih0, 0)
        iw0 = max(iw0, 0)
        ih1 = min(ih1, in_h)
        iw1 = min(iw1, in_w)

        if ih0 >= ih1 or iw0 >= iw1:
            # Window entirely in padding: torch leaves the output at 0.
            out_ptr[i] = Scalar[dtype](0)
        else:
            var divide_factor = 0
            if divisor_override != 0:
                divide_factor = divisor_override
            elif count_include_pad != 0:
                divide_factor = pool_size
            else:
                divide_factor = (ih1 - ih0) * (iw1 - iw0)
            var total = Float32(0)
            for ih in range(ih0, ih1):
                var row = in_base + ih * in_w
                for iw in range(iw0, iw1):
                    total += in_ptr[row + iw].cast[DType.float32]()
            out_ptr[i] = (total / Float32(divide_factor)).cast[dtype]()

    _parallel_for[func](planes * out_h * out_w, ctx)


def _avg_pool2d_go(
    out_ptr_obj: PyObjectPtr,
    in_ptr_obj: PyObjectPtr,
    params: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var out_addr = _raw_int(out_ptr_obj)
    var in_addr = _raw_int(in_ptr_obj)
    var in_h = _raw_tuple_int(params, 0)
    var in_w = _raw_tuple_int(params, 1)
    var out_h = _raw_tuple_int(params, 2)
    var out_w = _raw_tuple_int(params, 3)
    var kh = _raw_tuple_int(params, 4)
    var kw = _raw_tuple_int(params, 5)
    var stride_h = _raw_tuple_int(params, 6)
    var stride_w = _raw_tuple_int(params, 7)
    var pad_h = _raw_tuple_int(params, 8)
    var pad_w = _raw_tuple_int(params, 9)
    var count_include_pad = _raw_tuple_int(params, 10)
    var divisor_override = _raw_tuple_int(params, 11)
    var planes = _raw_tuple_int(params, 12)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _avg_pool2d[dt](
                    out_addr,
                    in_addr,
                    in_h,
                    in_w,
                    out_h,
                    out_w,
                    kh,
                    kw,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    count_include_pad,
                    divisor_override,
                    planes,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error("unsupported dtype for fast avg_pool2d: " + String(dtype))


# ---------------------------------------------------------------------------
# Adaptive average pool 2D over NCHW contiguous input. For output cell
# (oh, ow) the input window is [start(oh), end(oh)) x [start(ow), end(ow))
# with torch's integer start/end index formulas; the divisor is the window
# area (no padding). One parallel task per output element (CPU and GPU).
# ---------------------------------------------------------------------------


@always_inline
def _adaptive_avg_pool2d[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    in_h: Int,
    in_w: Int,
    out_h: Int,
    out_w: Int,
    planes: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var ow = i % out_w
        var oh = (i // out_w) % out_h
        var plane = i // (out_w * out_h)
        var in_base = plane * in_h * in_w

        # start_index(a, b, c) = (a // b) * c + ((a % b) * c) // b
        # end_index(a, b, c)   = 1 + ((a + 1) * c - 1) // b
        var ih0 = (oh // out_h) * in_h + ((oh % out_h) * in_h) // out_h
        var ih1 = 1 + ((oh + 1) * in_h - 1) // out_h
        var iw0 = (ow // out_w) * in_w + ((ow % out_w) * in_w) // out_w
        var iw1 = 1 + ((ow + 1) * in_w - 1) // out_w
        var area = (ih1 - ih0) * (iw1 - iw0)

        var total = Float32(0)
        for ih in range(ih0, ih1):
            var row = in_base + ih * in_w
            for iw in range(iw0, iw1):
                total += in_ptr[row + iw].cast[DType.float32]()
        out_ptr[i] = (total / Float32(area)).cast[dtype]()

    _parallel_for[func](planes * out_h * out_w, ctx)


def _adaptive_avg_pool2d_go(
    out_ptr_obj: PyObjectPtr,
    in_ptr_obj: PyObjectPtr,
    params: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var out_addr = _raw_int(out_ptr_obj)
    var in_addr = _raw_int(in_ptr_obj)
    var in_h = _raw_tuple_int(params, 0)
    var in_w = _raw_tuple_int(params, 1)
    var out_h = _raw_tuple_int(params, 2)
    var out_w = _raw_tuple_int(params, 3)
    var planes = _raw_tuple_int(params, 4)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _adaptive_avg_pool2d[dt](
                    out_addr, in_addr, in_h, in_w, out_h, out_w, planes, ctx
                )
                handled = True
    if not handled:
        raise Error(
            "unsupported dtype for fast adaptive_avg_pool2d: " + String(dtype)
        )


# ---------------------------------------------------------------------------
# Group norm: normalize each (sample, group) over its C/group channels AND all
# spatial elements together, then apply the per-channel affine. Input is
# viewed as (rows = N*group, cols = (C/group) * HxW) — the group's elements are
# a contiguous row because the input is NC(HxW) contiguous. Same two-pass
# mean/variance as layer norm; mean/rstd are float32, shape (N, group). The GPU
# path launches one block per row (row counts are small: N*group); the CPU
# path runs one task per row.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ROWRED_THREADS))
)
@__name(t"group_norm_block_{dtype}")
def _group_norm_block_kernel[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    mean_out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rstd_out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    beta_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    eps: Float32,
    cols_arg: Int64,
    hxw_arg: Int64,
    group_arg: Int64,
    cpg_arg: Int64,
):
    """One block per (sample, group) row (grid.x = N*group). Same shared-memory
    mean/variance reduction as `_layer_norm_block_kernel`; the affine is applied
    per channel, where channel = (row % group) * cpg + (position // hxw)."""
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var cols = Int(cols_arg)
    var hxw = Int(hxw_arg)
    var group = Int(group_arg)
    var cpg = Int(cpg_arg)
    var r = block_idx.x
    var tid = thread_idx.x
    var g = Int(r) % group
    var base = r * cols

    var red = stack_allocation[
        ROWRED_THREADS, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var bcast = stack_allocation[
        2, DType.float32, address_space=AddressSpace.SHARED
    ]()

    var s = Float32(0)
    for j in range(tid, cols, ROWRED_THREADS):
        s += in_ptr[base + j].cast[DType.float32]()
    red[tid] = s
    barrier()
    var stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            red[tid] += red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        bcast[0] = red[0] / Float32(cols)
    barrier()
    var mean = bcast[0]

    var vs = Float32(0)
    for j in range(tid, cols, ROWRED_THREADS):
        var d = in_ptr[base + j].cast[DType.float32]() - mean
        vs += d * d
    red[tid] = vs
    barrier()
    stride = ROWRED_THREADS // 2
    for _ in range(ROWRED_STAGES):
        if tid < stride:
            red[tid] += red[tid + stride]
        barrier()
        stride //= 2
    if tid == 0:
        var rstd0 = 1.0 / ieee_sqrt(red[0] / Float32(cols) + eps)
        bcast[1] = rstd0
        mean_out_ptr[r] = mean
        rstd_out_ptr[r] = rstd0
    barrier()
    var rstd = bcast[1]

    for j in range(tid, cols, ROWRED_THREADS):
        var c = g * cpg + j // hxw
        var x = in_ptr[base + j].cast[DType.float32]()
        var gm = gamma_ptr[c].cast[DType.float32]()
        var bt = beta_ptr[c].cast[DType.float32]()
        out_ptr[base + j] = ((x - mean) * rstd * gm + bt).cast[dtype]()


@always_inline
def _group_norm[
    dtype: DType
](
    out_addr: Int,
    mean_out_addr: Int,
    rstd_out_addr: Int,
    in_addr: Int,
    gamma_addr: Int,
    beta_addr: Int,
    eps: Float32,
    rows: Int,
    cols: Int,
    hxw: Int,
    group: Int,
    cpg: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var mean_out_ptr = _make_ptr[DType.float32](mean_out_addr)
    var rstd_out_ptr = _make_ptr[DType.float32](rstd_out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var gamma_ptr = _make_ptr[dtype](gamma_addr)
    var beta_ptr = _make_ptr[dtype](beta_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(
            out_ptr, mean_out_ptr, rstd_out_ptr, in_ptr, gamma_ptr, beta_ptr
        )
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var r = Int(idx[0].value())
            var g = r % group
            var base = r * cols
            var total = Float32(0)
            for j in range(cols):
                total += in_ptr[base + j].cast[DType.float32]()
            var mean = total / Float32(cols)
            var var_sum = Float32(0)
            for j in range(cols):
                var d = in_ptr[base + j].cast[DType.float32]() - mean
                var_sum += d * d
            var rstd = 1.0 / ieee_sqrt(var_sum / Float32(cols) + eps)
            for j in range(cols):
                var c = g * cpg + j // hxw
                var x = in_ptr[base + j].cast[DType.float32]()
                var gm = gamma_ptr[c].cast[DType.float32]()
                var bt = beta_ptr[c].cast[DType.float32]()
                out_ptr[base + j] = ((x - mean) * rstd * gm + bt).cast[dtype]()
            mean_out_ptr[r] = mean
            rstd_out_ptr[r] = rstd

        _parallel_for[func](rows, ctx)
    else:
        comptime if has_accelerator():
            _enqueue_cached[_group_norm_block_kernel[dtype]](
                ctx,
                String(t"group_norm_block_{dtype}"),
                rows,
                1,
                1,
                ROWRED_THREADS,
                out_ptr.as_unsafe_any_origin(),
                mean_out_ptr.as_unsafe_any_origin(),
                rstd_out_ptr.as_unsafe_any_origin(),
                in_ptr.as_unsafe_any_origin().as_immutable(),
                gamma_ptr.as_unsafe_any_origin().as_immutable(),
                beta_ptr.as_unsafe_any_origin().as_immutable(),
                eps,
                Int64(cols),
                Int64(hxw),
                Int64(group),
                Int64(cpg),
            )
        else:
            raise Error("no GPU accelerator available at compile time")


def _group_norm_go(
    out_ptr_obj: PyObjectPtr,
    mean_out_ptr_obj: PyObjectPtr,
    rstd_out_ptr_obj: PyObjectPtr,
    in_ptr_obj: PyObjectPtr,
    gamma_ptr_obj: PyObjectPtr,
    beta_ptr_obj: PyObjectPtr,
    params: PyObjectPtr,  # (eps, rows, cols, hxw, group, cpg)
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var out_addr = _raw_int(out_ptr_obj)
    var mean_out_addr = _raw_int(mean_out_ptr_obj)
    var rstd_out_addr = _raw_int(rstd_out_ptr_obj)
    var in_addr = _raw_int(in_ptr_obj)
    var gamma_addr = _raw_int(gamma_ptr_obj)
    var beta_addr = _raw_int(beta_ptr_obj)
    var eps_val = Float32(_raw_tuple_f64(params, 0))
    var rows_val = _raw_tuple_int(params, 1)
    var cols_val = _raw_tuple_int(params, 2)
    var hxw_val = _raw_tuple_int(params, 3)
    var group_val = _raw_tuple_int(params, 4)
    var cpg_val = _raw_tuple_int(params, 5)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _group_norm[dt](
                    out_addr,
                    mean_out_addr,
                    rstd_out_addr,
                    in_addr,
                    gamma_addr,
                    beta_addr,
                    eps_val,
                    rows_val,
                    cols_val,
                    hxw_val,
                    group_val,
                    cpg_val,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error("unsupported dtype for fast group_norm: " + String(dtype))


# ---------------------------------------------------------------------------
# Bilinear upsample 2D over NCHW contiguous input. The per-axis scale ratio and
# the align_corners flag are resolved Python-side (area_pixel_compute_scale);
# the kernel computes the source coordinate, the two neighbor indices, and the
# 1D lambda weights exactly as torch's compute_source_index_and_lambda, then
# blends the four corners. One parallel task per output element (CPU and GPU).
# ---------------------------------------------------------------------------


@always_inline
def _src_index_lambda(
    ratio: Float32,
    dst: Int,
    in_size: Int,
    out_size: Int,
    align_corners: Int,
) -> Tuple[Int, Int, Float32, Float32]:
    """torch compute_source_index_and_lambda for one axis: returns the two
    neighbor indices (idx0, idx1) and their weights (lam0, lam1)."""
    if out_size == in_size:
        return (dst, dst, Float32(1), Float32(0))
    var real: Float32
    if align_corners != 0:
        real = ratio * Float32(dst)
    else:
        real = ratio * (Float32(dst) + 0.5) - 0.5
        if real < 0.0:
            real = 0.0
    var idx0 = Int(floor(real))
    if idx0 > in_size - 1:
        idx0 = in_size - 1
    var lam1 = real - Float32(idx0)
    if lam1 < 0.0:
        lam1 = 0.0
    if lam1 > 1.0:
        lam1 = 1.0
    var idx1 = idx0 + 1 if idx0 < in_size - 1 else idx0
    var lam0 = Float32(1) - lam1
    return (idx0, idx1, lam0, lam1)


@always_inline
def _upsample_bilinear2d[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    ratio_h: Float32,
    ratio_w: Float32,
    in_h: Int,
    in_w: Int,
    out_h: Int,
    out_w: Int,
    planes: Int,
    align_corners: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var ow = i % out_w
        var oh = (i // out_w) % out_h
        var plane = i // (out_w * out_h)
        var in_base = plane * in_h * in_w

        var hh = _src_index_lambda(ratio_h, oh, in_h, out_h, align_corners)
        var ih0 = hh[0]
        var ih1 = hh[1]
        var h0 = hh[2]
        var h1 = hh[3]
        var ww = _src_index_lambda(ratio_w, ow, in_w, out_w, align_corners)
        var iw0 = ww[0]
        var iw1 = ww[1]
        var w0 = ww[2]
        var w1 = ww[3]

        var r0 = in_base + ih0 * in_w
        var r1 = in_base + ih1 * in_w
        var v00 = in_ptr[r0 + iw0].cast[DType.float32]()
        var v01 = in_ptr[r0 + iw1].cast[DType.float32]()
        var v10 = in_ptr[r1 + iw0].cast[DType.float32]()
        var v11 = in_ptr[r1 + iw1].cast[DType.float32]()
        var res = h0 * (w0 * v00 + w1 * v01) + h1 * (w0 * v10 + w1 * v11)
        out_ptr[i] = res.cast[dtype]()

    _parallel_for[func](planes * out_h * out_w, ctx)


def _upsample_bilinear2d_go(
    out_ptr_obj: PyObjectPtr,
    in_ptr_obj: PyObjectPtr,
    params: PyObjectPtr,  # (ratio_h, ratio_w, in_h, in_w, out_h, out_w, planes, align_corners)
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var out_addr = _raw_int(out_ptr_obj)
    var in_addr = _raw_int(in_ptr_obj)
    var ratio_h = Float32(_raw_tuple_f64(params, 0))
    var ratio_w = Float32(_raw_tuple_f64(params, 1))
    var in_h = _raw_tuple_int(params, 2)
    var in_w = _raw_tuple_int(params, 3)
    var out_h = _raw_tuple_int(params, 4)
    var out_w = _raw_tuple_int(params, 5)
    var planes = _raw_tuple_int(params, 6)
    var align_corners = _raw_tuple_int(params, 7)
    var ctx = _raw_ctx(device_context_ptr)

    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _upsample_bilinear2d[dt](
                    out_addr,
                    in_addr,
                    ratio_h,
                    ratio_w,
                    in_h,
                    in_w,
                    out_h,
                    out_w,
                    planes,
                    align_corners,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error(
            "unsupported dtype for fast upsample_bilinear2d: " + String(dtype)
        )


# ---------------------------------------------------------------------------
# METH_FASTCALL wrappers: raw CPython argument unpacking (no owning
# PythonObject per argument). Argument types are guaranteed by the internal
# Python callers; raise sites are unsupported-dtype guards gated upstream.
# ---------------------------------------------------------------------------


def _batch_norm_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _batch_norm_go(
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


def _layer_norm_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _layer_norm_go(
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


def _softmax_rows_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _softmax_rows_go(
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


def _softmax_rows_dropout_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _softmax_rows_dropout_go(
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
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


def _max_pool2d_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _max_pool2d_go(args[0], args[1], args[2], args[3], args[4], args[5])
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


def _avg_pool2d_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _avg_pool2d_go(args[0], args[1], args[2], args[3], args[4])
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


def _adaptive_avg_pool2d_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _adaptive_avg_pool2d_go(args[0], args[1], args[2], args[3], args[4])
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


def _group_norm_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _group_norm_go(
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


def _upsample_bilinear2d_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _upsample_bilinear2d_go(args[0], args[1], args[2], args[3], args[4])
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


def _gather0_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _gather0_go(
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


def _all_bool_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _all_bool_go(args[0], args[1], args[2], args[3])
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


def _any_bool_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _any_bool_go(args[0], args[1], args[2], args[3])
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


comptime SPEC_CUMSUM_DTYPES: List[DType] = [
    DType.int64,
    DType.int32,
    DType.float32,
]

comptime SPEC_MAXROWS_DTYPES: List[DType] = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.int64,
    DType.int32,
]


def _mean_spec_into_go(
    a_o: PyObjectPtr,
    rdims_t: PyObjectPtr,
    keepdim_o: PyObjectPtr,
    out_o: PyObjectPtr,
) raises:
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    if not _dtype_supported[List[DType](FLOAT_DTYPES)](a.dtype):
        raise Error("mojo spec mean: unsupported dtype ", a.dtype)
    var rows = 0
    var cols = 0
    var out_rank = 0
    var oshape = IndexList[MAX_RANK](1)
    _reduce_spec_geom(
        a,
        rdims_t,
        keepdim_o,
        rows,
        cols,
        out_rank,
        oshape,
    )
    if cols == 0:
        raise Error("mojo spec mean: empty reduce dim")

    var ctx = a.ctx()
    var nbytes = rows * a.itemsize
    _ = nbytes
    _check_into_sized(a, out, rows, a.dtype)
    var addr = out.ptr
    if rows > 0:
        comptime for dt in FLOAT_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if a.dtype == dt:
                    _mean_rows[dt](addr, a.ptr, rows, cols, ctx)


def _max_spec_into_go(
    a_o: PyObjectPtr,
    rdims_t: PyObjectPtr,
    keepdim_o: PyObjectPtr,
    out_o: PyObjectPtr,
) raises:
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    if not _dtype_supported[SPEC_MAXROWS_DTYPES](a.dtype):
        raise Error("mojo spec max: unsupported dtype ", a.dtype)
    if a.numel == 0:
        raise Error("mojo spec max: empty input")
    var rows = 0
    var cols = 0
    var out_rank = 0
    var oshape = IndexList[MAX_RANK](1)
    _reduce_spec_geom(
        a,
        rdims_t,
        keepdim_o,
        rows,
        cols,
        out_rank,
        oshape,
    )

    var ctx = a.ctx()
    var nbytes = rows * a.itemsize
    _ = nbytes
    _check_into_sized(a, out, rows, a.dtype)
    var addr = out.ptr
    if rows > 0:
        comptime for dt in SPEC_MAXROWS_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if a.dtype == dt:
                    _max_rows[dt](addr, a.ptr, rows, cols, ctx)


def _argmax_spec_into_go(
    a_o: PyObjectPtr,
    rdims_t: PyObjectPtr,
    keepdim_o: PyObjectPtr,
    out_o: PyObjectPtr,
) raises:
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    if not _dtype_supported[SPEC_MAXROWS_DTYPES](a.dtype):
        raise Error("mojo spec argmax: unsupported dtype ", a.dtype)
    if a.numel == 0:
        raise Error("mojo spec argmax: empty input")
    var rows = 0
    var cols = 0
    var out_rank = 0
    var oshape = IndexList[MAX_RANK](1)
    _reduce_spec_geom(
        a,
        rdims_t,
        keepdim_o,
        rows,
        cols,
        out_rank,
        oshape,
    )

    var ctx = a.ctx()
    var nbytes = rows * 8  # int64 output
    _ = nbytes
    _check_into_sized(a, out, rows, DType.int64)
    var addr = out.ptr
    if rows > 0:
        comptime for dt in SPEC_MAXROWS_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if a.dtype == dt:
                    _argmax_rows[dt](addr, a.ptr, rows, cols, ctx)


def _cumsum_spec_into_go(a_o: PyObjectPtr, out_o: PyObjectPtr) raises:
    """Cumulative sum over the trailing dim; full-shape output."""
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    if not _dtype_supported[SPEC_CUMSUM_DTYPES](a.dtype):
        raise Error("mojo spec cumsum: unsupported dtype ", a.dtype)
    if a.rank < 1 or a.numel == 0:
        raise Error("mojo spec cumsum: empty or rank-0 input")

    var cols = a.shape[MAX_RANK - 1]
    var rows = a.numel // cols
    var ctx = a.ctx()
    var nbytes = a.numel * a.itemsize
    _ = nbytes
    _check_into(a, out, a.dtype)
    var addr = out.ptr
    if a.contig:
        comptime for dt in [DType.int64, DType.int32, DType.float32]:
            comptime if _dtype_arg_on[0, dt]():
                if a.dtype == dt:
                    _cumsum_rows[dt](addr, a.ptr, rows, cols, ctx)
    else:
        raise Error(
            "mojo spec cumsum: input must be contiguous"
            " (Python pre-materializes)"
        )


def _batch_norm_spec_into_go(
    in_o: PyObjectPtr,
    mean_o: PyObjectPtr,
    var_o: PyObjectPtr,
    gamma_o: PyObjectPtr,
    beta_o: PyObjectPtr,
    eps_o: PyObjectPtr,
    out_o: PyObjectPtr,
) raises:
    """Inference batch norm: geometry (channels/inner) derived from the
    input spec, output alloc and launch in one boundary call, reusing the
    `_batch_norm` kernel above."""
    ref inp = _spec_ptr(in_o)[]
    ref out = _spec_ptr(out_o)[]
    ref meanp = _spec_ptr(mean_o)[]
    ref varp = _spec_ptr(var_o)[]
    ref gammap = _spec_ptr(gamma_o)[]
    ref betap = _spec_ptr(beta_o)[]
    var eps = Float32(_raw_f64(eps_o))

    if inp.rank < 2:
        raise Error("mojo spec batch_norm: input rank must be >= 2")
    if inp.numel == 0:
        raise Error("mojo spec batch_norm: empty input")
    if not (
        inp.contig
        and meanp.contig
        and varp.contig
        and gammap.contig
        and betap.contig
    ):
        raise Error("mojo spec batch_norm: all inputs must be contiguous")
    if (
        meanp.dtype != inp.dtype
        or varp.dtype != inp.dtype
        or gammap.dtype != inp.dtype
        or betap.dtype != inp.dtype
    ):
        raise Error("mojo spec batch_norm: stat/affine dtypes must match input")
    if not _dtype_supported[List[DType](FLOAT_DTYPES)](inp.dtype):
        raise Error("mojo spec batch_norm: unsupported dtype ", inp.dtype)

    var channels = inp.dim(1)
    var inner = 1
    for i in range(MAX_RANK - inp.rank + 2, MAX_RANK):
        inner *= inp.shape[i]

    var ctx = inp.ctx()
    var nbytes = inp.numel * inp.itemsize
    _ = nbytes
    _check_into(inp, out, inp.dtype)
    var addr = out.ptr
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if inp.dtype == dt:
                _batch_norm[dt](
                    addr,
                    inp.ptr,
                    meanp.ptr,
                    varp.ptr,
                    gammap.ptr,
                    betap.ptr,
                    eps,
                    channels,
                    inner,
                    inp.numel,
                    ctx,
                )


def _softmax_spec_into_go(a_o: PyObjectPtr, out_o: PyObjectPtr) raises:
    """Plain softmax over the trailing dim (scale=1, no causal mask);
    full-shape output. The non-trailing dim transpose recursion and the
    half_to_float cast stay in Python."""
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    if not _dtype_supported[List[DType](FLOAT_DTYPES)](a.dtype):
        raise Error("mojo spec softmax: unsupported dtype ", a.dtype)
    if a.rank < 1 or a.numel == 0:
        raise Error("mojo spec softmax: empty or rank-0 input")

    var cols = a.shape[MAX_RANK - 1]
    var rows = a.numel // cols
    var ctx = a.ctx()
    var nbytes = a.numel * a.itemsize
    _ = nbytes
    _check_into(a, out, a.dtype)
    var addr = out.ptr
    if a.contig:
        comptime for dt in FLOAT_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if a.dtype == dt:
                    _softmax_rows[dt](
                        addr, a.ptr, rows, cols, Float32(1.0), 0, 1, ctx
                    )
    else:
        raise Error(
            "mojo spec softmax: input must be contiguous"
            " (Python pre-materializes)"
        )


def _attn_decode_spec_into_go(
    q_o: PyObjectPtr,
    k_o: PyObjectPtr,
    v_o: PyObjectPtr,
    scale_o: PyObjectPtr,
    out_o: PyObjectPtr,
) raises:
    """Fused decode attention (q_len == 1, not causal), GPU only — one
    boundary call replacing the Python gates + geometry + alloc + launch.
    Q/K/V read through their real strides; the innermost head dimension must
    remain contiguous for vector loads."""
    ref q = _spec_ptr(q_o)[]
    ref out = _spec_ptr(out_o)[]
    ref k = _spec_ptr(k_o)[]
    ref v = _spec_ptr(v_o)[]
    var scale = Float32(_raw_f64(scale_o))

    if q.rank != 4 or k.rank != 4 or v.rank != 4:
        raise Error("mojo spec attn_decode: rank != 4")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise Error("mojo spec attn_decode: dtypes differ")
    if not _dtype_supported[List[DType](FLOAT_DTYPES)](q.dtype):
        raise Error("mojo spec attn_decode: unsupported dtype ", q.dtype)
    var b = q.shape[MAX_RANK - 4]
    var h = q.shape[MAX_RANK - 3]
    var q_len = q.shape[MAX_RANK - 2]
    var head_dim = q.shape[MAX_RANK - 1]
    var kv_len = k.shape[MAX_RANK - 2]
    if q_len != 1:
        raise Error("mojo spec attn_decode: q_len != 1")
    for i in range(4):
        if k.shape[MAX_RANK - 4 + i] != v.shape[MAX_RANK - 4 + i]:
            raise Error("mojo spec attn_decode: k/v shapes differ")
    if (
        b != k.shape[MAX_RANK - 4]
        or h != k.shape[MAX_RANK - 3]
        or head_dim != k.shape[MAX_RANK - 1]
    ):
        raise Error("mojo spec attn_decode: q/k shapes incompatible")
    if b * h * kv_len * head_dim == 0:
        raise Error("mojo spec attn_decode: empty input")
    if (
        q.strides[MAX_RANK - 1] != 1
        or k.strides[MAX_RANK - 1] != 1
        or v.strides[MAX_RANK - 1] != 1
        or k.strides[MAX_RANK - 2] != head_dim
        or v.strides[MAX_RANK - 2] != head_dim
    ):
        raise Error("mojo spec attn_decode: unsupported q/k/v strides")

    var ctx = q.ctx()
    if ctx.api() == "cpu":
        # The CPU device takes the bmm+softmax+bmm chain today; keep it.
        raise Error("mojo spec attn_decode: GPU only")
    if head_dim % 4 != 0 or head_dim > ATTN_MAX_HD or kv_len > ATTN_MAX_KV:
        raise Error("mojo spec attn_decode: size caps")

    var numel = b * h * head_dim
    var nbytes = numel * q.itemsize
    _ = nbytes
    _check_into_sized(q, out, numel, q.dtype)
    var addr = out.ptr
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if q.dtype == dt:
                _attn_decode[dt](
                    addr,
                    q.ptr,
                    k.ptr,
                    v.ptr,
                    b * h,
                    kv_len,
                    head_dim,
                    scale,
                    h,
                    q.strides[MAX_RANK - 4],
                    q.strides[MAX_RANK - 3],
                    k.strides[MAX_RANK - 4],
                    k.strides[MAX_RANK - 3],
                    k.strides[MAX_RANK - 2],
                    v.strides[MAX_RANK - 4],
                    v.strides[MAX_RANK - 3],
                    v.strides[MAX_RANK - 2],
                    ctx,
                )
    var oshape = IndexList[MAX_RANK](1)
    oshape[MAX_RANK - 4] = b
    oshape[MAX_RANK - 3] = h
    oshape[MAX_RANK - 2] = 1
    oshape[MAX_RANK - 1] = head_dim


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_nn_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("nn_ops")
        comptime if _op_on["MeanSpec"]():
            _register_call(
                b,
                _spec_dispatcher4[_mean_spec_into_go, "MeanSpec"],
                docstring="(a_spec, rdims, keepdim, out_spec)",
            )
        comptime if _op_on["MaxSpec"]():
            _register_call(
                b,
                _spec_dispatcher4[_max_spec_into_go, "MaxSpec"],
                docstring="(a_spec, rdims, keepdim, out_spec)",
            )
        comptime if _op_on["ArgmaxSpec"]():
            _register_call(
                b,
                _spec_dispatcher4[_argmax_spec_into_go, "ArgmaxSpec"],
                docstring="(a_spec, rdims, keepdim, out_spec); int64 indices",
            )
        comptime if _op_on["CumsumSpec"]():
            _register_call(
                b,
                _spec_dispatcher2[_cumsum_spec_into_go, "CumsumSpec"],
                docstring="(a_spec, out_spec); trailing dim",
            )
        comptime if _op_on["BatchNormSpec"]():
            _register_call(
                b,
                _spec_dispatcher7[_batch_norm_spec_into_go, "BatchNormSpec"],
                docstring=(
                    "(in, mean, var, gamma, beta specs, eps, out_spec);"
                    " inference batch norm, geometry from specs"
                ),
            )
        comptime if _op_on["SoftmaxSpec"]():
            _register_call(
                b,
                _spec_dispatcher2[_softmax_spec_into_go, "SoftmaxSpec"],
                docstring="(a_spec, out_spec); trailing dim",
            )
        comptime if _op_on["AttnDecodeSpec"]():
            _register_call(
                b,
                _spec_dispatcher5[_attn_decode_spec_into_go, "AttnDecodeSpec"],
                docstring="(q, k, v specs, scale, out_spec); q_len==1, GPU",
            )
        comptime if _op_on["BatchNormInference"]():
            _register_call(
                b,
                _batch_norm_dispatcher,
                docstring=(
                    "out = (x - mean[c]) * gamma[c] / sqrt(var[c] + eps) +"
                    " beta[c] (NC..., contiguous)"
                ),
            )
        comptime if _op_on["LayerNorm"]():
            _register_call(
                b,
                _layer_norm_dispatcher,
                docstring=(
                    "layer norm over the last dim; also writes float32"
                    " mean/rstd per row"
                ),
            )
        comptime if _op_on["SoftmaxRows"]():
            _register_call(
                b,
                _softmax_rows_dispatcher,
                docstring="row softmax of scale*x with optional causal mask",
            )
        comptime if _op_on["SoftmaxRowsDropoutF32"]():
            _register_call(
                b,
                _softmax_rows_dropout_dispatcher,
                docstring=(
                    "fused causal row softmax + Philox native dropout (Apple"
                    " GPU f32): writes pre-dropout probs, dropped probs, and"
                    " the bool keep-mask in one pass"
                ),
            )
        comptime if _op_on["MaxPool2dWithIndices"]():
            _register_call(
                b,
                _max_pool2d_dispatcher,
                docstring=(
                    "max pool over NCHW contiguous input, returns values and"
                    " int64 plane indices"
                ),
            )
        comptime if _op_on["AvgPool2d"]():
            _register_call(
                b,
                _avg_pool2d_dispatcher,
                docstring=(
                    "average pool over NCHW contiguous input (count_include_pad"
                    " / divisor_override honored)"
                ),
            )
        comptime if _op_on["AdaptiveAvgPool2d"]():
            _register_call(
                b,
                _adaptive_avg_pool2d_dispatcher,
                docstring="adaptive average pool over NCHW contiguous input",
            )
        comptime if _op_on["GroupNorm"]():
            _register_call(
                b,
                _group_norm_dispatcher,
                docstring=(
                    "group norm over NC(HxW) contiguous input; writes float32"
                    " mean/rstd per (sample, group)"
                ),
            )
        comptime if _op_on["UpsampleBilinear2d"]():
            _register_call(
                b,
                _upsample_bilinear2d_dispatcher,
                docstring="bilinear upsample over NCHW contiguous input",
            )
        comptime if _op_on["Gather0"]():
            _register_call(
                b,
                _gather0_dispatcher,
                docstring="embedding lookup: gather rows of a 2D table",
            )
        comptime if _op_on["AllBool"]():
            _register_call(
                b,
                _all_bool_dispatcher,
                docstring="all() over a bool tensor -> scalar bool",
            )
        comptime if _op_on["AnyBool"]():
            _register_call(
                b,
                _any_bool_dispatcher,
                docstring="any() over a bool tensor -> scalar bool",
            )
        return b.finalize()
    except e:
        abort(t"failed to create nn_ops python module: {e}")
