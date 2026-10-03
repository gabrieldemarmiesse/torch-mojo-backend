# ===----------------------------------------------------------------------=== #
# Pairwise distance kernels for mojo_device: `_cdist_forward`,
# `_cdist_backward`, `_pdist_forward`, `_pdist_backward`.
#
# The math is ATen's CUDA kernels' (aten/src/ATen/native/cuda/DistanceKernel.cu
# at v2.14.0): the same `dists<scalar_t>` norms (zero, one, two, inf, general
# p) with their `inc` / `agg` / `finish` / `backward` formulas, computed in the
# tensor's own dtype (CUDA dispatches float and double only; the half types are
# declined by the op before any build). Forward: one 256-thread block per
# output distance, each thread striding over the columns, then a block
# reduction with the norm's `agg`. Backward: where CUDA writes one term per
# (pair, column) into a buffer and sums it with `at::sum_out`, a thread here
# owns one gradient element and sums its terms itself, in the buffer's order
# along the summed axis -- no buffer, no atomics, deterministic.
#
# pdist's pair index k maps to rows (i, j), i < j, as on CUDA; the row is found
# with a float32 estimate and an exact integer correction instead of CUDA's
# float64 square root, so Apple GPUs (no float64) run the same kernel.
# ===----------------------------------------------------------------------=== #

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from std.math import sqrt
from std.memory import AddressSpace, stack_allocation
from std.utils.numerics import isinf, isnan
from std.utils.static_tuple import StaticTuple

from tmb.kernels.common.math_utils import ieee_sqrt
from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    _enqueue_cached,
    _make_ptr,
    _raw_ctx,
    _raw_f64,
    _raw_int,
)
from tmb.kernels.common.pow_math import pow_c99
from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)

# The norm of a launch (`dists<scalar_t>::<norm>`). The backward ladder has
# its own `lt_two` member (0 < p < 2, p != 1) and no `zero` one.
comptime NORM_ZERO = 0
comptime NORM_ONE = 1
comptime NORM_TWO = 2
comptime NORM_INF = 3
comptime NORM_P = 4
comptime NORM_LT_TWO = 5

# kCUDANumThreads: one block per distance.
comptime _THREADS = 256
# Grid cap; larger launches stride over their outputs.
comptime _MAX_BLOCKS = 65535

comptime _DTYPES = [DType.float32, DType.float64]


# ---------------------------------------------------------------------------
# dists<scalar_t>
# ---------------------------------------------------------------------------


@always_inline
def _sign[dt: DType](v: Scalar[dt]) -> Scalar[dt]:
    """`(0 < val) - (val < 0)`: 0 for zero and NaN."""
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


@always_inline
def _inc[
    dt: DType, norm: Int
](agg: Scalar[dt], diff: Scalar[dt], p: Scalar[dt]) -> Scalar[dt]:
    comptime if norm == NORM_ZERO:
        if isnan(diff):
            return diff
        if diff != 0:
            return agg + 1
        return agg
    elif norm == NORM_ONE:
        return agg + diff
    elif norm == NORM_TWO:
        return agg + diff * diff
    elif norm == NORM_INF:
        # `if (diff > agg)`: a NaN difference never wins.
        if diff > agg:
            return diff
        return agg
    else:
        return agg + pow_c99[dt](diff, p)


@always_inline
def _agg[dt: DType, norm: Int](a: Scalar[dt], b: Scalar[dt]) -> Scalar[dt]:
    comptime if norm == NORM_INF:
        if b > a:
            return b
        return a
    else:
        return a + b


@always_inline
def _finish[dt: DType, norm: Int](agg: Scalar[dt], p: Scalar[dt]) -> Scalar[dt]:
    comptime if norm == NORM_TWO:
        return ieee_sqrt(agg)
    elif norm == NORM_P:
        return pow_c99[dt](agg, Scalar[dt](1) / p)
    else:
        return agg


@always_inline
def _backward[
    dt: DType, norm: Int
](
    diff: Scalar[dt], grad: Scalar[dt], dist: Scalar[dt], p: Scalar[dt]
) -> Scalar[dt]:
    comptime if norm == NORM_ONE:
        return grad * _sign(diff)
    elif norm == NORM_LT_TWO:
        if dist == 0 or (diff == 0 and p < 1):
            return 0
        return (
            _sign(diff)
            * pow_c99[dt](abs(diff), p - 1)
            * grad
            / pow_c99[dt](dist, p - 1)
        )
    elif norm == NORM_TWO:
        if dist == 0:
            return 0
        return grad * diff / dist
    elif norm == NORM_INF:
        return (
            grad
            * _sign(diff)
            * (Scalar[dt](1) if abs(diff) == dist else Scalar[dt](0))
        )
    else:
        if dist == 0:
            return 0
        return (
            diff
            * pow_c99[dt](abs(diff), p - 2)
            * grad
            / pow_c99[dt](dist, p - 1)
        )


# ---------------------------------------------------------------------------
# pdist pair indexing
# ---------------------------------------------------------------------------


@always_inline
def _pair_start(n: Int, i: Int) -> Int:
    """The pair index of (i, i + 1): pairs are ordered row-major over i < j."""
    return n * i - i * (i + 1) // 2


@always_inline
def _pair_row(n: Int, k: Int) -> Int:
    """The row i of pair k: CUDA's `n2 - sqrt(n2^2 - 1 - 2k)` estimate (here in
    float32), then corrected exactly against the integer pair starts."""
    var n2 = Float32(n) - 0.5
    var est = n2 - sqrt(max(n2 * n2 - 1 - 2 * Float32(k), Float32(0)))
    var i = min(max(Int(est), 0), n - 2)
    while i > 0 and _pair_start(n, i) > k:
        i -= 1
    while i < n - 2 and _pair_start(n, i + 1) <= k:
        i += 1
    return i


# ---------------------------------------------------------------------------
# Forward: one block per distance
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_THREADS))
)
@__name(t"dist_fwd_block_reduce_{dt}_norm{norm}_pdist{pdist}")
def _dist_fwd_kernel[
    dt: DType, norm: Int, pdist: Bool
](
    out_ptr: Pointer[Scalar[dt], MutAnyOrigin],
    x1_ptr: Pointer[Scalar[dt], MutAnyOrigin],
    x2_ptr: Pointer[Scalar[dt], MutAnyOrigin],
    p: Scalar[dt],
    count_arg: Int64,
    r1_arg: Int64,
    r2_arg: Int64,
    m_arg: Int64,
):
    """cdist_kernel_cuda_impl (`pdist` False: x1 (B, r1, m) against x2
    (B, r2, m), out (B, r1, r2)) and pdist_kernel_cuda_impl (`pdist` True:
    rows of x1 (r1, m) against each other, out the r1 * (r1 - 1) / 2 pairs)."""
    var count = Int(count_arg)
    var r1 = Int(r1_arg)
    var r2 = Int(r2_arg)
    var m = Int(m_arg)
    var tid = Int(thread_idx.x)
    var red = stack_allocation[
        _THREADS, dt, address_space=AddressSpace.SHARED
    ]()
    var b_ptr = x1_ptr if pdist else x2_ptr
    var k = Int(block_idx.x)
    while k < count:
        var a_row: Int
        var b_row: Int
        comptime if pdist:
            var i = _pair_row(r1, k)
            var j = k - _pair_start(r1, i) + i + 1
            a_row = i * m
            b_row = j * m
        else:
            var r_size = r1 * r2
            var l = k // r_size
            var kk = k - l * r_size
            var i = kk // r2
            var j = kk - i * r2
            a_row = (l * r1 + i) * m
            b_row = (l * r2 + j) * m
        var agg = Scalar[dt](0)
        var c = tid
        while c < m:
            var diff = abs(
                x1_ptr[unsafe_offset=a_row + c] - b_ptr[unsafe_offset=b_row + c]
            )
            agg = _inc[dt, norm](agg, diff, p)
            c += _THREADS
        red[unsafe_offset=tid] = agg
        barrier()
        var stride = _THREADS // 2
        while stride > 0:
            if tid < stride:
                red[unsafe_offset=tid] = _agg[dt, norm](
                    red[unsafe_offset=tid], red[unsafe_offset=tid + stride]
                )
            barrier()
            stride //= 2
        if tid == 0:
            out_ptr[unsafe_offset=k] = _finish[dt, norm](
                red[unsafe_offset=0], p
            )
        barrier()  # red[0] is read before the next distance overwrites it
        k += Int(grid_dim.x)


# ---------------------------------------------------------------------------
# Backward: one thread per (gradient element, slice of x2's rows)
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_THREADS))
)
@__name(t"cdist_bwd_gather_{dt}_norm{norm}")
def _cdist_bwd_kernel[
    dt: DType, norm: Int
](
    gx_ptr: Pointer[Scalar[dt], MutAnyOrigin],
    grad_ptr: Pointer[Scalar[dt], MutAnyOrigin],
    x1_ptr: Pointer[Scalar[dt], MutAnyOrigin],
    x2_ptr: Pointer[Scalar[dt], MutAnyOrigin],
    dist_ptr: Pointer[Scalar[dt], MutAnyOrigin],
    p: Scalar[dt],
    batch_arg: Int64,
    r1_arg: Int64,
    r2_arg: Int64,
    m_arg: Int64,
    slices_arg: Int64,
):
    """grad_x1[l, i, c] = sum_j backward(x1[l, i, c] - x2[l, j, c],
    grad[l, i, j], dist[l, i, j]): cdist_backward_kernel_cuda_impl's buffer
    (B, r2, r1, m) summed over its r2 axis.

    With `slices` > 1 the r2 axis is cut into that many contiguous slices
    and each thread sums one slice of one element into `gx_ptr[s * total +
    e]`, a workspace `_cdist_bwd_sum` then reduces in slice order: a small
    x1 against a long x2 (few elements, many rows) still fills the GPU, as
    CUDA's 2-D grid over (r2, r1 * m) does."""
    var r1 = Int(r1_arg)
    var r2 = Int(r2_arg)
    var m = Int(m_arg)
    var slices = Int(slices_arg)
    var total = Int(batch_arg) * r1 * m
    var chunk = (r2 + slices - 1) // slices
    var t = Int(block_idx.x) * _THREADS + Int(thread_idx.x)
    while t < total * slices:
        var e = t % total
        var sl = t // total
        var c = e % m
        var row = e // m  # l * r1 + i
        var l = row // r1
        var xi = x1_ptr[unsafe_offset=e]
        var acc = Scalar[dt](0)
        var x2_base = l * r2 * m + c
        var g_base = row * r2
        var j_end = min(r2, (sl + 1) * chunk)
        for j in range(sl * chunk, j_end):
            acc += _backward[dt, norm](
                xi - x2_ptr[unsafe_offset=x2_base + j * m],
                grad_ptr[unsafe_offset=g_base + j],
                dist_ptr[unsafe_offset=g_base + j],
                p,
            )
        gx_ptr[unsafe_offset=t] = acc
        t += Int(grid_dim.x) * _THREADS


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_THREADS))
)
@__name(t"cdist_bwd_slice_sum_{dt}")
def _cdist_bwd_sum[
    dt: DType
](
    gx_ptr: Pointer[Scalar[dt], MutAnyOrigin],
    ws_ptr: Pointer[Scalar[dt], MutAnyOrigin],
    total_arg: Int64,
    slices_arg: Int64,
):
    """gx[e] = sum over slices of the workspace, in slice order
    (deterministic)."""
    var total = Int(total_arg)
    var slices = Int(slices_arg)
    var e = Int(block_idx.x) * _THREADS + Int(thread_idx.x)
    while e < total:
        var acc = Scalar[dt](0)
        for sl in range(slices):
            acc += ws_ptr[unsafe_offset=sl * total + e]
        gx_ptr[unsafe_offset=e] = acc
        e += Int(grid_dim.x) * _THREADS


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_THREADS))
)
@__name(t"pdist_bwd_gather_{dt}_norm{norm}")
def _pdist_bwd_kernel[
    dt: DType, norm: Int
](
    gx_ptr: Pointer[Scalar[dt], MutAnyOrigin],
    grad_ptr: Pointer[Scalar[dt], MutAnyOrigin],
    x_ptr: Pointer[Scalar[dt], MutAnyOrigin],
    dist_ptr: Pointer[Scalar[dt], MutAnyOrigin],
    p: Scalar[dt],
    grad_stride_arg: Int64,
    n_arg: Int64,
    m_arg: Int64,
):
    """pdist_backward_kernel_cuda_impl's buffer (n - 1, n, m) summed over its
    first axis: row r collects +backward(x_r - x_j) for every j > r (buffer
    slots 0 .. n - r - 2), then -backward(x_i - x_r) for i = r - 1 down to 0
    (the remaining slots)."""
    var n = Int(n_arg)
    var m = Int(m_arg)
    var gs = Int(grad_stride_arg)
    var total = n * m
    var e = Int(block_idx.x) * _THREADS + Int(thread_idx.x)
    while e < total:
        var c = e % m
        var r = e // m
        var xr = x_ptr[unsafe_offset=e]
        var acc = Scalar[dt](0)
        var k0 = _pair_start(n, r) - r - 1  # pair (r, j) is k0 + j
        for j in range(r + 1, n):
            var k = k0 + j
            acc += _backward[dt, norm](
                xr - x_ptr[unsafe_offset=j * m + c],
                grad_ptr[unsafe_offset=k * gs],
                dist_ptr[unsafe_offset=k],
                p,
            )
        var i = r - 1
        while i >= 0:
            var k = _pair_start(n, i) + r - i - 1
            acc += -_backward[dt, norm](
                x_ptr[unsafe_offset=i * m + c] - xr,
                grad_ptr[unsafe_offset=k * gs],
                dist_ptr[unsafe_offset=k],
                p,
            )
            i -= 1
        gx_ptr[unsafe_offset=e] = acc
        e += Int(grid_dim.x) * _THREADS


# ---------------------------------------------------------------------------
# Host launchers
# ---------------------------------------------------------------------------


@always_inline
def _ptr[dt: DType](a: Arg) -> Pointer[Scalar[dt], MutAnyOrigin]:
    return _make_ptr[dt](_raw_int(a)).as_unsafe_any_origin()


def _blocks(work: Int) -> Int:
    return max(1, min(work, _MAX_BLOCKS))


def _fwd[
    dt: DType, norm: Int, pdist: Bool
](argv: Argv, p: Scalar[dt], ctx: DeviceContext) raises:
    var count = _raw_int(argv[unsafe_offset=5])
    _enqueue_cached[_dist_fwd_kernel[dt, norm, pdist]](
        ctx,
        _blocks(count),
        1,
        1,
        _THREADS,
        _ptr[dt](argv[unsafe_offset=0]),
        _ptr[dt](argv[unsafe_offset=1]),
        _ptr[dt](argv[unsafe_offset=2]),
        p,
        Int64(count),
        Int64(_raw_int(argv[unsafe_offset=6])),
        Int64(_raw_int(argv[unsafe_offset=7])),
        Int64(_raw_int(argv[unsafe_offset=8])),
    )


def _fwd_dispatch[dt: DType, pdist: Bool](argv: Argv, argc: Int) raises:
    """Slots: out, x1, x2 (ignored by pdist), p (f64), norm, count, r1, r2,
    m, ctx."""
    if argc != 10:
        raise Error("DistForward: expected 10 arguments, got ", argc)
    var p = Scalar[dt](_raw_f64(argv[unsafe_offset=3]))
    var norm = _raw_int(argv[unsafe_offset=4])
    var ctx = _raw_ctx(argv[unsafe_offset=9])
    if norm == NORM_ZERO:
        _fwd[dt, NORM_ZERO, pdist](argv, p, ctx)
    elif norm == NORM_ONE:
        _fwd[dt, NORM_ONE, pdist](argv, p, ctx)
    elif norm == NORM_TWO:
        _fwd[dt, NORM_TWO, pdist](argv, p, ctx)
    elif norm == NORM_INF:
        _fwd[dt, NORM_INF, pdist](argv, p, ctx)
    else:
        _fwd[dt, NORM_P, pdist](argv, p, ctx)


def _cdist_bwd[
    dt: DType, norm: Int
](argv: Argv, p: Scalar[dt], ctx: DeviceContext) raises:
    var batch = _raw_int(argv[unsafe_offset=7])
    var r1 = _raw_int(argv[unsafe_offset=8])
    var m = _raw_int(argv[unsafe_offset=10])
    var ws = _raw_int(argv[unsafe_offset=12])
    var slices = _raw_int(argv[unsafe_offset=13])
    var total = batch * r1 * m
    var gx = _ptr[dt](argv[unsafe_offset=0])
    var dst = gx if slices == 1 else _make_ptr[dt](ws).as_unsafe_any_origin()
    _enqueue_cached[_cdist_bwd_kernel[dt, norm]](
        ctx,
        _blocks((total * slices + _THREADS - 1) // _THREADS),
        1,
        1,
        _THREADS,
        dst,
        _ptr[dt](argv[unsafe_offset=1]),
        _ptr[dt](argv[unsafe_offset=2]),
        _ptr[dt](argv[unsafe_offset=3]),
        _ptr[dt](argv[unsafe_offset=4]),
        p,
        Int64(batch),
        Int64(r1),
        Int64(_raw_int(argv[unsafe_offset=9])),
        Int64(m),
        Int64(slices),
    )
    if slices > 1:
        _enqueue_cached[_cdist_bwd_sum[dt]](
            ctx,
            _blocks((total + _THREADS - 1) // _THREADS),
            1,
            1,
            _THREADS,
            gx,
            dst,
            Int64(total),
            Int64(slices),
        )


def _cdist_bwd_dispatch[dt: DType](argv: Argv, argc: Int) raises:
    """Slots: grad_x1, grad, x1, x2, dist, p (f64), norm, batch, r1, r2, m,
    ctx, workspace (slices * batch * r1 * m elements, or 0), slices."""
    if argc != 14:
        raise Error("CdistBackward: expected 14 arguments, got ", argc)
    var p = Scalar[dt](_raw_f64(argv[unsafe_offset=5]))
    var norm = _raw_int(argv[unsafe_offset=6])
    var ctx = _raw_ctx(argv[unsafe_offset=11])
    if norm == NORM_ONE:
        _cdist_bwd[dt, NORM_ONE](argv, p, ctx)
    elif norm == NORM_LT_TWO:
        _cdist_bwd[dt, NORM_LT_TWO](argv, p, ctx)
    elif norm == NORM_TWO:
        _cdist_bwd[dt, NORM_TWO](argv, p, ctx)
    elif norm == NORM_INF:
        _cdist_bwd[dt, NORM_INF](argv, p, ctx)
    else:
        _cdist_bwd[dt, NORM_P](argv, p, ctx)


def _pdist_bwd[
    dt: DType, norm: Int
](argv: Argv, p: Scalar[dt], ctx: DeviceContext) raises:
    var n = _raw_int(argv[unsafe_offset=7])
    var m = _raw_int(argv[unsafe_offset=8])
    _enqueue_cached[_pdist_bwd_kernel[dt, norm]](
        ctx,
        _blocks((n * m + _THREADS - 1) // _THREADS),
        1,
        1,
        _THREADS,
        _ptr[dt](argv[unsafe_offset=0]),
        _ptr[dt](argv[unsafe_offset=1]),
        _ptr[dt](argv[unsafe_offset=2]),
        _ptr[dt](argv[unsafe_offset=3]),
        p,
        Int64(_raw_int(argv[unsafe_offset=6])),
        Int64(n),
        Int64(m),
    )


def _pdist_bwd_dispatch[dt: DType](argv: Argv, argc: Int) raises:
    """Slots: grad_self, grad, self, dist, p (f64), norm, grad stride, n, m,
    ctx."""
    if argc != 10:
        raise Error("PdistBackward: expected 10 arguments, got ", argc)
    var p = Scalar[dt](_raw_f64(argv[unsafe_offset=4]))
    var norm = _raw_int(argv[unsafe_offset=5])
    var ctx = _raw_ctx(argv[unsafe_offset=9])
    if norm == NORM_ONE:
        _pdist_bwd[dt, NORM_ONE](argv, p, ctx)
    elif norm == NORM_LT_TWO:
        _pdist_bwd[dt, NORM_LT_TWO](argv, p, ctx)
    elif norm == NORM_TWO:
        _pdist_bwd[dt, NORM_TWO](argv, p, ctx)
    elif norm == NORM_INF:
        _pdist_bwd[dt, NORM_INF](argv, p, ctx)
    else:
        _pdist_bwd[dt, NORM_P](argv, p, ctx)


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime for dt in _DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                comptime if _op_on["CdistForward"]():
                    _fwd_dispatch[dt, False](argv, argc)
                    return 0
                comptime if _op_on["PdistForward"]():
                    _fwd_dispatch[dt, True](argv, argc)
                    return 0
                comptime if _op_on["CdistBackward"]():
                    _cdist_bwd_dispatch[dt](argv, argc)
                    return 0
                comptime if _op_on["PdistBackward"]():
                    _pdist_bwd_dispatch[dt](argv, argc)
                    return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
