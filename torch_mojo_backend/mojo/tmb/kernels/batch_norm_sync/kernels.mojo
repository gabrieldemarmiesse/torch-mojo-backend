"""SyncBatchNorm's building blocks: per-channel statistics, the elementwise
normalization, the cross-replica statistics merge and the two halves of the
backward (`batch_norm_stats`, `batch_norm_elemt`, `batch_norm_gather_stats*`,
`batch_norm_backward_reduce`, `batch_norm_backward_elemt`,
`batch_norm_update_stats`).

Every kernel reads a contiguous `[N, C, HxW]` input (the op materializes
it) and follows torch's CUDA formulas (aten/src/ATen/native/cuda/
Normalization.cuh and Normalization.cu at v2.14.0) in the accumulation dtype
`acc_type` -- float32 for half inputs, float64 for float64 -- with the same
operation order in every elementwise expression:

  * statistics: Welford mean and BIASED variance per channel (per-thread
    passes merged pairwise, as CUDA's kernel), `InvStd` = 0 when `var == 0 and
    eps == 0`, else `1 / sqrt(var + eps)` with `var + eps` formed in double
    as CUDA's `T + double` is (in float32 on Apple GPUs, which have no
    double);
  * running statistics: `mean * momentum + (1 - momentum) * running` and
    the unbiased variance `var * N / (N - 1)` (N = 1 gives NaN, like CUDA);
  * elementwise: `gamma * (x - mean) * invstd + beta`;
  * the gather merges replicas' (mean, invstd, count) rows in order with
    batch_norm_reduce_statistics_kernel's integer running count.
"""

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from std.math import sqrt
from std.memory import stack_allocation
from std.sys.info import has_apple_gpu_accelerator
from std.utils.coord import Coord

from tmb.kernels.common.op_utils import (
    _enqueue_cached,
    _make_ptr,
    _parallel_for_dt,
)
from tmb.kernels.common.block_reduce import block_sum

comptime BN_SYNC_THREADS = 256
# `var + epsilon` is a double sum in CUDA's InvStd; Apple GPUs have no
# double, so there it is a float32 one (and no Float64 reaches the kernel).
comptime EPS_T = (
    DType.float32 if has_apple_gpu_accelerator() else DType.float64
)


@always_inline
def _acc[dtype: DType]() -> DType:
    """torch's `acc_type<scalar_t, /*is_cuda=*/true>`."""
    return DType.float64 if dtype == DType.float64 else DType.float32


@always_inline
def _inv_std[t: DType](var_: Scalar[t], eps: Scalar[EPS_T]) -> Scalar[t]:
    """Normalization.cuh `InvStd`: `T + double` is a double sum."""
    if var_ == 0 and eps == 0:
        return 0
    var s = sqrt(var_.cast[EPS_T]() + eps)
    return (1 / s).cast[t]()


@__name(t"bn_sync_stats_{dtype}_{rdtype}")
def _stats_kernel[
    dtype: DType, rdtype: DType
](
    mean_ptr: Pointer[Scalar[_acc[dtype]()], MutAnyOrigin],
    var_ptr: Pointer[Scalar[_acc[dtype]()], MutAnyOrigin],
    run_mean_ptr: Pointer[Scalar[rdtype], MutAnyOrigin],
    run_var_ptr: Pointer[Scalar[rdtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    channels_arg: Int64,
    batch_arg: Int64,
    hxw_arg: Int64,
    invstd_arg: Int64,
    has_running_arg: Int64,
    eps: Scalar[EPS_T],
    momentum: Scalar[_acc[rdtype]()],
    bessel: Scalar[_acc[rdtype]()],
):
    """One block per channel: Welford mean and biased variance, then
    `InvStd` (batch_norm_stats) or the variance plus the running update
    (batch_norm_update_stats)."""
    comptime acc_t = _acc[dtype]()
    var channels = Int(channels_arg)
    var hxw = Int(hxw_arg)
    var count = Int(batch_arg) * hxw
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    # batch_norm_collect_statistics_kernel: a Welford pass per thread,
    # then the pairwise Welford merge across the block (CUDA merges by
    # warp shuffles; here a shared-memory tree, the same merge formula).
    var avg = Scalar[acc_t](0)
    var m2_t = Scalar[acc_t](0)
    var n_t = 0
    var j = tid
    while j < count:
        var n = j // hxw
        var v = in_ptr[
            unsafe_offset=(n * channels + c) * hxw + j - n * hxw
        ].cast[acc_t]()
        var d1 = v - avg
        n_t += 1
        avg += d1 / Scalar[acc_t](n_t)
        m2_t += d1 * (v - avg)
        j += BN_SYNC_THREADS
    var sh_avg = stack_allocation[
        BN_SYNC_THREADS, acc_t, address_space=AddressSpace.SHARED
    ]()
    var sh_m2 = stack_allocation[
        BN_SYNC_THREADS, acc_t, address_space=AddressSpace.SHARED
    ]()
    var sh_n = stack_allocation[
        BN_SYNC_THREADS, DType.int64, address_space=AddressSpace.SHARED
    ]()
    sh_avg[unsafe_offset=tid] = avg
    sh_m2[unsafe_offset=tid] = m2_t
    sh_n[unsafe_offset=tid] = Int64(n_t)
    barrier()
    var stride = BN_SYNC_THREADS // 2
    while stride > 0:
        if tid < stride:
            var n_a = Int(sh_n[unsafe_offset=tid])
            var n_b = Int(sh_n[unsafe_offset=tid + stride])
            var avg_a = sh_avg[unsafe_offset=tid]
            var avg_b = sh_avg[unsafe_offset=tid + stride]
            var factor = Scalar[acc_t](1) / Scalar[acc_t](max(1, n_a + n_b))
            var delta = avg_a - avg_b
            sh_m2[unsafe_offset=tid] = (
                sh_m2[unsafe_offset=tid]
                + sh_m2[unsafe_offset=tid + stride]
                + delta
                * delta
                * Scalar[acc_t](n_a)
                * Scalar[acc_t](n_b)
                * factor
            )
            sh_avg[unsafe_offset=tid] = (
                Scalar[acc_t](n_a) * avg_a + Scalar[acc_t](n_b) * avg_b
            ) * factor
            sh_n[unsafe_offset=tid] = Int64(n_a + n_b)
        barrier()
        stride //= 2
    if tid != 0:
        return
    var mean = sh_avg[unsafe_offset=0]
    var m2 = sh_m2[unsafe_offset=0]
    var var_ = m2 / Scalar[acc_t](count)
    mean_ptr[unsafe_offset=c] = mean
    var mode = Int(invstd_arg)
    if mode == 1:
        var_ptr[unsafe_offset=c] = _inv_std[acc_t](var_, eps)
        return
    if mode == 2:
        # batch_norm_update_stats_and_invert: `rsqrt(var + eps)` in acc_t.
        var_ptr[unsafe_offset=c] = 1 / sqrt(var_ + eps.cast[acc_t]())
    else:
        var_ptr[unsafe_offset=c] = var_
    if Int(has_running_arg) != 0:
        comptime racc = _acc[rdtype]()
        # batch_norm_update_stats: acc_t of the running dtype throughout.
        var mom = momentum
        var unbiased = var_.cast[racc]() * bessel
        var rm = run_mean_ptr[unsafe_offset=c].cast[racc]()
        var rv = run_var_ptr[unsafe_offset=c].cast[racc]()
        run_mean_ptr[unsafe_offset=c] = (
            mean.cast[racc]() * mom + (1 - mom) * rm
        ).cast[rdtype]()
        run_var_ptr[unsafe_offset=c] = (unbiased * mom + (1 - mom) * rv).cast[
            rdtype
        ]()


def bn_stats[
    dtype: DType, rdtype: DType
](
    mean_addr: Int,
    var_addr: Int,
    run_mean_addr: Int,
    run_var_addr: Int,
    in_addr: Int,
    channels: Int,
    batch: Int,
    hxw: Int,
    mode: Int,
    has_running: Bool,
    eps: Float64,
    momentum: Float64,
    ctx: DeviceContext,
) raises:
    """`mode` 0: the biased variance (batch_norm_update_stats), 1: InvStd
    (batch_norm_stats), 2: `rsqrt(var + eps)` (native batch norm training);
    the running update runs whenever `has_running`."""
    comptime acc_t = _acc[dtype]()
    comptime racc = _acc[rdtype]()
    var count = batch * hxw
    # `static_cast<acc_t>(double(N) / double(N - 1))`, inf for N = 1.
    var bessel = Float64(count) / Float64(count - 1)
    _enqueue_cached[_stats_kernel[dtype, rdtype]](
        ctx,
        channels,
        1,
        1,
        BN_SYNC_THREADS,
        _make_ptr[acc_t](mean_addr).as_unsafe_any_origin(),
        _make_ptr[acc_t](var_addr).as_unsafe_any_origin(),
        _make_ptr[rdtype](run_mean_addr).as_unsafe_any_origin(),
        _make_ptr[rdtype](run_var_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](in_addr).as_unsafe_any_origin(),
        Int64(channels),
        Int64(batch),
        Int64(hxw),
        Int64(mode),
        Int64(1 if has_running else 0),
        Scalar[EPS_T](eps),
        Scalar[racc](momentum),
        Scalar[racc](bessel),
    )


def bn_elemt[
    dtype: DType, sdtype: DType, pdtype: DType
](
    out_addr: Int,
    in_addr: Int,
    weight_addr: Int,
    bias_addr: Int,
    mean_addr: Int,
    invstd_addr: Int,
    channels: Int,
    hxw: Int,
    numel: Int,
    ctx: DeviceContext,
) raises:
    """batch_norm_transform_input_kernel (train): `gamma * (x - mean) *
    invstd + beta` in the accumulation dtype, gamma 1 / beta 0 when absent."""
    comptime acc_t = _acc[dtype]()
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var w_ptr = _make_ptr[pdtype](weight_addr)
    var b_ptr = _make_ptr[pdtype](bias_addr)
    var m_ptr = _make_ptr[sdtype](mean_addr)
    var s_ptr = _make_ptr[sdtype](invstd_addr)
    var has_w = weight_addr != 0
    var has_b = bias_addr != 0

    @always_inline
    @__parameter
    @__copy_capture(
        out_ptr, in_ptr, w_ptr, b_ptr, m_ptr, s_ptr, has_w, has_b, channels, hxw
    )
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var c = (i // hxw) % channels
        var gamma = w_ptr[unsafe_offset=c].cast[acc_t]() if has_w else Scalar[
            acc_t
        ](1)
        var beta = b_ptr[unsafe_offset=c].cast[acc_t]() if has_b else Scalar[
            acc_t
        ](0)
        var mean = m_ptr[unsafe_offset=c].cast[acc_t]()
        var invstd = s_ptr[unsafe_offset=c].cast[acc_t]()
        var x = in_ptr[unsafe_offset=i].cast[acc_t]()
        out_ptr[unsafe_offset=i] = (gamma * (x - mean) * invstd + beta).cast[
            dtype
        ]()

    _parallel_for_dt[dtype, func](numel, ctx)


def bn_gather[
    sdtype: DType, rdtype: DType
](
    save_mean_addr: Int,
    save_invstd_addr: Int,
    mean_addr: Int,
    invstd_addr: Int,
    run_mean_addr: Int,
    run_var_addr: Int,
    counts_addr: Int,
    world: Int,
    features: Int,
    eps: Float64,
    momentum: Float64,
    ctx: DeviceContext,
) raises:
    """batch_norm_reduce_statistics_kernel, one thread per feature; `rdtype`
    is its `scalar_t` (the running statistics' and counts' dtype), the math
    in `acc_type<scalar_t>`."""
    comptime acc_t = _acc[rdtype]()
    var sm_ptr = _make_ptr[acc_t](save_mean_addr)
    var si_ptr = _make_ptr[acc_t](save_invstd_addr)
    var m_ptr = _make_ptr[sdtype](mean_addr)
    var v_ptr = _make_ptr[sdtype](invstd_addr)
    var rm_ptr = _make_ptr[rdtype](run_mean_addr)
    var rv_ptr = _make_ptr[rdtype](run_var_addr)
    var c_ptr = _make_ptr[rdtype](counts_addr)
    var has_rm = run_mean_addr != 0
    var has_rv = run_var_addr != 0
    var epsilon = Scalar[acc_t](eps)
    var mom = Scalar[acc_t](momentum)

    @always_inline
    @__parameter
    @__copy_capture(
        sm_ptr,
        si_ptr,
        m_ptr,
        v_ptr,
        rm_ptr,
        rv_ptr,
        c_ptr,
        has_rm,
        has_rv,
        epsilon,
        mom,
        world,
        features,
    )
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var avg = Scalar[acc_t](0)
        var var_n = Scalar[acc_t](0)
        var n = 0
        for j in range(world):
            var count = c_ptr[unsafe_offset=j]
            var m = m_ptr[unsafe_offset=j * features + i].cast[acc_t]()
            var v = (
                Scalar[acc_t](1)
                / v_ptr[unsafe_offset=j * features + i].cast[acc_t]()
            )
            v = (v * v - epsilon) * count.cast[acc_t]()
            # `n + count`: an int plus scalar_t, in scalar_t.
            var factor = (
                Scalar[acc_t](1) / (Scalar[rdtype](n) + count).cast[acc_t]()
            )
            var nf = Scalar[acc_t](n)
            var cf = count.cast[acc_t]()
            var_n += v + (avg - m) * (avg - m) * nf * cf * factor
            avg = nf * factor * avg + cf * factor * m
            # `index_t n += scalar_t count`: the sum in scalar_t's
            # arithmetic (c10::Half / BFloat16 round it), truncated.
            n = Int(Scalar[rdtype](n) + count)
        sm_ptr[unsafe_offset=i] = avg
        si_ptr[unsafe_offset=i] = Scalar[acc_t](1) / sqrt(
            var_n / Scalar[acc_t](n) + epsilon
        )
        if has_rm:
            rm_ptr[unsafe_offset=i] = (
                (1 - mom) * rm_ptr[unsafe_offset=i].cast[acc_t]() + mom * avg
            ).cast[rdtype]()
        var unbiased = var_n / Scalar[acc_t](n - 1)
        if has_rv:
            rv_ptr[unsafe_offset=i] = (
                (1 - mom) * rv_ptr[unsafe_offset=i].cast[acc_t]()
                + mom * unbiased
            ).cast[rdtype]()

    _parallel_for_dt[rdtype, func](features, ctx)


@__name(t"bn_sync_backward_reduce_{dtype}_{sdtype}_{wdtype}")
def _backward_reduce_kernel[
    dtype: DType, sdtype: DType, wdtype: DType
](
    sum_dy_ptr: Pointer[Scalar[sdtype], MutAnyOrigin],
    sum_dy_xmu_ptr: Pointer[Scalar[sdtype], MutAnyOrigin],
    gw_ptr: Pointer[Scalar[wdtype], MutAnyOrigin],
    gb_ptr: Pointer[Scalar[wdtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    go_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    mean_ptr: Pointer[Scalar[sdtype], MutAnyOrigin],
    invstd_ptr: Pointer[Scalar[sdtype], MutAnyOrigin],
    flags_arg: Int64,
    channels_arg: Int64,
    batch_arg: Int64,
    hxw_arg: Int64,
):
    """batch_norm_backward_reduce_kernel: per channel `sum(dy)` and
    `sum(dy * (x - mean))` in the accumulation dtype; `flags` bit 0 writes
    the two sums, bit 1 grad_weight (`dot * invstd`), bit 2 grad_bias, and
    bit 3 rounds each product through the input dtype (the non-channels-last
    route's GradOp does, the channels-last kernel does not)."""
    comptime acc_t = _acc[dtype]()
    var channels = Int(channels_arg)
    var hxw = Int(hxw_arg)
    var count = Int(batch_arg) * hxw
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var mean = mean_ptr[unsafe_offset=c].cast[acc_t]()
    var round_prod = (Int(flags_arg) & 8) != 0
    var sum_dy = Scalar[acc_t](0)
    var dot = Scalar[acc_t](0)
    var j = tid
    while j < count:
        var n = j // hxw
        var at = (n * channels + c) * hxw + j - n * hxw
        var g = go_ptr[unsafe_offset=at].cast[acc_t]()
        sum_dy += g
        var prod = g * (in_ptr[unsafe_offset=at].cast[acc_t]() - mean)
        if round_prod:
            # GradOp builds `Float2<scalar_t, acc_t>(g, g * c)`: the product
            # is rounded through the input dtype before it is accumulated.
            prod = prod.cast[dtype]().cast[acc_t]()
        dot += prod
        j += BN_SYNC_THREADS
    var t_dy = block_sum[acc_t, BN_SYNC_THREADS](sum_dy)
    var t_dot = block_sum[acc_t, BN_SYNC_THREADS](dot)
    if tid != 0:
        return
    var flags = Int(flags_arg)
    if flags & 1:
        sum_dy_ptr[unsafe_offset=c] = t_dy.cast[sdtype]()
        sum_dy_xmu_ptr[unsafe_offset=c] = t_dot.cast[sdtype]()
    if flags & 2:
        var invstd = invstd_ptr[unsafe_offset=c].cast[acc_t]()
        gw_ptr[unsafe_offset=c] = (t_dot * invstd).cast[wdtype]()
    if flags & 4:
        gb_ptr[unsafe_offset=c] = t_dy.cast[wdtype]()


# The non-channels-last CUDA route reduces through `reduce<Float2<scalar_t,
# acc_t>>` (Normalization.cuh), whose warp shuffle rebuilds a Float2 from two
# shuffled acc_t values through its `(scalar_t, scalar_t)` constructor: for a
# half / bfloat16 input every shuffled partial sum is rounded to that dtype.
# Reproducing those roundings needs CUDA's exact thread geometry and
# reduction tree, emulated here over shared memory (lanes of 32, MAX_BLOCK_SIZE
# 512) so it is the same on every GPU.
comptime CUDA_WARP = 32
comptime CUDA_MAX_BLOCK = 512


@always_inline
def _round_through[
    dtype: DType, acc_t: DType
](v: Scalar[acc_t]) -> Scalar[acc_t]:
    return v.cast[dtype]().cast[acc_t]()


@__name(t"bn_sync_backward_reduce_cuda_order_{dtype}_{sdtype}_{wdtype}")
def _backward_reduce_cuda_order_kernel[
    dtype: DType, sdtype: DType, wdtype: DType
](
    sum_dy_ptr: Pointer[Scalar[sdtype], MutAnyOrigin],
    sum_dy_xmu_ptr: Pointer[Scalar[sdtype], MutAnyOrigin],
    gw_ptr: Pointer[Scalar[wdtype], MutAnyOrigin],
    gb_ptr: Pointer[Scalar[wdtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    go_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    mean_ptr: Pointer[Scalar[sdtype], MutAnyOrigin],
    invstd_ptr: Pointer[Scalar[sdtype], MutAnyOrigin],
    flags_arg: Int64,
    channels_arg: Int64,
    batch_arg: Int64,
    hxw_arg: Int64,
    bx_arg: Int64,
    by_arg: Int64,
):
    """batch_norm_backward_reduce_kernel with CUDA's (block_x, block_y)
    geometry and reduction order, launched with CUDA_MAX_BLOCK threads of
    which the first `bx * by` are CUDA's."""
    comptime acc_t = _acc[dtype]()
    var s1 = stack_allocation[
        CUDA_MAX_BLOCK, acc_t, address_space=AddressSpace.SHARED
    ]()
    var s2 = stack_allocation[
        CUDA_MAX_BLOCK, acc_t, address_space=AddressSpace.SHARED
    ]()
    var w1 = stack_allocation[
        CUDA_WARP, acc_t, address_space=AddressSpace.SHARED
    ]()
    var w2 = stack_allocation[
        CUDA_WARP, acc_t, address_space=AddressSpace.SHARED
    ]()
    var channels = Int(channels_arg)
    var batch = Int(batch_arg)
    var hxw = Int(hxw_arg)
    var bx = Int(bx_arg)
    var by = Int(by_arg)
    var active = bx * by
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var mean = mean_ptr[unsafe_offset=c].cast[acc_t]()
    var v1 = Scalar[acc_t](0)
    var v2 = Scalar[acc_t](0)
    if tid < active:
        var tx = tid % bx
        var n = tid // bx
        while n < batch:
            var base = (n * channels + c) * hxw
            var x = tx
            while x < hxw:
                var g = go_ptr[unsafe_offset=base + x].cast[acc_t]()
                var cc = in_ptr[unsafe_offset=base + x].cast[acc_t]() - mean
                # Float2(scalar_t g, scalar_t g * c)
                v1 += _round_through[dtype](g)
                v2 += _round_through[dtype](g * cc)
                x += bx
            n += by
    var lane = tid % CUDA_WARP
    # block_reduce.cuh WarpReduce over SumReduceOp<Float2>: offsets 16..1,
    # `val + Float2(shfl_down(val.v1), shfl_down(val.v2))`, a lane past the
    # warp reading its own value as `__shfl_down_sync` does.
    var offset = CUDA_WARP // 2
    while offset > 0:
        s1[unsafe_offset=tid] = v1
        s2[unsafe_offset=tid] = v2
        barrier()
        var src = tid + offset if lane + offset < CUDA_WARP else tid
        var o1 = s1[unsafe_offset=src]
        var o2 = s2[unsafe_offset=src]
        barrier()
        v1 = v1 + _round_through[dtype](o1)
        v2 = v2 + _round_through[dtype](o2)
        offset //= 2
    var wid = tid // CUDA_WARP
    if lane == 0 and wid < CUDA_WARP:
        w1[unsafe_offset=wid] = v1
        w2[unsafe_offset=wid] = v2
    barrier()
    var warps = active // CUDA_WARP
    v1 = w1[unsafe_offset=lane] if tid < warps else Scalar[acc_t](0)
    v2 = w2[unsafe_offset=lane] if tid < warps else Scalar[acc_t](0)
    # block_reduce.cuh WarpReduce over SumReduceOp<Float2>: offsets 16..1,
    # `val + Float2(shfl_down(val.v1), shfl_down(val.v2))`, a lane past the
    # warp reading its own value as `__shfl_down_sync` does.
    offset = CUDA_WARP // 2
    while offset > 0:
        s1[unsafe_offset=tid] = v1
        s2[unsafe_offset=tid] = v2
        barrier()
        var src = tid + offset if lane + offset < CUDA_WARP else tid
        var o1 = s1[unsafe_offset=src]
        var o2 = s2[unsafe_offset=src]
        barrier()
        v1 = v1 + _round_through[dtype](o1)
        v2 = v2 + _round_through[dtype](o2)
        offset //= 2
    if tid != 0:
        return
    var flags = Int(flags_arg)
    if flags & 1:
        sum_dy_ptr[unsafe_offset=c] = v1.cast[sdtype]()
        sum_dy_xmu_ptr[unsafe_offset=c] = v2.cast[sdtype]()
    if flags & 2:
        var invstd = invstd_ptr[unsafe_offset=c].cast[acc_t]()
        gw_ptr[unsafe_offset=c] = (v2 * invstd).cast[wdtype]()
    if flags & 4:
        gb_ptr[unsafe_offset=c] = v1.cast[wdtype]()


def _get_num_threads(n: Int) -> Int:
    """Normalization.cuh getNumThreads (CUDA's table)."""
    for t in [32, 64, 128, 256]:
        if n <= t:
            return t
    return CUDA_MAX_BLOCK


def _last_pow2(n: Int) -> Int:
    """LaunchUtils.h lastPow2: 2**floor(log2(n)), at least 1."""
    var p = 1
    while p * 2 <= n:
        p *= 2
    return p


def bn_backward_reduce[
    dtype: DType, sdtype: DType, wdtype: DType
](
    sum_dy_addr: Int,
    sum_dy_xmu_addr: Int,
    gw_addr: Int,
    gb_addr: Int,
    in_addr: Int,
    go_addr: Int,
    mean_addr: Int,
    invstd_addr: Int,
    flags: Int,
    channels: Int,
    batch: Int,
    hxw: Int,
    ctx: DeviceContext,
) raises:
    comptime if dtype == DType.float16 or dtype == DType.bfloat16:
        if flags & 8:
            var by = min(_last_pow2(batch), CUDA_MAX_BLOCK // CUDA_WARP)
            var bx = min(
                max(_get_num_threads(hxw), CUDA_WARP), CUDA_MAX_BLOCK // by
            )
            _enqueue_cached[
                _backward_reduce_cuda_order_kernel[dtype, sdtype, wdtype]
            ](
                ctx,
                channels,
                1,
                1,
                CUDA_MAX_BLOCK,
                _make_ptr[sdtype](sum_dy_addr).as_unsafe_any_origin(),
                _make_ptr[sdtype](sum_dy_xmu_addr).as_unsafe_any_origin(),
                _make_ptr[wdtype](gw_addr).as_unsafe_any_origin(),
                _make_ptr[wdtype](gb_addr).as_unsafe_any_origin(),
                _make_ptr[dtype](in_addr).as_unsafe_any_origin(),
                _make_ptr[dtype](go_addr).as_unsafe_any_origin(),
                _make_ptr[sdtype](mean_addr).as_unsafe_any_origin(),
                _make_ptr[sdtype](invstd_addr).as_unsafe_any_origin(),
                Int64(flags),
                Int64(channels),
                Int64(batch),
                Int64(hxw),
                Int64(bx),
                Int64(by),
            )
            return
    _enqueue_cached[_backward_reduce_kernel[dtype, sdtype, wdtype]](
        ctx,
        channels,
        1,
        1,
        BN_SYNC_THREADS,
        _make_ptr[sdtype](sum_dy_addr).as_unsafe_any_origin(),
        _make_ptr[sdtype](sum_dy_xmu_addr).as_unsafe_any_origin(),
        _make_ptr[wdtype](gw_addr).as_unsafe_any_origin(),
        _make_ptr[wdtype](gb_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](in_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](go_addr).as_unsafe_any_origin(),
        _make_ptr[sdtype](mean_addr).as_unsafe_any_origin(),
        _make_ptr[sdtype](invstd_addr).as_unsafe_any_origin(),
        Int64(flags),
        Int64(channels),
        Int64(batch),
        Int64(hxw),
    )


def bn_backward_elemt[
    dtype: DType, sdtype: DType, wdtype: DType
](
    gi_addr: Int,
    go_addr: Int,
    in_addr: Int,
    mean_addr: Int,
    invstd_addr: Int,
    weight_addr: Int,
    sum_dy_addr: Int,
    sum_dy_xmu_addr: Int,
    count_addr: Int,
    world: Int,
    channels: Int,
    hxw: Int,
    numel: Int,
    ctx: DeviceContext,
) raises:
    """batch_norm_backward_elemt_kernel_impl, with `norm_fct = 1 / sum(count)`
    over the int32 per-replica counts read on the device, as CUDA does."""
    comptime acc_t = _acc[dtype]()
    var gi_ptr = _make_ptr[dtype](gi_addr)
    var go_ptr = _make_ptr[dtype](go_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var m_ptr = _make_ptr[sdtype](mean_addr)
    var s_ptr = _make_ptr[sdtype](invstd_addr)
    var w_ptr = _make_ptr[wdtype](weight_addr)
    var dy_ptr = _make_ptr[sdtype](sum_dy_addr)
    var xmu_ptr = _make_ptr[sdtype](sum_dy_xmu_addr)
    var cnt_ptr = _make_ptr[DType.int32](count_addr)
    var has_w = weight_addr != 0

    @always_inline
    @__parameter
    @__copy_capture(
        gi_ptr,
        go_ptr,
        in_ptr,
        m_ptr,
        s_ptr,
        w_ptr,
        dy_ptr,
        xmu_ptr,
        cnt_ptr,
        has_w,
        world,
        channels,
        hxw,
    )
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var c = (i // hxw) % channels
        var total = 0
        for k in range(world):
            total += Int(cnt_ptr[unsafe_offset=k])
        var norm = Scalar[acc_t](1) / Scalar[acc_t](total)
        var m_c = m_ptr[unsafe_offset=c].cast[acc_t]()
        var m_dy_c = dy_ptr[unsafe_offset=c].cast[acc_t]() * norm
        var factor_1_c = s_ptr[unsafe_offset=c].cast[acc_t]()
        var factor_2_c = w_ptr[unsafe_offset=c].cast[
            acc_t
        ]() if has_w else Scalar[acc_t](1)
        factor_2_c *= factor_1_c
        factor_1_c = (
            factor_1_c
            * factor_1_c
            * xmu_ptr[unsafe_offset=c].cast[acc_t]()
            * norm
        )
        var g = go_ptr[unsafe_offset=i].cast[acc_t]()
        var x = in_ptr[unsafe_offset=i].cast[acc_t]()
        gi_ptr[unsafe_offset=i] = (
            (g - m_dy_c - (x - m_c) * factor_1_c) * factor_2_c
        ).cast[dtype]()

    _parallel_for_dt[dtype, func](numel, ctx)
