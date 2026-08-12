"""Batch norm forward device kernels: the training and the inference form.

Both end in the same elementwise pass — `out = (x - mean) * invstd * gamma +
beta`, one channel's four coefficients per block, so the reciprocal square
root is evaluated once per block instead of once per element — and they differ
only in where `(mean, invstd)` comes from:

  * inference (`aten::_native_batch_norm_legit_no_training`, and
    `aten::native_batch_norm` with `training=False`) reads the running
    statistics, so the op is purely elementwise.
  * training (`aten::native_batch_norm` with `training=True`) computes them
    over N*H*W per channel first.

That training reduction is the interesting geometry: over NCHW it reduces
`{0, 2, 3}` keeping C, a reduce axis set that is neither contiguous nor
trailing. Materializing a permuted copy to make it trailing would cost two
extra passes over the whole tensor, so the statistics kernel here reads the
tensor where it lies: channel `c` owns `N` contiguous runs of `HxW` elements
at stride `C*HxW`, and one block walks a shard of those runs with the SHARED
shifted-moment scan (`op_utils._moments_scan_contig`, the same core
`aten.var.correction` uses). Consecutive lanes read consecutive addresses
inside a run, so every load is coalesced.

The shift is `x[0, c, 0]`, one broadcast load, and it is the same for every
shard of a channel — which is what lets the shards merge by plain addition.
`op_utils._moment_cancels` then guards the one case a single-pass shifted
formulation cannot survive on its own (that first element being a wild
outlier): the channel is re-read about its now-known accurate mean. The
decision is made on the device, so the split route enqueues its second
scan+merge pair unconditionally; with nothing flagged those two launches read
one float per channel and exit.

Statistics accumulate in float32 and `save_mean` / `save_invstd` are float32
whatever the input dtype, matching ATen (`at::acc_type`). ATen's own rules,
reproduced exactly:

  * the BIASED variance normalizes the output (`invstd = rsqrt(var + eps)`),
  * the UNBIASED one updates the running variance
    (`var * N/(N-1)`, with N the reduced count),
  * `running_mean = (1 - momentum) * running_mean + momentum * mean`.
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.primitives import block
from std.math import ceildiv, max, min
from std.sys.info import has_accelerator, size_of
from std.utils.static_tuple import StaticTuple

from op_utils import (
    _device_sm_count,
    _enqueue_cached,
    _make_ptr,
    _moment_cancels,
    _moment_partition,
    _moments_scan_contig,
    _vec16_phase,
    ieee_sqrt,
)


comptime BN_THREADS = 256
comptime _MAX_GRID = 65535

# Statistics blocks in flight targeted when the channel count alone cannot
# fill the device. The unsplit grid is one block per channel, so a 64-channel
# convolution activation would otherwise run on 64 of this card's 114 SMs;
# splitting is skipped entirely once `channels >= sm_count`, because the fused
# route it enables (finalize in the statistics kernel, no workspace and no
# merge launch) is worth more than the extra shards — measured 15.6us against
# 26.8us on N8 C256 28x28 f32.
#
# FITTED ON AN H100 PCIe (114 SMs, queried at RUNTIME — the compile-time
# `default_device_info.sm_count` table reports 132 for this part). Device us
# of the whole `aten::native_batch_norm(training=True)` over blocks/SM of
# 1/2/4/8:
#
#   N32 C64 112x112  f32   201.9  188.4  185.7  188.3
#                    bf16  113.1  101.9  100.0  104.5
#   N8  C256 28x28   f32    15.7   15.6   15.6   15.7  (fused; unaffected)
#
# i.e. it bottoms out at 4: below it the device is starved, above it the
# workspace the extra shards write (and the merge reads back) costs more than
# the parallelism buys.
comptime BN_BLOCKS_PER_SM = 4

# Floor on the run count handed to one shard: below a couple of runs the
# workspace slot costs more than the shard computes.
comptime BN_MIN_RUNS_PER_SPLIT = 2


@always_inline
def _bn_scale_shift[
    pdtype: DType, sdtype: DType, //, from_invstd: Bool
](
    mean_ptr: UnsafePointer[Scalar[sdtype], ImmutAnyOrigin],
    var_ptr: UnsafePointer[Scalar[sdtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[pdtype], ImmutAnyOrigin],
    beta_ptr: UnsafePointer[Scalar[pdtype], ImmutAnyOrigin],
    c: Int,
    eps: Float32,
    has_weight: Bool,
    has_bias: Bool,
    mut mean: Float32,
    mut scale: Float32,
    mut shift: Float32,
):
    """One channel's `(mean, gamma*invstd, beta)`, read once per block.

    `from_invstd` says the second statistics buffer already holds
    `1/sqrt(var + eps)` (the training route, whose merge inverted it) rather
    than the variance (the inference route, reading a running variance).
    """
    mean = mean_ptr[c].cast[DType.float32]()
    comptime if from_invstd:
        scale = var_ptr[c].cast[DType.float32]()
    else:
        scale = 1.0 / ieee_sqrt(var_ptr[c].cast[DType.float32]() + eps)
    if has_weight:
        scale *= gamma_ptr[c].cast[DType.float32]()
    shift = Float32(0)
    if has_bias:
        shift = beta_ptr[c].cast[DType.float32]()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BN_THREADS))
)
@__name(t"batch_norm_elemwise_{dtype}_{from_invstd}_v{V}")
def _bn_elementwise_kernel[
    dtype: DType,
    pdtype: DType,
    sdtype: DType,
    V: Int,
    from_invstd: Bool,
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mean_ptr: UnsafePointer[Scalar[sdtype], ImmutAnyOrigin],
    var_ptr: UnsafePointer[Scalar[sdtype], ImmutAnyOrigin],
    gamma_ptr: UnsafePointer[Scalar[pdtype], ImmutAnyOrigin],
    beta_ptr: UnsafePointer[Scalar[pdtype], ImmutAnyOrigin],
    save_mean_ptr: UnsafePointer[Scalar[sdtype], MutAnyOrigin],
    save_invstd_ptr: UnsafePointer[Scalar[sdtype], MutAnyOrigin],
    eps: Float32,
    inner_slots_arg: Int64,
    channels_arg: Int64,
    planes_arg: Int64,
    has_weight_arg: Int64,
    has_bias_arg: Int64,
):
    """Block (x = slot tile, y = plane): one (sample, channel) plane per block
    row, so the channel — and therefore the four coefficients and the
    reciprocal square root — is uniform across the whole block and is
    evaluated once, not once per element.

    The inference route also emits the two statistics torch returns from
    `_native_batch_norm_legit_no_training`: `save_mean` is the running mean and
    `save_invstd` the `rsqrt(var + eps)` the per-channel prologue already forms
    before folding gamma into it.  Emitting them here costs one thread per
    channel; doing it outside cost three extra launches over C elements, which
    at small `inner` is the whole op again.  The training route
    (`from_invstd`) gets those statistics from its own stats pass and passes
    null, so the write is compiled out."""
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var inner_slots = Int(inner_slots_arg)
    var channels = Int(channels_arg)
    var planes = Int(planes_arg)
    var has_weight = Int(has_weight_arg) != 0
    var has_bias = Int(has_bias_arg) != 0
    comptime align = V * size_of[dtype]()
    var slot = Int(block_idx.x) * BN_THREADS + Int(thread_idx.x)
    if slot >= inner_slots:
        return
    var plane = Int(block_idx.y)
    var plane_stride = Int(grid_dim.y)
    comptime if not from_invstd:
        # Emitted OUTSIDE the plane loop: a per-iteration branch to catch the
        # channel-naming plane cost 16% on N8 C256 28x28, where the loop is the
        # whole kernel.  Its own grid-stride walk covers every channel whatever
        # grid_dim.y is, instead of assuming the y grid reaches `channels`.
        if slot == 0:
            var c = Int(block_idx.y)
            while c < channels:
                save_mean_ptr[c] = mean_ptr[c]
                save_invstd_ptr[c] = (
                    1.0 / ieee_sqrt(var_ptr[c].cast[DType.float32]() + eps)
                ).cast[sdtype]()
                c += plane_stride
    while plane < planes:
        var mean = Float32(0)
        var scale = Float32(0)
        var shift = Float32(0)
        _bn_scale_shift[from_invstd=from_invstd](
            mean_ptr,
            var_ptr,
            gamma_ptr,
            beta_ptr,
            plane % channels,
            eps,
            has_weight,
            has_bias,
            mean,
            scale,
            shift,
        )
        var at = plane * inner_slots * V + slot * V
        var x = in_ptr.load[width=V, alignment=align](at).cast[DType.float32]()
        out_ptr.store[width=V, alignment=align](
            at, ((x - mean) * scale + shift).cast[dtype]()
        )
        plane += plane_stride


@always_inline
def _bn_run_base(n: Int, c: Int, channels: Int, hxw: Int) -> Int:
    """Flat index of the first element of channel `c`'s run in sample `n`."""
    return (n * channels + c) * hxw


@always_inline
def _bn_scan_shard[
    dtype: DType, //
](
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    c: Int,
    channels: Int,
    hxw: Int,
    n0: Int,
    n1: Int,
    j0: Int,
    jn: Int,
    tid: Int,
    shift: Float32,
    mut s_tot: Float32,
    mut q_tot: Float32,
):
    """One thread's shifted moment pair over a rectangle of channel `c`'s
    (run, position) space, through the shared contiguous scan once per run."""
    comptime V = 16 // size_of[dtype]()
    comptime vec_align = V * size_of[dtype]()  # 16 bytes
    s_tot = Float32(0)
    q_tot = Float32(0)
    if n1 <= n0:
        return
    # Consecutive runs of a channel are `channels * hxw` apart, so when that
    # stride is a whole number of 16-byte vectors every run of the shard has
    # the SAME head/tail split and it is computed once. Recomputing it per run
    # puts an integer division on the dependency chain between one run's loads
    # and the next, which is what a shard of many short runs feels: hoisting it
    # took the 8x256x28x28 f32 statistics pass from 14.6us to 6.4us.
    var run_stride = channels * hxw
    var head = 0
    var n_vec = 0
    var vec_start = 0
    var tail_start = 0
    var uniform = run_stride % V == 0
    var first = _bn_run_base(n0, c, channels, hxw) + j0
    _moment_partition[dtype](
        Int(in_ptr), first, jn, head, n_vec, vec_start, tail_start
    )
    for n in range(n0, n1):
        var start = _bn_run_base(n, c, channels, hxw) + j0
        if not uniform:
            _moment_partition[dtype](
                Int(in_ptr), start, jn, head, n_vec, vec_start, tail_start
            )
        else:
            vec_start = start + head
        var s = Float32(0)
        var q = Float32(0)
        _moments_scan_contig[V=V, vec_align=vec_align, threads=BN_THREADS](
            in_ptr,
            start,
            jn,
            head,
            n_vec,
            vec_start,
            tail_start,
            tid,
            shift,
            s,
            q,
        )
        s_tot += s
        q_tot += q


@always_inline
def _bn_finalize[
    sdtype: DType, //
](
    mean_out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    invstd_out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    run_mean_ptr: UnsafePointer[Scalar[sdtype], MutAnyOrigin],
    run_var_ptr: UnsafePointer[Scalar[sdtype], MutAnyOrigin],
    c: Int,
    mean: Float32,
    m2_in: Float32,
    count: Int,
    eps: Float32,
    momentum: Float32,
    has_running: Bool,
):
    """ATen's rules, verbatim: the BIASED variance normalizes the output while
    the UNBIASED one updates the running variance."""
    var m2 = m2_in
    if m2 < 0:  # a few ulps below zero on a constant channel
        m2 = Float32(0)
    var nf = Float32(count)
    var biased = m2 / nf
    mean_out_ptr[c] = mean
    invstd_out_ptr[c] = 1.0 / ieee_sqrt(biased + eps)
    if has_running:
        var keep = 1.0 - momentum
        run_mean_ptr[c] = (
            keep * run_mean_ptr[c].cast[DType.float32]() + momentum * mean
        ).cast[sdtype]()
        var unbiased = biased
        if count > 1:
            unbiased = m2 / (nf - 1.0)
        run_var_ptr[c] = (
            keep * run_var_ptr[c].cast[DType.float32]() + momentum * unbiased
        ).cast[sdtype]()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BN_THREADS))
)
@__name(t"batch_norm_moments_fused_{dtype}")
def _bn_moments_fused_kernel[
    dtype: DType, sdtype: DType
](
    mean_out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    invstd_out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    run_mean_ptr: UnsafePointer[Scalar[sdtype], MutAnyOrigin],
    run_var_ptr: UnsafePointer[Scalar[sdtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    channels_arg: Int64,
    runs_arg: Int64,
    hxw_arg: Int64,
    eps: Float32,
    momentum: Float32,
    has_running_arg: Int64,
):
    """One block per channel, no workspace and no second launch: the block
    owns the whole channel, so the cancellation re-pass is a re-scan in place
    and the statistics are finalized right here. This is the route whenever
    the channel count alone fills the device."""
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var channels = Int(channels_arg)
    var runs = Int(runs_arg)
    var hxw = Int(hxw_arg)
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var count = runs * hxw
    var shift = in_ptr[c * hxw].cast[DType.float32]()

    var s = Float32(0)
    var q = Float32(0)
    _bn_scan_shard(in_ptr, c, channels, hxw, 0, runs, 0, hxw, tid, shift, s, q)
    var bs = block.sum[block_size=BN_THREADS](s)
    var bq = block.sum[block_size=BN_THREADS](q)
    # Cold path; `block.sum` broadcasts, so the whole block agrees and the
    # branch cannot desynchronize the barriers inside the second reduction.
    if _moment_cancels(bs, bq, count):
        shift += bs / Float32(count)
        _bn_scan_shard(
            in_ptr, c, channels, hxw, 0, runs, 0, hxw, tid, shift, s, q
        )
        bs = block.sum[block_size=BN_THREADS](s)
        bq = block.sum[block_size=BN_THREADS](q)
    if tid == 0:
        _bn_finalize(
            mean_out_ptr,
            invstd_out_ptr,
            run_mean_ptr,
            run_var_ptr,
            c,
            shift + bs / Float32(count),
            bq - bs * bs / Float32(count),
            count,
            eps,
            momentum,
            Int(has_running_arg) != 0,
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BN_THREADS))
)
@__name(t"batch_norm_moments_{dtype}")
def _bn_moments_kernel[
    dtype: DType
](
    ws_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    channels_arg: Int64,
    runs_arg: Int64,
    hxw_arg: Int64,
    splits_n_arg: Int64,
    splits_j_arg: Int64,
    repass_arg: Int64,
):
    """Block (x = channel, y = shard): the shifted moment pair of one shard of
    one channel, written to the workspace split-major so the merge below reads
    consecutive addresses.

    A shard is a rectangle of the channel's `(run, position)` space: `runs`
    are the N samples, each a contiguous `hxw`-element interval, so sharding
    the runs costs nothing and sharding inside a run is what keeps a
    small-N/large-image tensor from starving the device.
    """
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var channels = Int(channels_arg)
    var runs = Int(runs_arg)
    var hxw = Int(hxw_arg)
    var splits_n = Int(splits_n_arg)
    var splits_j = Int(splits_j_arg)
    var repass = Int(repass_arg) != 0
    var splits = splits_n * splits_j
    var meta = 2 * splits * channels

    var c = Int(block_idx.x)
    var k = Int(block_idx.y)
    var tid = Int(thread_idx.x)

    # Assumed mean: the channel's very first element, identical in every shard
    # of this channel so the partial moment pairs merge by addition. On a
    # re-pass it is the accurate mean instead, which leaves no cancellation.
    var shift = in_ptr[c * hxw].cast[DType.float32]()
    if repass:
        # Uniform across the block: this early exit cannot desynchronize the
        # barriers inside `block.sum` below.
        if ws_ptr[meta + channels + c] == 0:
            return
        shift = ws_ptr[meta + c]

    var chunk_n = ceildiv(runs, splits_n)
    var n0 = (k // splits_j) * chunk_n
    var n1 = min(runs, n0 + chunk_n)
    var chunk_j = ceildiv(hxw, splits_j)
    var j0 = (k % splits_j) * chunk_j
    var jn = min(hxw, j0 + chunk_j) - j0

    var s_tot = Float32(0)
    var q_tot = Float32(0)
    _bn_scan_shard(
        in_ptr, c, channels, hxw, n0, n1, j0, jn, tid, shift, s_tot, q_tot
    )
    # Outside every range guard above: each thread of the block must reach
    # these (they barrier internally).
    var bs = block.sum[block_size=BN_THREADS](s_tot)
    var bq = block.sum[block_size=BN_THREADS](q_tot)
    if tid == 0:
        ws_ptr[k * channels + c] = bs
        ws_ptr[(splits + k) * channels + c] = bq


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BN_THREADS))
)
@__name(t"batch_norm_moments_merge_{dtype}")
def _bn_merge_kernel[
    dtype: DType, sdtype: DType
](
    mean_out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    invstd_out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    run_mean_ptr: UnsafePointer[Scalar[sdtype], MutAnyOrigin],
    run_var_ptr: UnsafePointer[Scalar[sdtype], MutAnyOrigin],
    ws_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    channels_arg: Int64,
    splits_arg: Int64,
    count_arg: Int64,
    hxw_arg: Int64,
    eps: Float32,
    momentum: Float32,
    has_running_arg: Int64,
    finalize_arg: Int64,
):
    """One thread per channel: sum that channel's shard partials.

    Run twice. The first pass (`finalize_arg == 0`) only decides whether the
    assumed mean cost too many significand bits, recording the accurate mean
    and a flag the re-pass scan consumes. The second pass writes every output
    — for an unflagged channel the workspace still holds the first pass's
    partials, untouched, so re-summing gives the same answer. Splitting it
    this way is what keeps the momentum update EXACTLY ONCE: it lives only in
    the finalizing pass, which cannot recover a running value it had already
    overwritten.
    """
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var channels = Int(channels_arg)
    var splits = Int(splits_arg)
    var count = Int(count_arg)
    var meta = 2 * splits * channels
    var finalize = Int(finalize_arg) != 0
    var c = Int(block_idx.x) * BN_THREADS + Int(thread_idx.x)
    if c >= channels:
        return

    var s = Float32(0)
    var q = Float32(0)
    for k in range(splits):
        s += ws_ptr[k * channels + c]
        q += ws_ptr[(splits + k) * channels + c]
    var nf = Float32(count)
    var shift = in_ptr[c * Int(hxw_arg)].cast[DType.float32]()
    if finalize and ws_ptr[meta + channels + c] != 0:
        shift = ws_ptr[meta + c]
    var mean = shift + s / nf

    if not finalize:
        if _moment_cancels(s, q, count):
            ws_ptr[meta + c] = mean
            ws_ptr[meta + channels + c] = Float32(1)
        else:
            ws_ptr[meta + channels + c] = Float32(0)
        return

    _bn_finalize(
        mean_out_ptr,
        invstd_out_ptr,
        run_mean_ptr,
        run_var_ptr,
        c,
        mean,
        q - s * s / nf,
        count,
        eps,
        momentum,
        Int(has_running_arg) != 0,
    )


@always_inline
def _bn_splits(
    channels: Int, runs: Int, hxw: Int, vec_slots: Int, ctx: DeviceContext
) -> Tuple[Int, Int]:
    """How many ways to shard a channel's (run, position) space.

    `(1, 1)` — the common answer once there are as many channels as SMs — is
    what lets the caller take the fused route, which finalizes inside the
    statistics kernel and launches nothing else. Runs are sharded first
    because a run boundary costs nothing; only when the sample count runs out
    is the interior of a run cut, and never below one full vector per lane.
    """
    var sm_count = _device_sm_count(ctx)
    if channels >= sm_count or channels <= 0:
        return (1, 1)
    var want = ceildiv(BN_BLOCKS_PER_SM * sm_count, channels)
    var splits_n = max(1, min(want, runs // BN_MIN_RUNS_PER_SPLIT))
    var splits_j = 1
    if splits_n < want:
        splits_j = max(1, min(ceildiv(want, splits_n), vec_slots))
    return (splits_n, splits_j)


def enqueue_batch_norm_stats[
    dtype: DType, sdtype: DType
](
    mean_addr: Int,
    invstd_addr: Int,
    run_mean_addr: Int,
    run_var_addr: Int,
    in_addr: Int,
    channels: Int,
    runs: Int,
    hxw: Int,
    eps: Float32,
    momentum: Float32,
    has_running: Bool,
    ctx: DeviceContext,
) raises:
    """Per-channel mean and 1/sqrt(biased var + eps) over `runs * hxw`
    elements, plus the ATen running-statistics update."""
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    var in_ptr = _make_ptr[dtype](in_addr).as_unsafe_any_origin().as_immutable()
    var mean_ptr = _make_ptr[DType.float32](mean_addr).as_unsafe_any_origin()
    var invstd_ptr = _make_ptr[DType.float32](
        invstd_addr
    ).as_unsafe_any_origin()
    var run_mean_ptr = _make_ptr[sdtype](run_mean_addr).as_unsafe_any_origin()
    var run_var_ptr = _make_ptr[sdtype](run_var_addr).as_unsafe_any_origin()
    var count = runs * hxw

    comptime V = 16 // size_of[dtype]()
    var splits = _bn_splits(
        channels, runs, hxw, max(1, hxw // (V * BN_THREADS)), ctx
    )
    var splits_n = splits[0]
    var splits_j = splits[1]
    if splits_n == 1 and splits_j == 1:
        # Fused: one block owns a whole channel, so it handles its own
        # cancellation re-pass and finalizes in place. No workspace, one
        # launch.
        _enqueue_cached[_bn_moments_fused_kernel[dtype, sdtype]](
            ctx,
            String(t"bn_moments_fused_{dtype}_{sdtype}"),
            channels,
            1,
            1,
            BN_THREADS,
            mean_ptr,
            invstd_ptr,
            run_mean_ptr,
            run_var_ptr,
            in_ptr,
            Int64(channels),
            Int64(runs),
            Int64(hxw),
            eps,
            momentum,
            Int64(1 if has_running else 0),
        )
        return

    var n_splits = splits_n * splits_j
    # Two meta slots per channel sit past the partials: the accurate mean and
    # the cancellation flag the first merge writes.
    var ws = ctx.enqueue_create_buffer[DType.float32](
        2 * channels * n_splits + 2 * channels
    )
    var ws_ptr = ws.unsafe_ptr().as_unsafe_any_origin()
    var merge_blocks = ceildiv(channels, BN_THREADS)

    # Pass 0 assumes each channel's first element is a usable mean; its merge
    # flags the channels where `q - s^2/n` cancelled, and pass 1 re-reads ONLY
    # those. Both passes are enqueued unconditionally because the decision is
    # made on the device — testing it on the host would need a sync.
    for repass in range(2):
        _enqueue_cached[_bn_moments_kernel[dtype]](
            ctx,
            String(t"bn_moments_{dtype}"),
            channels,
            n_splits,
            1,
            BN_THREADS,
            ws_ptr,
            in_ptr,
            Int64(channels),
            Int64(runs),
            Int64(hxw),
            Int64(splits_n),
            Int64(splits_j),
            Int64(repass),
        )
        _enqueue_cached[_bn_merge_kernel[dtype, sdtype]](
            ctx,
            String(t"bn_merge_{dtype}_{sdtype}"),
            merge_blocks,
            1,
            1,
            BN_THREADS,
            mean_ptr,
            invstd_ptr,
            run_mean_ptr,
            run_var_ptr,
            ws_ptr,
            in_ptr,
            Int64(channels),
            Int64(n_splits),
            Int64(count),
            Int64(hxw),
            eps,
            momentum,
            Int64(1 if has_running else 0),
            Int64(repass),
        )
    # Dropping `ws` schedules a stream-ordered free after the kernels.
    _ = ws^


def enqueue_batch_norm_elementwise[
    dtype: DType, pdtype: DType, sdtype: DType, from_invstd: Bool
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
    planes: Int,
    has_weight: Bool,
    has_bias: Bool,
    ctx: DeviceContext,
    save_mean_addr: Int = 0,
    save_invstd_addr: Int = 0,
) raises:
    """`out = (x - mean[c]) * invstd[c] * gamma[c] + beta[c]` over an NC...
    contiguous tensor, `inner` being the product of the dims after C."""
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    var out_ptr = _make_ptr[dtype](out_addr).as_unsafe_any_origin()
    var in_ptr = _make_ptr[dtype](in_addr).as_unsafe_any_origin().as_immutable()
    var mean_ptr = (
        _make_ptr[sdtype](mean_addr).as_unsafe_any_origin().as_immutable()
    )
    var var_ptr = (
        _make_ptr[sdtype](var_addr).as_unsafe_any_origin().as_immutable()
    )
    var gamma_ptr = (
        _make_ptr[pdtype](gamma_addr).as_unsafe_any_origin().as_immutable()
    )
    var beta_ptr = (
        _make_ptr[pdtype](beta_addr).as_unsafe_any_origin().as_immutable()
    )
    var save_mean_ptr = _make_ptr[sdtype](save_mean_addr).as_unsafe_any_origin()
    var save_invstd_ptr = _make_ptr[sdtype](
        save_invstd_addr
    ).as_unsafe_any_origin()
    var hw = 1 if has_weight else 0
    var hb = 1 if has_bias else 0

    comptime V = 16 // size_of[dtype]()
    # A plane starts at `plane * inner`, so `inner % V == 0` plus a 16-byte
    # aligned base is what makes every plane's vectors land on a boundary.
    var vectorized = (
        _vec16_phase[dtype](in_addr) == 0
        and _vec16_phase[dtype](out_addr) == 0
        and inner % V == 0
    )
    var slots = inner // V if vectorized else inner
    var grid_x = ceildiv(slots, BN_THREADS)
    var grid_y = max(1, min(planes, _MAX_GRID))
    if vectorized:
        _enqueue_cached[
            _bn_elementwise_kernel[dtype, pdtype, sdtype, V, from_invstd]
        ](
            ctx,
            String(t"bn_elemwise_{dtype}_{pdtype}_{sdtype}_{from_invstd}_v"),
            grid_x,
            grid_y,
            1,
            BN_THREADS,
            out_ptr,
            in_ptr,
            mean_ptr,
            var_ptr,
            gamma_ptr,
            beta_ptr,
            save_mean_ptr,
            save_invstd_ptr,
            eps,
            Int64(slots),
            Int64(channels),
            Int64(planes),
            Int64(hw),
            Int64(hb),
        )
        return
    _enqueue_cached[
        _bn_elementwise_kernel[dtype, pdtype, sdtype, 1, from_invstd]
    ](
        ctx,
        String(t"bn_elemwise_{dtype}_{pdtype}_{sdtype}_{from_invstd}_s"),
        grid_x,
        grid_y,
        1,
        BN_THREADS,
        out_ptr,
        in_ptr,
        mean_ptr,
        var_ptr,
        gamma_ptr,
        beta_ptr,
        save_mean_ptr,
        save_invstd_ptr,
        eps,
        Int64(slots),
        Int64(channels),
        Int64(planes),
        Int64(hw),
        Int64(hb),
    )
