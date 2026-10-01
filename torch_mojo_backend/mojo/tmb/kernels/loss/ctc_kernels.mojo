"""CTC loss: the alpha (forward) and beta recursions of the forward-backward
algorithm in log space, and the gradient, for float32 / float64 log-probs
and int32 / int64 targets.

Ported from torch's CUDA kernels (aten/src/ATen/native/cuda/LossCTC.cu at
v2.14.0, after Graves et al. 2006): `ctc_loss_log_alpha_gpu_kernel`,
`ctc_loss_backward_log_beta_gpu_kernel` and the per-(sample, timestep)
`ctc_loss_backward_collect_gpu_kernel`, with the same log-sum-exp
formulation (`log(exp(a - m) + exp(b - m) + exp(c - m)) + m`, `m = 0` when
every term is -inf) and the same -inf fill of out-of-range cells.

Layouts: log_probs and the gradient are contiguous `[T, B, C]`, log_alpha /
log_beta contiguous `[B, T, 2 * L + 1]` (L the longest target), targets
contiguous: `[B, S]` rows, or the concatenation of every sample's target
(each block sums the lengths before it to find its own). One block per
sample; a sample's cells only depend on its own earlier timestep, so the
values do not depend on CUDA's (target, batch) thread geometry.

CUDA picks its gradient formula by problem size, and so does this port
(`ctc_is_large`): the collect kernel for small problems, and for large ones
`exp(log_probs)` minus a logsumexp over the blank positions plus an
atomicAdd kernel for the other labels. The two differ beyond rounding (an
impossible alignment gives NaN for every label on the small route, finite
values for the absent labels on the large one). The large route's
non-blank subtractions are summed by one thread per (sample, timestep) in
target order instead of CUDA's atomics: deterministic, same terms.
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from std.utils.numerics import inf, neg_inf

from tmb.kernels.common.libdevice_port import nv_exp, nv_expf, nv_log, nv_logf
from tmb.kernels.common.op_utils import _enqueue_cached, _make_ptr


@always_inline
def _target_prime[
    tdtype: DType
](
    targets: Pointer[Scalar[tdtype], MutAnyOrigin],
    offset: Int,
    idx: Int,
    blank: Int,
) -> Int:
    """`l'` = blank l_0 blank l_1 ... blank: index `idx` of the augmented
    target (no bound check, as CUDA's get_target_prime)."""
    if idx % 2 == 0:
        return blank
    return Int(targets[unsafe_offset=offset + idx // 2])


@always_inline
def _target_offset[
    tdtype: DType
](
    target_lengths: Pointer[Scalar[DType.int64], MutAnyOrigin],
    b: Int,
    batch_stride: Int,
) -> Int:
    """Where sample `b`'s targets start: `b * S` for `[B, S]` targets
    (`batch_stride = S`), else the sum of the earlier target lengths."""
    if batch_stride > 0:
        return b * batch_stride
    var pos = 0
    for i in range(b):
        pos += Int(target_lengths[unsafe_offset=i])
    return pos


@always_inline
def _exp[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """CUDA's `std::exp`: libdevice `__nv_exp` / `__nv_expf`, ported."""
    comptime if dtype == DType.float64:
        return rebind[Scalar[dtype]](nv_exp(rebind[Float64](x)))
    else:
        return rebind[Scalar[dtype]](nv_expf(rebind[Float32](x)))


@always_inline
def _log[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """CUDA's `std::log`: libdevice `__nv_log` / `__nv_logf`, ported."""
    comptime if dtype == DType.float64:
        return rebind[Scalar[dtype]](nv_log(rebind[Float64](x)))
    else:
        return rebind[Scalar[dtype]](nv_logf(rebind[Float32](x)))


@always_inline
def _lse3[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype], c: Scalar[dtype]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    var m = a
    if b > m:
        m = b
    if c > m:
        m = c
    if m == neg_inf[dtype]():
        m = 0
    return _log(_exp(a - m) + _exp(b - m) + _exp(c - m)) + m


@__name(t"ctc_loss_log_alpha_{dtype}_{tdtype}")
def _alpha_kernel[
    dtype: DType, tdtype: DType
](
    la_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    lp_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    il_ptr: Pointer[Scalar[DType.int64], MutAnyOrigin],
    tg_ptr: Pointer[Scalar[tdtype], MutAnyOrigin],
    tl_ptr: Pointer[Scalar[DType.int64], MutAnyOrigin],
    nll_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    max_input_arg: Int64,
    max_target_arg: Int64,
    batch_arg: Int64,
    labels_arg: Int64,
    blank_arg: Int64,
    tg_batch_stride_arg: Int64,
) where dtype.is_floating_point():
    comptime neginf = neg_inf[dtype]()
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var max_input = Int(max_input_arg)
    var s_count = 2 * Int(max_target_arg) + 1
    var batch = Int(batch_arg)
    var labels = Int(labels_arg)
    var blank = Int(blank_arg)
    var input_length = Int(il_ptr[unsafe_offset=b])
    var target_length = Int(tl_ptr[unsafe_offset=b])
    var lp_b = b * labels
    var lp_t = batch * labels
    var la_b = b * max_input * s_count
    var tg_off = _target_offset[tdtype](tl_ptr, b, Int(tg_batch_stride_arg))
    if input_length == 0:
        if tid == 0:
            nll_ptr[unsafe_offset=b] = (
                Scalar[dtype](0) if target_length == 0 else inf[dtype]()
            )
        return
    # t = 0: the three equations above eq (6).
    var block_s = 0
    while block_s < s_count:
        var s = tid + block_s
        var la = neginf
        if s == 0:
            la = lp_ptr[unsafe_offset=lp_b + blank]
        elif s == 1 and target_length > 0:
            la = lp_ptr[
                unsafe_offset=lp_b
                + _target_prime[tdtype](tg_ptr, tg_off, 1, blank)
            ]
        if s < s_count:
            la_ptr[unsafe_offset=la_b + s] = la
        block_s += bs
    block_s = 0
    while block_s < s_count:
        var s = tid + block_s
        var current_char = blank
        var have_three = False
        if s < 2 * target_length + 1 and target_length > 0:
            current_char = _target_prime[tdtype](tg_ptr, tg_off, s, blank)
            have_three = s > 1 and (
                _target_prime[tdtype](tg_ptr, tg_off, s - 2, blank)
                != current_char
            )
        for t in range(1, max_input):
            barrier()
            var row = la_b + t * s_count
            var prev = row - s_count
            if t < input_length and s < 2 * target_length + 1:
                var la1 = la_ptr[unsafe_offset=prev + s]
                var la2 = (
                    la_ptr[unsafe_offset=prev + s - 1] if s > 0 else neginf
                )
                var la3 = la_ptr[
                    unsafe_offset=prev + s - 2
                ] if have_three else neginf
                la_ptr[unsafe_offset=row + s] = (
                    _lse3(la1, la2, la3)
                    + lp_ptr[unsafe_offset=t * lp_t + lp_b + current_char]
                )
            elif s < s_count:
                la_ptr[unsafe_offset=row + s] = neginf
        block_s += bs
    barrier()
    # The loss, eq (8).
    if tid == 0:
        var last = la_b + (input_length - 1) * s_count
        var l1 = la_ptr[unsafe_offset=last + 2 * target_length]
        var l2 = (
            la_ptr[unsafe_offset=last + 2 * target_length - 1] if target_length
            > 0 else neginf
        )
        var m = l1 if l1 > l2 else l2
        if m == neginf:
            m = 0
        nll_ptr[unsafe_offset=b] = -(_log(_exp(l1 - m) + _exp(l2 - m)) + m)


@__name(t"ctc_loss_log_beta_{dtype}_{tdtype}")
def _beta_kernel[
    dtype: DType, tdtype: DType
](
    lb_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    lp_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    il_ptr: Pointer[Scalar[DType.int64], MutAnyOrigin],
    tg_ptr: Pointer[Scalar[tdtype], MutAnyOrigin],
    tl_ptr: Pointer[Scalar[DType.int64], MutAnyOrigin],
    max_input_arg: Int64,
    max_target_arg: Int64,
    batch_arg: Int64,
    labels_arg: Int64,
    blank_arg: Int64,
    tg_batch_stride_arg: Int64,
) where dtype.is_floating_point():
    """ctc_loss_backward_log_beta_gpu_kernel over a -inf filled log_beta."""
    comptime neginf = neg_inf[dtype]()
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var max_input = Int(max_input_arg)
    var max_target = Int(max_target_arg)
    var s_count = 2 * max_target + 1
    var batch = Int(batch_arg)
    var labels = Int(labels_arg)
    var blank = Int(blank_arg)
    var input_length = Int(il_ptr[unsafe_offset=b])
    var target_length = Int(tl_ptr[unsafe_offset=b])
    var lp_b = b * labels
    var lp_t = batch * labels
    var lb_b = b * max_input * s_count
    var tg_off = _target_offset[tdtype](tl_ptr, b, Int(tg_batch_stride_arg))
    if input_length == 0:
        return
    var top = 2 * max_target - (2 * max_target % bs)
    # The initialization before eq (10), at t = input_length - 1.
    var block_s = top
    while block_s >= 0:
        var s = tid + block_s
        var lb = neginf
        var lp_last = (input_length - 1) * lp_t + lp_b
        if s == 2 * target_length:
            lb = lp_ptr[unsafe_offset=lp_last + blank]
        elif s == 2 * target_length - 1:
            lb = lp_ptr[
                unsafe_offset=lp_last
                + _target_prime[tdtype](tg_ptr, tg_off, s, blank)
            ]
        if s < s_count:
            lb_ptr[unsafe_offset=lb_b + (input_length - 1) * s_count + s] = lb
        block_s -= bs
    block_s = top
    while block_s >= 0:
        var s = tid + block_s
        var current = blank
        var have_three = False
        if s < 2 * target_length + 1 and target_length > 0:
            current = _target_prime[tdtype](tg_ptr, tg_off, s, blank)
            have_three = s < 2 * target_length - 1 and (
                _target_prime[tdtype](tg_ptr, tg_off, s + 2, blank) != current
            )
        var t = max_input - 2
        while t >= 0:
            barrier()
            var row = lb_b + t * s_count
            var nxt = row + s_count
            if t < input_length - 1 and s < 2 * target_length + 1:
                var lb1 = lb_ptr[unsafe_offset=nxt + s]
                var lb2 = (
                    lb_ptr[unsafe_offset=nxt + s + 1] if s
                    < 2 * target_length else neginf
                )
                var lb3 = lb_ptr[
                    unsafe_offset=nxt + s + 2
                ] if have_three else neginf
                lb_ptr[unsafe_offset=row + s] = (
                    _lse3(lb1, lb2, lb3)
                    + lp_ptr[unsafe_offset=t * lp_t + lp_b + current]
                )
            elif s < s_count and (
                (target_length == 0 and s > 0)
                or s >= 2 * target_length + 1
                or t >= input_length
            ):
                lb_ptr[unsafe_offset=row + s] = neginf
            t -= 1
        block_s -= bs


@__name(t"ctc_loss_collect_{dtype}_{tdtype}")
def _collect_kernel[
    dtype: DType, tdtype: DType
](
    gr_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    go_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    la_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    lb_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    lp_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    il_ptr: Pointer[Scalar[DType.int64], MutAnyOrigin],
    tg_ptr: Pointer[Scalar[tdtype], MutAnyOrigin],
    tl_ptr: Pointer[Scalar[DType.int64], MutAnyOrigin],
    nll_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    max_input_arg: Int64,
    max_target_arg: Int64,
    batch_arg: Int64,
    labels_arg: Int64,
    blank_arg: Int64,
    tg_batch_stride_arg: Int64,
    zero_infinity_arg: Int64,
) where dtype.is_floating_point():
    """ctc_loss_backward_collect_gpu_kernel: one thread per (sample,
    timestep) over a -inf filled gradient, which first log-accumulates
    alpha * beta per label and then becomes eq (16) times grad_out (0 past
    the input length, or for an infinite loss under zero_infinity)."""
    comptime neginf = neg_inf[dtype]()
    var b = Int(block_idx.y)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var max_input = Int(max_input_arg)
    var batch = Int(batch_arg)
    if t >= max_input or b >= batch:
        return
    var s_count = 2 * Int(max_target_arg) + 1
    var labels = Int(labels_arg)
    var blank = Int(blank_arg)
    var input_length = Int(il_ptr[unsafe_offset=b])
    var target_length = Int(tl_ptr[unsafe_offset=b])
    var g_row = t * batch * labels + b * labels
    var ab_row = (b * max_input + t) * s_count
    var tg_off = _target_offset[tdtype](tl_ptr, b, Int(tg_batch_stride_arg))
    for s in range(s_count):
        if s < 2 * target_length + 1:
            var c = _target_prime[tdtype](tg_ptr, tg_off, s, blank)
            var lab = (
                la_ptr[unsafe_offset=ab_row + s]
                + lb_ptr[unsafe_offset=ab_row + s]
            )
            var lcab = gr_ptr[unsafe_offset=g_row + c]
            if lcab == neginf:
                gr_ptr[unsafe_offset=g_row + c] = lab
            else:
                var m = lcab if lcab > lab else lab
                gr_ptr[unsafe_offset=g_row + c] = (
                    _log(_exp(lcab - m) + _exp(lab - m)) + m
                )
    var nll = nll_ptr[unsafe_offset=b]
    var gr = go_ptr[unsafe_offset=b]
    var keep = t < input_length and (
        Int(zero_infinity_arg) == 0 or nll != inf[dtype]()
    )
    for c in range(labels):
        var res = gr_ptr[unsafe_offset=g_row + c]
        if keep:
            var lp = lp_ptr[unsafe_offset=g_row + c]
            gr_ptr[unsafe_offset=g_row + c] = (
                _exp(lp) - _exp(res + nll - lp)
            ) * gr
        else:
            gr_ptr[unsafe_offset=g_row + c] = 0


@__name(t"ctc_loss_collect_large_{dtype}_{tdtype}")
def _collect_large_kernel[
    dtype: DType, tdtype: DType
](
    gr_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    go_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    la_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    lb_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    lp_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    il_ptr: Pointer[Scalar[DType.int64], MutAnyOrigin],
    tg_ptr: Pointer[Scalar[tdtype], MutAnyOrigin],
    tl_ptr: Pointer[Scalar[DType.int64], MutAnyOrigin],
    nll_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    max_input_arg: Int64,
    max_target_arg: Int64,
    batch_arg: Int64,
    labels_arg: Int64,
    blank_arg: Int64,
    tg_batch_stride_arg: Int64,
    zero_infinity_arg: Int64,
) where dtype.is_floating_point():
    """LossCTC.cu's large-problem route, one thread per (sample, timestep):
    `exp(log_probs)`; the blank's `-= exp(logsumexp_s(alpha + beta at even
    s) + nll - lp_blank)`; `*= grad_out`; zero_infinity's `where`; then
    ctc_loss_backward_collect_nonblank_gpu_kernel's subtractions; then
    ctc_loss_zero_padded_gradients past the input length."""
    comptime neginf = neg_inf[dtype]()
    var b = Int(block_idx.y)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var max_input = Int(max_input_arg)
    var batch = Int(batch_arg)
    if t >= max_input or b >= batch:
        return
    var max_target = Int(max_target_arg)
    var s_count = 2 * max_target + 1
    var labels = Int(labels_arg)
    var blank = Int(blank_arg)
    var input_length = Int(il_ptr[unsafe_offset=b])
    var target_length = Int(tl_ptr[unsafe_offset=b])
    var g_row = t * batch * labels + b * labels
    var ab_row = (b * max_input + t) * s_count
    if t >= input_length:
        for c in range(labels):
            gr_ptr[unsafe_offset=g_row + c] = 0
        return
    var nll = nll_ptr[unsafe_offset=b]
    var gr = go_ptr[unsafe_offset=b]
    for c in range(labels):
        gr_ptr[unsafe_offset=g_row + c] = _exp(lp_ptr[unsafe_offset=g_row + c])
    # at::logsumexp over the max_target + 1 blank positions: an infinite
    # maximum is replaced by 0.
    var m = neginf
    for k in range(max_target + 1):
        var v = (
            la_ptr[unsafe_offset=ab_row + 2 * k]
            + lb_ptr[unsafe_offset=ab_row + 2 * k]
        )
        if v > m:
            m = v
    if m == neginf or m == -neginf:
        m = 0
    var acc = Scalar[dtype](0)
    for k in range(max_target + 1):
        acc += _exp(
            la_ptr[unsafe_offset=ab_row + 2 * k]
            + lb_ptr[unsafe_offset=ab_row + 2 * k]
            - m
        )
    var lse = _log(acc) + m
    var lp_blank = lp_ptr[unsafe_offset=g_row + blank]
    gr_ptr[unsafe_offset=g_row + blank] = gr_ptr[
        unsafe_offset=g_row + blank
    ] - _exp(lse + nll - lp_blank)
    var zeroed = Int(zero_infinity_arg) != 0 and nll == -neginf
    for c in range(labels):
        if zeroed:
            gr_ptr[unsafe_offset=g_row + c] = 0
        else:
            gr_ptr[unsafe_offset=g_row + c] = (
                gr_ptr[unsafe_offset=g_row + c] * gr
            )
    if zeroed:
        return
    var tg_off = _target_offset[tdtype](tl_ptr, b, Int(tg_batch_stride_arg))
    for s in range(target_length):
        var target = Int(tg_ptr[unsafe_offset=tg_off + s])
        var lp = lp_ptr[unsafe_offset=g_row + target]
        var term = (
            -_exp(
                la_ptr[unsafe_offset=ab_row + 2 * s + 1]
                + lb_ptr[unsafe_offset=ab_row + 2 * s + 1]
                + nll
                - lp
            )
            * gr
        )
        gr_ptr[unsafe_offset=g_row + target] = (
            gr_ptr[unsafe_offset=g_row + target] + term
        )


def ctc_is_large(max_input: Int, batch: Int, labels: Int) -> Bool:
    """LossCTC.cu's size heuristic for the gradient route."""
    return (2 * max_input + (24 * batch) // 10 + (2 * labels) // 10) > 450


def _block_for(max_target: Int) -> Int:
    """Threads per sample block: enough to cover the 2L + 1 augmented
    target in one chunk, up to 512 (values do not depend on it)."""
    var threads = 32
    while threads < 2 * max_target + 1 and threads < 512:
        threads *= 2
    return threads


def ctc_forward[
    dtype: DType, tdtype: DType
](
    la_addr: Int,
    nll_addr: Int,
    lp_addr: Int,
    il_addr: Int,
    tg_addr: Int,
    tl_addr: Int,
    max_input: Int,
    max_target: Int,
    batch: Int,
    labels: Int,
    blank: Int,
    tg_batch_stride: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    _enqueue_cached[_alpha_kernel[dtype, tdtype]](
        ctx,
        batch,
        1,
        1,
        _block_for(max_target),
        _make_ptr[dtype](la_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](lp_addr).as_unsafe_any_origin(),
        _make_ptr[DType.int64](il_addr).as_unsafe_any_origin(),
        _make_ptr[tdtype](tg_addr).as_unsafe_any_origin(),
        _make_ptr[DType.int64](tl_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](nll_addr).as_unsafe_any_origin(),
        Int64(max_input),
        Int64(max_target),
        Int64(batch),
        Int64(labels),
        Int64(blank),
        Int64(tg_batch_stride),
    )


def ctc_backward[
    dtype: DType, tdtype: DType
](
    grad_addr: Int,
    lb_addr: Int,
    go_addr: Int,
    la_addr: Int,
    lp_addr: Int,
    il_addr: Int,
    tg_addr: Int,
    tl_addr: Int,
    nll_addr: Int,
    max_input: Int,
    max_target: Int,
    batch: Int,
    labels: Int,
    blank: Int,
    tg_batch_stride: Int,
    zero_infinity: Bool,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """`grad` and `log_beta` arrive filled with -inf."""
    var lp = _make_ptr[dtype](lp_addr).as_unsafe_any_origin()
    var il = _make_ptr[DType.int64](il_addr).as_unsafe_any_origin()
    var tg = _make_ptr[tdtype](tg_addr).as_unsafe_any_origin()
    var tl = _make_ptr[DType.int64](tl_addr).as_unsafe_any_origin()
    var lb = _make_ptr[dtype](lb_addr).as_unsafe_any_origin()
    _enqueue_cached[_beta_kernel[dtype, tdtype]](
        ctx,
        batch,
        1,
        1,
        _block_for(max_target),
        lb,
        lp,
        il,
        tg,
        tl,
        Int64(max_input),
        Int64(max_target),
        Int64(batch),
        Int64(labels),
        Int64(blank),
        Int64(tg_batch_stride),
    )
    var threads = 32
    while threads < max_input and threads < 256:
        threads *= 2
    if ctc_is_large(max_input, batch, labels):
        _enqueue_cached[_collect_large_kernel[dtype, tdtype]](
            ctx,
            (max_input + threads - 1) // threads,
            batch,
            1,
            threads,
            _make_ptr[dtype](grad_addr).as_unsafe_any_origin(),
            _make_ptr[dtype](go_addr).as_unsafe_any_origin(),
            _make_ptr[dtype](la_addr).as_unsafe_any_origin(),
            lb,
            lp,
            il,
            tg,
            tl,
            _make_ptr[dtype](nll_addr).as_unsafe_any_origin(),
            Int64(max_input),
            Int64(max_target),
            Int64(batch),
            Int64(labels),
            Int64(blank),
            Int64(tg_batch_stride),
            Int64(1 if zero_infinity else 0),
        )
        return
    _enqueue_cached[_collect_kernel[dtype, tdtype]](
        ctx,
        (max_input + threads - 1) // threads,
        batch,
        1,
        threads,
        _make_ptr[dtype](grad_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](go_addr).as_unsafe_any_origin(),
        _make_ptr[dtype](la_addr).as_unsafe_any_origin(),
        lb,
        lp,
        il,
        tg,
        tl,
        _make_ptr[dtype](nll_addr).as_unsafe_any_origin(),
        Int64(max_input),
        Int64(max_target),
        Int64(batch),
        Int64(labels),
        Int64(blank),
        Int64(tg_batch_stride),
        Int64(1 if zero_infinity else 0),
    )
