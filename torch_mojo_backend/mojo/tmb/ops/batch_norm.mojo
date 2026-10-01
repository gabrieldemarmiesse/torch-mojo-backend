"""ATen ops: the batch-norm overloads beyond `native_batch_norm` itself, and
SyncBatchNorm's building blocks (see agents_docs/native_backend.md).

The training / inference forward and `native_batch_norm_backward` are
tmb/ops/nn.mojo's and tmb/ops/composed.mojo's; every overload here that
torch's CUDA backend implements by calling them (Normalization.cu:
`_batch_norm_legit_cuda*`, `_batch_norm_with_update_cuda*` on its
non-cuDNN route, `_new_batch_norm_backward_cuda`) reaches them through the
dispatcher the same way. The SyncBatchNorm blocks (`batch_norm_stats`,
`batch_norm_elemt`, `batch_norm_gather_stats*`, `batch_norm_backward_*`,
`batch_norm_update_stats`) are the `batch_norm_sync` kernel family.
"""
from std.utils import IndexList

from tmb.backend.abi import (
    Owned,
    Results,
    ST_FLOAT32,
    ST_FLOAT64,
    ST_INT32,
    ST_UINT8,
    T,
    TAG_SCALAR_DOUBLE,
    TAG_TENSOR,
    TAG_SCALAR_INT,
    Value,
    Values,
    bool_arg,
    f64_bits,
    index_error,
    _channels_last_strides,
    max_dtype,
    new_strided,
    new_tensor,
    none_arg,
    own,
    ret_owned,
    ret_ref,
    ret_tensor,
    retain,
    tensor_arg,
    v_bool,
    v_f64,
    v_int,
    v_is_none,
    v_tensor,
)
from tmb.backend.device import ctx_for, ctx_ptr
from tmb.backend.kernel_call import KernelCall
from tmb.kernels.common.op_utils import MAX_RANK, _f64_slot
from tmb.ops.common import (
    call_op,
    cast_to,
    check_out_as,
    copy_strided_into,
    fill_value,
    forward_args,
    is_channels_last_layout,
    like_layout,
    store_out,
)
from tmb.ops.data_movement import _scalar_type_name
from tmb.ops.loss import (
    Dense,
    OwnedPair,
    _expect_dtype,
    _loss_float,
    _same_device,
)
from tmb.ops.nn import (
    _bool_list,
    op_batch_norm_legit_no_training,
    op_native_batch_norm,
    sizes_str,
)
from tmb.backend.registry import Site, impl


def _acc_stype(t: T) -> Int32:
    """`toAccumulateType(dtype, /*is_cuda=*/true)`'s ScalarType."""
    return ST_FLOAT64 if t.stype == ST_FLOAT64 else ST_FLOAT32


def _vec(n: Int, stype: Int32, device: Int) raises -> Owned:
    var s = IndexList[MAX_RANK](1)
    s[MAX_RANK - 1] = n
    return own(new_tensor(s, 1, stype, device))


def _store_any(mut dst: T, var src: T) raises:
    """`store_out` that also converts the dtype, as the eval route's
    `save_mean.copy_(running_mean)` does."""
    if dst.stype == src.stype:
        store_out(dst, src^)
        return
    var held = own(src^)
    var cast = cast_to(held.t, dst.stype)
    store_out(dst, cast^)


# ---------------------------------------------------------------------------
# Forward overloads over native_batch_norm
# ---------------------------------------------------------------------------


def _native_bn(var args: List[Value]) raises -> Results:
    return call_op(String("aten::native_batch_norm"), String(""), args^, 3)


def _bn_store3(args: Values, rets: Values, var r: Results, out_i: Int) raises:
    """The three native_batch_norm results into the caller's `out`,
    `save_mean`, `save_invstd`."""
    var out = v_tensor(args[unsafe_offset=out_i])
    var save_mean = v_tensor(args[unsafe_offset=out_i + 1])
    var save_invstd = v_tensor(args[unsafe_offset=out_i + 2])
    store_out(out, r.take_tensor(0))
    _store_any(save_mean, r.take_tensor(1))
    _store_any(save_invstd, r.take_tensor(2))
    ret_ref(rets, 0, out)
    ret_ref(rets, 1, save_mean)
    ret_ref(rets, 2, save_invstd)


def _check_bn_out(args: Values, out_i: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    check_out_as(v_tensor(args[unsafe_offset=out_i]), a.stype, a)


# aten::native_batch_norm.out(Tensor input, Tensor? weight, Tensor? bias,
#   Tensor? running_mean, Tensor? running_var, bool training, float momentum,
#   float eps, *, Tensor(a!) out, Tensor(b!) save_mean,
#   Tensor(c!) save_invstd) -> (Tensor(a!), Tensor(b!), Tensor(c!))
# aten::_native_batch_norm_legit.out: the same schema with non-optional
# running statistics.
def op_native_batch_norm_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _check_bn_out(args, 8)
    _bn_store3(args, rets, _native_bn(forward_args(args, 8)), 8)


def _no_stats_args(args: Values) -> List[Value]:
    """`_native_batch_norm_legit.no_stats`'s arguments as native_batch_norm's:
    no running statistics."""
    var out = List[Value](capacity=8)
    out.append(args[unsafe_offset=0].copy())
    out.append(args[unsafe_offset=1].copy())
    out.append(args[unsafe_offset=2].copy())
    out.append(none_arg())
    out.append(none_arg())
    out.append(args[unsafe_offset=3].copy())
    out.append(args[unsafe_offset=4].copy())
    out.append(args[unsafe_offset=5].copy())
    return out^


# aten::_native_batch_norm_legit.no_stats(Tensor input, Tensor? weight,
#   Tensor? bias, bool training, float momentum, float eps)
#   -> (Tensor, Tensor, Tensor)
def op_batch_norm_legit_no_stats(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var r = _native_bn(_no_stats_args(args))
    for i in range(3):
        ret_tensor(rets, i, r.take_tensor(i))


# aten::_native_batch_norm_legit.no_stats_out(Tensor input, Tensor? weight,
#   Tensor? bias, bool training, float momentum, float eps, *,
#   Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd)
#   -> (Tensor(a!), Tensor(b!), Tensor(c!))
def op_batch_norm_legit_no_stats_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _check_bn_out(args, 6)
    _bn_store3(args, rets, _native_bn(_no_stats_args(args)), 6)


def _with_update_args(args: Values) -> List[Value]:
    """`_batch_norm_with_update`'s arguments as native_batch_norm's, with
    training=True."""
    var out = List[Value](capacity=8)
    for i in range(5):
        out.append(args[unsafe_offset=i].copy())
    out.append(bool_arg(True))
    out.append(args[unsafe_offset=5].copy())
    out.append(args[unsafe_offset=6].copy())
    return out^


# aten::_batch_norm_with_update(Tensor input, Tensor? weight, Tensor? bias,
#   Tensor(a!) running_mean, Tensor(b!) running_var, float momentum,
#   float eps) -> (Tensor, Tensor, Tensor, Tensor)
def op_batch_norm_with_update(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """`_batch_norm_with_update_cuda` on its native route: a training
    native_batch_norm, and an empty uint8 `reserve` (cuDNN's workspace,
    which only the cuDNN backward reads)."""
    var a = v_tensor(args[unsafe_offset=0])
    var r = _native_bn(_with_update_args(args))
    for i in range(3):
        ret_tensor(rets, i, r.take_tensor(i))
    ret_tensor(
        rets, 3, new_tensor(IndexList[MAX_RANK](0), 1, ST_UINT8, a.device)
    )


# aten::_batch_norm_with_update.out(Tensor input, Tensor? weight,
#   Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var,
#   float momentum, float eps, *, Tensor(d!) out, Tensor(e!) save_mean,
#   Tensor(f!) save_invstd, Tensor(g!) reserve)
#   -> (Tensor(d!), Tensor(e!), Tensor(f!), Tensor(g!))
def op_batch_norm_with_update_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _check_bn_out(args, 7)
    _bn_store3(args, rets, _native_bn(_with_update_args(args)), 7)
    ret_ref(rets, 3, v_tensor(args[unsafe_offset=10]))


# aten::batch_norm_backward(Tensor grad_out, Tensor input, Tensor weight,
#   Tensor? running_mean, Tensor? running_var, Tensor? save_mean,
#   Tensor? save_var, bool update, float eps, bool[3] output_mask,
#   Tensor reserve) -> (Tensor, Tensor, Tensor)
def op_batch_norm_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """`_new_batch_norm_backward_cuda` on its native route:
    native_batch_norm_backward with `train = update` (the empty `reserve`
    of the native forward is not read)."""
    var r = call_op(
        String("aten::native_batch_norm_backward"),
        String(""),
        forward_args(args, 10),
        3,
    )
    var mask = _bool_list(args[unsafe_offset=9])
    for i in range(3):
        if i >= len(mask) or not mask[i] or r[i].tag != TAG_TENSOR:
            # An unrequested gradient is undefined, as on CUDA's native route.
            rets[unsafe_offset=i] = none_arg()
        else:
            ret_tensor(rets, i, r.take_tensor(i))


# ---------------------------------------------------------------------------
# SyncBatchNorm building blocks
# ---------------------------------------------------------------------------


struct Planes(Movable):
    """A batch-norm operand as the dense `[N, C, HxW]` the kernels read:
    contiguous NCHW (a copy when needed), or a channels-last input read
    where it lies (`cl`)."""

    var dense: Dense
    var cl: Bool
    var n: Int
    var c: Int
    var hxw: Int

    def __init__(out self, t: T) raises:
        if t.rank < 2:
            index_error(
                String("Dimension out of range (expected to be in range of [")
                + String(-max(t.rank, 1))
                + ", "
                + String(max(t.rank, 1) - 1)
                + "], but got 1)"
            )
        self.cl = is_channels_last_layout(t)
        if self.cl:
            self.dense = Dense(t, True)
        else:
            self.dense = Dense(t)
        self.n = t.dim(0)
        self.c = t.dim(1)
        self.hxw = 1
        for i in range(2, t.rank):
            self.hxw *= t.dim(i)


comptime L_INPUT = 1
comptime L_GRAD = 2
comptime L_OUT = 4
comptime L_FLAGS_SHIFT = 4  # kernels/batch_norm_sync/kernels.mojo's layout bits


def _out_like(p: Planes, a: T) raises -> Owned:
    """The output the kernels write in `p`'s layout: channels-last for a
    channels-last input (CUDA's `empty_like`), else contiguous."""
    if p.cl:
        return own(
            new_strided(
                a.shape,
                _channels_last_strides(a.shape, a.rank),
                a.rank,
                a.stype,
                a.device,
            )
        )
    return own(new_tensor(a.shape, a.rank, a.stype, a.device))


def _opt(args: Values, i: Int) -> Bool:
    return not v_is_none(args[unsafe_offset=i])


def _stats_launch(
    p: Planes,
    a: T,
    mean: T,
    var_: T,
    running: Optional[Tuple[T, T]],
    mode: Int,
    eps: Float64,
    momentum: Float64,
) raises:
    var ctx = ctx_for(a.device)
    var call = KernelCall("batch_norm_sync", "BnStats")
    call.arg_dtype(0, a.dtype)
    var rm = 0
    var rv = 0
    if running:
        var pair = running.value().copy()
        call.arg_dtype(1, pair[0].dtype)
        rm = pair[0].ptr
        rv = pair[1].ptr
    else:
        call.arg_dtype(1, a.dtype)
    call.int(mean.ptr)
    call.int(var_.ptr)
    call.int(rm)
    call.int(rv)
    call.int(p.dense.t.ptr)
    var params = List[Int]()
    params.append(p.c)
    params.append(p.n)
    params.append(p.hxw)
    params.append(mode)
    params.append(1 if running else 0)
    params.append(L_INPUT if p.cl else 0)
    call.tuple(params)
    call.f64(eps)
    call.f64(momentum)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx


# aten::batch_norm_stats(Tensor input, float eps) -> (Tensor, Tensor)
def op_batch_norm_stats(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """batch_norm_stats_cuda: per-channel mean and InvStd(biased var, eps)
    in the accumulation dtype."""
    var a = v_tensor(args[unsafe_offset=0])
    var eps = v_f64(args[unsafe_offset=1])
    _loss_float(a)
    var p = Planes(a)
    var mean = _vec(p.c, _acc_stype(a), a.device)
    var invstd = _vec(p.c, _acc_stype(a), a.device)
    if p.c > 0:
        _stats_launch(p, a, mean.t, invstd.t, None, 1, eps, 0.0)
    _ = p^
    ret_owned(rets, 0, mean)
    ret_owned(rets, 1, invstd)


# aten::batch_norm_update_stats(Tensor input, Tensor? running_mean,
#   Tensor? running_var, float momentum) -> (Tensor, Tensor)
def op_batch_norm_update_stats(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """batch_norm_update_stats_cuda: per-channel mean and biased variance,
    and the running statistics' momentum update with the unbiased one."""
    var a = v_tensor(args[unsafe_offset=0])
    var momentum = v_f64(args[unsafe_offset=3])
    _loss_float(a)
    var p = Planes(a)
    if a.numel == 0:
        raise Error(
            (
                "input tensor must have at least one element, but got"
                " input_sizes = "
            ),
            sizes_str(a),
        )
    if _opt(args, 1) != _opt(args, 2):
        raise Error(
            "Expected running_mean->defined() == running_var->defined() to be"
            " true, but got false.  (Could this error message be improved?  If"
            " so, please report an enhancement request to PyTorch.)"
        )
    var mean = _vec(p.c, _acc_stype(a), a.device)
    var var_ = _vec(p.c, _acc_stype(a), a.device)
    var running: Optional[Tuple[T, T]] = None
    if _opt(args, 1):
        var rm = v_tensor(args[unsafe_offset=1])
        var rv = v_tensor(args[unsafe_offset=2])
        _loss_float(rm)
        _same_device(a, rm)
        _same_device(a, rv)
        _expect_dtype(rv, rm)
        if not rm.contig or not rv.contig or rm.numel != p.c or rv.numel != p.c:
            raise Error(
                "batch_norm_update_stats: running statistics must be"
                " contiguous [C] tensors"
            )
        running = (rm.copy(), rv.copy())
    _stats_launch(p, a, mean.t, var_.t, running, 0, 0.0, momentum)
    _ = p^
    ret_owned(rets, 0, mean)
    ret_owned(rets, 1, var_)


def _expect_type(t: T, stype: Int32, name: StaticString) raises:
    """Normalization.cuh `get_packed_accessor`'s dtype check."""
    if t.stype != stype:
        raise Error(
            "Expected ",
            name,
            " to have type ",
            _scalar_type_name(max_dtype(stype)),
            " but got ",
            _scalar_type_name(t.dtype),
        )


def _acc_of(stype: Int32) -> Int32:
    return ST_FLOAT64 if stype == ST_FLOAT64 else ST_FLOAT32


def _channels_last_dense(t: T) -> Bool:
    """`is_contiguous(ChannelsLast)` / `(ChannelsLast3d)`: C fastest, then
    the spatial dims innermost-first, then N; size-1 dims are free."""
    if t.rank != 4 and t.rank != 5:
        return False
    var order = List[Int]()
    order.append(1)
    for d in range(t.rank - 1, 1, -1):
        order.append(d)
    order.append(0)
    var expected = 1
    for d in order:
        if t.dim(d) == 1:
            continue
        if t.stride(d) != expected:
            return False
        expected *= t.dim(d)
    return True


def _uses_channels_last(t: T) -> Bool:
    """Normalization.cu `batch_norm_use_channels_last_kernels`."""
    return _channels_last_dense(t) or (t.contig and t.stride(1) == 1)


def _channel_vec(
    args: Values, i: Int, a: T, c: Int, what: StaticString
) raises -> Dense:
    var t = v_tensor(args[unsafe_offset=i])
    _same_device(a, t)
    _loss_float(t)
    if t.numel != c:
        raise Error(what, " must have ", c, " elements, but got ", sizes_str(t))
    return Dense(t)


def _as_stype(var d: Dense, stype: Int32) raises -> Dense:
    """`d` converted to `stype` (a fresh dense tensor), or itself."""
    if d.t.stype == stype:
        return d^
    var held = own(cast_to(d.t, stype))
    var out = Dense(held.t)
    if not out.mine:
        # `held` is already dense: hand its reference to `out`.
        out.mine = True
        _ = held.take()
    return out^


def _elemt(args: Values) raises -> Owned:
    """batch_norm_elemt_cuda: `gamma * (x - mean) * invstd + beta`."""
    var a = v_tensor(args[unsafe_offset=0])
    _loss_float(a)
    var p = Planes(a)
    if a.contig and not _uses_channels_last(a):
        # batch_norm_elementwise's Contiguous route: typed accessors, the
        # affine in `stat_scalar_t` (the accumulation dtype when it is
        # "mixed", else the input's), the statistics in its acc_type.
        var stat = a.stype
        var first = 1 if _opt(args, 1) else (2 if _opt(args, 2) else 0)
        if first != 0 and v_tensor(args[unsafe_offset=first]).stype != a.stype:
            stat = _acc_of(a.stype)
        if _opt(args, 1):
            _expect_type(v_tensor(args[unsafe_offset=1]), stat, "weight")
        if _opt(args, 2):
            _expect_type(v_tensor(args[unsafe_offset=2]), stat, "bias")
        _expect_type(v_tensor(args[unsafe_offset=3]), _acc_of(stat), "mean")
        _expect_type(v_tensor(args[unsafe_offset=4]), _acc_of(stat), "invstd")
    # Every route reads each statistic in its own dtype into an acc_type
    # computation (the typed routes' checks above pin the dtypes; the
    # TensorIterator one converts each operand on its own), so each is
    # handed to the kernel converted -- exactly -- to the accumulation dtype.
    var mean = _as_stype(
        _channel_vec(args, 3, a, p.c, "mean"), _acc_of(a.stype)
    )
    var invstd = _as_stype(
        _channel_vec(args, 4, a, p.c, "invstd"), _acc_of(a.stype)
    )
    # Every route computes in the accumulation dtype with each parameter
    # read in its own dtype (the typed routes' checks above require one
    # dtype; the TensorIterator route mixes them), so both are handed to the
    # kernel converted to that dtype -- exact, nothing is rounded down.
    var acc = _acc_of(a.stype)
    var pdtype = max_dtype(acc)
    var w_ptr = 0
    var b_ptr = 0
    var w: Optional[Dense] = None
    var b: Optional[Dense] = None
    if _opt(args, 1):
        w = _as_stype(_channel_vec(args, 1, a, p.c, "weight"), acc)
        w_ptr = w.value().t.ptr
    if _opt(args, 2):
        var bd = _as_stype(_channel_vec(args, 2, a, p.c, "bias"), acc)
        b_ptr = bd.t.ptr
        b = bd^
    var out = _out_like(p, a)
    if a.numel > 0:
        var ctx = ctx_for(a.device)
        var call = KernelCall("batch_norm_sync", "BnElemt")
        call.arg_dtype(0, a.dtype)
        call.arg_dtype(1, mean.t.dtype)
        call.arg_dtype(2, pdtype)
        call.int(out.t.ptr)
        call.int(p.dense.t.ptr)
        call.int(w_ptr)
        call.int(b_ptr)
        call.int(mean.t.ptr)
        call.int(invstd.t.ptr)
        var params = List[Int]()
        params.append(p.c)
        params.append(p.hxw)
        params.append(a.numel)
        params.append((L_INPUT | L_OUT) if p.cl else 0)
        call.tuple(params)
        call.int(ctx_ptr(ctx))
        call.run()
        _ = ctx
    _ = p^
    _ = mean^
    _ = invstd^
    _ = w^
    _ = b^
    # `empty_like(self)`: a dense permuted input keeps its strides.
    return like_layout(out^, a, True)


# aten::batch_norm_elemt(Tensor input, Tensor? weight, Tensor? bias,
#   Tensor mean, Tensor invstd, float eps) -> Tensor
def op_batch_norm_elemt(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var r = _elemt(args)
    ret_owned(rets, 0, r)


# aten::batch_norm_elemt.out(Tensor input, Tensor? weight, Tensor? bias,
#   Tensor mean, Tensor invstd, float eps, *, Tensor(a!) out) -> Tensor(a!)
def op_batch_norm_elemt_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var dst = v_tensor(args[unsafe_offset=6])
    check_out_as(dst, a.stype, a)
    var r = _elemt(args)
    store_out(dst, r.take())
    ret_ref(rets, 0, dst)


def _gather(args: Values, counts: T) raises -> OwnedPair:
    """batch_norm_gather_stats_with_counts_cuda over a `counts` vector of
    the kernel's scalar_t (the running mean's dtype, else the input's)."""
    var a = v_tensor(args[unsafe_offset=0])
    var mean = v_tensor(args[unsafe_offset=1])
    var invstd = v_tensor(args[unsafe_offset=2])
    var momentum = v_f64(args[unsafe_offset=5])
    var eps = v_f64(args[unsafe_offset=6])
    if mean.rank != 2:
        raise Error(
            (
                "batch_norm_gather_stats_with_counts: expected mean to be"
                " 2-dimensional (world_size, num_features), but got mean of"
                " sizes "
            ),
            sizes_str(mean),
        )
    if not invstd.same_shape(mean):
        raise Error(
            (
                "batch_norm_gather_stats_with_counts: expected invstd to have"
                " the same shape as mean ("
            ),
            sizes_str(mean),
            "), but got invstd of sizes ",
            sizes_str(invstd),
        )
    var world = mean.dim(0)
    var features = mean.dim(1)
    if counts.numel < world:
        raise Error(
            (
                "batch_norm_gather_stats_with_counts: expected counts to have"
                " at least one element per entry in mean's first dimension ("
            ),
            world,
            "), but got ",
            counts.numel,
            " elements",
        )
    _loss_float(mean)
    _same_device(a, mean)
    _same_device(a, invstd)
    _same_device(a, counts)
    _loss_float(a)
    var scalar = a.stype
    var rm_ptr = 0
    var rv_ptr = 0
    if _opt(args, 3):
        var rm = v_tensor(args[unsafe_offset=3])
        _same_device(a, rm)
        _loss_float(rm)
        if not rm.contig or rm.numel != features:
            raise Error("running_mean must be a contiguous [num_features]")
        scalar = rm.stype
        rm_ptr = rm.ptr
    if _opt(args, 4):
        var rv = v_tensor(args[unsafe_offset=4])
        _same_device(a, rv)
        if not rv.contig or rv.numel != features:
            raise Error("running_var must be a contiguous [num_features]")
        rv_ptr = rv.ptr
    var acc = ST_FLOAT64 if scalar == ST_FLOAT64 else ST_FLOAT32
    # batch_norm_gather_stats_cuda_template's typed accessors (it names the
    # running variance "running_mean" too).
    _expect_type(mean, acc, "mean")
    _expect_type(invstd, acc, "invstd")
    if _opt(args, 4):
        _expect_type(v_tensor(args[unsafe_offset=4]), scalar, "running_mean")
    _expect_type(counts, scalar, "counts")
    var m = Dense(mean)
    var s = _as_stype(Dense(invstd), mean.stype)
    var c = _as_stype(Dense(counts), scalar)
    var save_mean = _vec(features, acc, a.device)
    var save_invstd = _vec(features, acc, a.device)
    if features > 0:
        var ctx = ctx_for(a.device)
        var call = KernelCall("batch_norm_sync", "BnGather")
        call.arg_dtype(0, mean.dtype)
        call.arg_dtype(1, max_dtype(scalar))
        call.int(save_mean.t.ptr)
        call.int(save_invstd.t.ptr)
        call.int(m.t.ptr)
        call.int(s.t.ptr)
        call.int(rm_ptr)
        call.int(rv_ptr)
        call.int(c.t.ptr)
        var params = List[Int]()
        params.append(world)
        params.append(features)
        call.tuple(params)
        call.f64(eps)
        call.f64(momentum)
        call.int(ctx_ptr(ctx))
        call.run()
        _ = ctx
    _ = m^
    _ = s^
    _ = c^
    return OwnedPair(save_mean^, save_invstd^)


# aten::batch_norm_gather_stats_with_counts(Tensor input, Tensor mean,
#   Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum,
#   float eps, Tensor counts) -> (Tensor, Tensor)
def op_batch_norm_gather_stats_with_counts(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var r = _gather(args, v_tensor(args[unsafe_offset=7]))
    ret_owned(rets, 0, r.first)
    ret_owned(rets, 1, r.second)


# aten::batch_norm_gather_stats(Tensor input, Tensor mean, Tensor invstd,
#   Tensor? running_mean, Tensor? running_var, float momentum, float eps,
#   int count) -> (Tensor, Tensor)
def op_batch_norm_gather_stats(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """batch_norm_gather_stats_cuda: every replica counted `count`."""
    var a = v_tensor(args[unsafe_offset=0])
    var mean = v_tensor(args[unsafe_offset=1])
    if mean.rank != 2:
        raise Error(
            (
                "batch_norm_gather_stats: expected mean to be 2-dimensional"
                " (world_size, num_features), but got mean of sizes "
            ),
            sizes_str(mean),
        )
    var stype = a.stype
    if _opt(args, 3):
        stype = v_tensor(args[unsafe_offset=3]).stype
    var counts = _vec(mean.dim(0), stype, a.device)
    fill_value(counts.t, Float64(v_int(args[unsafe_offset=7])))
    var r = _gather(args, counts.t)
    _ = counts^
    ret_owned(rets, 0, r.first)
    ret_owned(rets, 1, r.second)


# aten::batch_norm_backward_reduce(Tensor grad_out, Tensor input,
#   Tensor mean, Tensor invstd, Tensor? weight, bool input_g, bool weight_g,
#   bool bias_g) -> (Tensor, Tensor, Tensor, Tensor)
def op_batch_norm_backward_reduce(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """batch_norm_backward_reduce_cuda: per channel `sum_dy`, `sum_dy_xmu`
    (in mean's dtype) and the affine gradients (in the weight's).

    CUDA has two routes with different results, chosen by layout: the
    channels-last one (both operands channels-last dense, or contiguous with
    a unit channel stride, e.g. any 2-D input) computes all four whatever is
    asked, the affine gradients `[0]`-sized without a weight; the other one
    computes only what is asked and leaves the rest undefined, and has
    typed accessors (the "mixed type" rule of `is_mixed_type`)."""
    var grad = v_tensor(args[unsafe_offset=0])
    var a = v_tensor(args[unsafe_offset=1])
    var input_g = v_bool(args[unsafe_offset=5])
    var weight_g = v_bool(args[unsafe_offset=6])
    var bias_g = v_bool(args[unsafe_offset=7])
    var has_w = _opt(args, 4)
    _loss_float(a)
    _same_device(a, grad)
    _expect_dtype(grad, a)
    var p = Planes(a)
    if not grad.same_shape(a):
        raise Error(
            (
                "batch_norm_backward_reduce: grad_out and input must share a"
                " shape, got "
            ),
            sizes_str(grad),
            " and ",
            sizes_str(a),
        )
    var mean_t = v_tensor(args[unsafe_offset=2])
    var invstd_t = v_tensor(args[unsafe_offset=3])
    var cl = (
        _uses_channels_last(grad)
        and _uses_channels_last(a)
        and (not has_w or v_tensor(args[unsafe_offset=4]).contig)
        and mean_t.contig
        and invstd_t.contig
    )
    if not cl:
        if invstd_t.stype != mean_t.stype:
            raise Error("mean and invstd need to have the same data types")
        var stat = a.stype
        if has_w and v_tensor(args[unsafe_offset=4]).stype != a.stype:
            stat = _acc_of(a.stype)
        if (weight_g or bias_g) and not has_w:
            # `at::empty({C}, weight.options())` of an undefined weight.
            raise Error("tensor does not have a device")
        if weight_g or bias_g:
            var wt = v_tensor(args[unsafe_offset=4])
            if weight_g:
                _expect_type(wt, stat, "grad_weight")
            if bias_g:
                _expect_type(wt, stat, "grad_bias")
        _expect_type(mean_t, _acc_of(stat), "mean")
        _expect_type(invstd_t, _acc_of(stat), "invstd")
        if input_g:
            _expect_type(mean_t, _acc_of(stat), "sum_dy")
    else:
        # batch_norm_backward_reduce_cuda_channels_last_template reads both
        # statistics as `accscalar_t` pointers.
        _expect_scalar(mean_t, _acc_of(a.stype))
        _expect_scalar(invstd_t, _acc_of(a.stype))
    var g = Planes(grad)
    var mean = _channel_vec(args, 2, a, p.c, "mean")
    var invstd = _as_stype(
        _channel_vec(args, 3, a, p.c, "invstd"), mean_t.stype
    )
    var wstype = mean_t.stype
    var wlen = 0
    if has_w:
        wstype = v_tensor(args[unsafe_offset=4]).stype
        wlen = p.c
    var flags = 7
    if not cl:
        flags = (
            (1 if input_g else 0)
            | (2 if weight_g else 0)
            | (4 if bias_g else 0)
        )
    elif not has_w:
        flags = 1
    if not cl:
        flags |= 8
    var layout = (L_INPUT if p.cl else 0) | (L_GRAD if g.cl else 0)
    flags |= layout << L_FLAGS_SHIFT
    var sum_dy = _vec(p.c, mean_t.stype, a.device)
    var sum_dy_xmu = _vec(p.c, mean_t.stype, a.device)
    var gw = _vec(wlen, wstype, a.device)
    var gb = _vec(wlen, wstype, a.device)
    if p.c > 0 and (flags & 7) != 0:
        var ctx = ctx_for(a.device)
        var call = KernelCall("batch_norm_sync", "BnBackwardReduce")
        call.arg_dtype(0, a.dtype)
        call.arg_dtype(1, mean.t.dtype)
        call.arg_dtype(2, max_dtype(wstype))
        call.int(sum_dy.t.ptr)
        call.int(sum_dy_xmu.t.ptr)
        call.int(gw.t.ptr)
        call.int(gb.t.ptr)
        call.int(p.dense.t.ptr)
        call.int(g.dense.t.ptr)
        call.int(mean.t.ptr)
        call.int(invstd.t.ptr)
        var params = List[Int]()
        params.append(flags)
        params.append(p.c)
        params.append(p.n)
        params.append(p.hxw)
        call.tuple(params)
        call.int(ctx_ptr(ctx))
        call.run()
        _ = ctx
    _ = p^
    _ = g^
    _ = mean^
    _ = invstd^
    if cl or input_g:
        ret_owned(rets, 0, sum_dy)
        ret_owned(rets, 1, sum_dy_xmu)
    else:
        rets[unsafe_offset=0] = none_arg()
        rets[unsafe_offset=1] = none_arg()
    if cl or weight_g:
        ret_owned(rets, 2, gw)
    else:
        rets[unsafe_offset=2] = none_arg()
    if cl or bias_g:
        ret_owned(rets, 3, gb)
    else:
        rets[unsafe_offset=3] = none_arg()


# aten::batch_norm_backward_elemt(Tensor grad_out, Tensor input,
#   Tensor mean, Tensor invstd, Tensor? weight, Tensor sum_dy,
#   Tensor sum_dy_xmu, Tensor count) -> Tensor
def op_batch_norm_backward_elemt(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """batch_norm_backward_elemt_cuda: `(dy - sum_dy / M - (x - mean) *
    invstd^2 * sum_dy_xmu / M) * weight * invstd` with M the sum of the
    int32 `count` vector (read on the device)."""
    var grad = v_tensor(args[unsafe_offset=0])
    var a = v_tensor(args[unsafe_offset=1])
    var count = v_tensor(args[unsafe_offset=7])
    _loss_float(a)
    _same_device(a, grad)
    _same_device(a, count)
    _expect_dtype(grad, a)
    var p = Planes(a)
    if not grad.same_shape(a):
        raise Error(
            (
                "batch_norm_backward_elemt: grad_out and input must share a"
                " shape, got "
            ),
            sizes_str(grad),
            " and ",
            sizes_str(a),
        )
    var mean_t = v_tensor(args[unsafe_offset=2])
    if v_tensor(args[unsafe_offset=3]).stype != mean_t.stype:
        raise Error("mean and invstd need to have the same data types")
    if not (_uses_channels_last(grad) and _uses_channels_last(a)):
        # The typed route: `stat_scalar_t` is float32 for a half / bfloat16
        # input with float32 statistics, else the input's dtype.
        var stat = a.stype
        if a.dtype != DType.float32 and a.dtype != DType.float64:
            if mean_t.stype == ST_FLOAT32:
                stat = ST_FLOAT32
        var sacc = _acc_of(stat)
        _expect_type(mean_t, sacc, "mean")
        _expect_type(v_tensor(args[unsafe_offset=3]), sacc, "invstd")
        if _opt(args, 4):
            _expect_type(v_tensor(args[unsafe_offset=4]), stat, "weight")
        _expect_type(v_tensor(args[unsafe_offset=5]), sacc, "sum_dy")
        _expect_type(v_tensor(args[unsafe_offset=6]), sacc, "sum_dy_xmu")
    var g = Planes(grad)
    # Each statistic and the weight converted on its own -- exactly -- to
    # the accumulation dtype the kernel computes in (the typed route's
    # checks above pin their dtypes; the channels-last one converts).
    var acc = _acc_of(a.stype)
    var mean = _as_stype(_channel_vec(args, 2, a, p.c, "mean"), acc)
    var invstd = _as_stype(_channel_vec(args, 3, a, p.c, "invstd"), acc)
    var sum_dy = _as_stype(_channel_vec(args, 5, a, p.c, "sum_dy"), acc)
    var sum_dy_xmu = _as_stype(_channel_vec(args, 6, a, p.c, "sum_dy_xmu"), acc)
    if count.stype != ST_INT32:
        raise Error(
            "expected scalar type Int but found ",
            _scalar_type_name(count.dtype),
        )
    var cnt = Dense(count)
    var w: Optional[Dense] = None
    var wdtype = mean.t.dtype
    var w_ptr = 0
    if _opt(args, 4):
        w = _as_stype(_channel_vec(args, 4, a, p.c, "weight"), acc)
        wdtype = w.value().t.dtype
        w_ptr = w.value().t.ptr
    # CUDA's channels-last route (both operands channels-last) allocates
    # `empty_like(input)`, the other one a contiguous result.
    var out_cl = p.cl and g.cl
    var gi = own(new_tensor(a.shape, a.rank, a.stype, a.device))
    if out_cl:
        gi = _out_like(p, a)
    if a.numel > 0:
        var ctx = ctx_for(a.device)
        var call = KernelCall("batch_norm_sync", "BnBackwardElemt")
        call.arg_dtype(0, a.dtype)
        call.arg_dtype(1, mean.t.dtype)
        call.arg_dtype(2, wdtype)
        call.int(gi.t.ptr)
        call.int(g.dense.t.ptr)
        call.int(p.dense.t.ptr)
        call.int(mean.t.ptr)
        call.int(invstd.t.ptr)
        call.int(w_ptr)
        call.int(sum_dy.t.ptr)
        call.int(sum_dy_xmu.t.ptr)
        call.int(cnt.t.ptr)
        var params = List[Int]()
        params.append(count.numel)
        params.append(p.c)
        params.append(p.hxw)
        params.append(a.numel)
        params.append(
            (L_INPUT if p.cl else 0)
            | (L_GRAD if g.cl else 0)
            | (L_OUT if out_cl else 0)
        )
        call.tuple(params)
        call.int(ctx_ptr(ctx))
        call.run()
        _ = ctx
    _ = p^
    _ = g^
    _ = mean^
    _ = invstd^
    _ = sum_dy^
    _ = sum_dy_xmu^
    _ = cnt^
    _ = w^
    ret_owned(rets, 0, gi)


# ---------------------------------------------------------------------------
# float64 native batch norm (the normalization_forward kernels are float32
# throughout): batch_norm_cuda_out's own steps on the batch_norm_sync kernels
# ---------------------------------------------------------------------------


def _elemt_launch(
    dst: T,
    p: Planes,
    a: T,
    w: T,
    b: T,
    has_w: Bool,
    has_b: Bool,
    mean: T,
    invstd: T,
) raises:
    """BnElemt with `dst` laid out as `p` is (see `_out_like`); the affine
    parameters arrive in the accumulation dtype."""
    var ctx = ctx_for(a.device)
    var call = KernelCall("batch_norm_sync", "BnElemt")
    call.arg_dtype(0, a.dtype)
    call.arg_dtype(1, mean.dtype)
    call.arg_dtype(2, mean.dtype)
    call.int(dst.ptr)
    call.int(p.dense.t.ptr)
    call.int(w.ptr if has_w else 0)
    call.int(b.ptr if has_b else 0)
    call.int(mean.ptr)
    call.int(invstd.ptr)
    var params = List[Int]()
    params.append(p.c)
    params.append(p.hxw)
    params.append(a.numel)
    params.append((L_INPUT | L_OUT) if p.cl else 0)
    call.tuple(params)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx


def _expect_scalar(t: T, stype: Int32) raises:
    """`const_data_ptr<scalar_t>()`'s check."""
    if t.stype != stype:
        raise Error(
            "expected scalar type ",
            _scalar_type_name(max_dtype(stype)),
            " but found ",
            _scalar_type_name(t.dtype),
        )


def _affine_checks(a: T, args: Values) raises:
    """batch_norm_elementwise's typed reads of weight and bias, per route:
    contiguous (`batch_norm_elemt_cuda_template`, `stat_scalar_t` the
    accumulation dtype when the first defined parameter's dtype differs from
    the input's), channels-last (`batch_norm_elemt_channels_last_cuda_
    template`: `acc_t` pointers when the weight's -- else the bias's --
    dtype differs from the input's, `scalar_t` ones otherwise), or the
    TensorIterator route, which converts whatever it is given."""
    var has_w = _opt(args, 1)
    var has_b = _opt(args, 2)
    if not has_w and not has_b:
        return
    var first = v_tensor(args[unsafe_offset=1 if has_w else 2])
    var mixed = first.stype != a.stype
    var want = _acc_of(a.stype) if mixed else a.stype
    if a.contig and not _uses_channels_last(a):
        if has_w:
            _expect_type(v_tensor(args[unsafe_offset=1]), want, "weight")
        if has_b:
            _expect_type(v_tensor(args[unsafe_offset=2]), want, "bias")
    elif _uses_channels_last(a):
        if has_w:
            _expect_scalar(v_tensor(args[unsafe_offset=1]), want)
        if has_b:
            _expect_scalar(v_tensor(args[unsafe_offset=2]), want)


def _invstd_of(rv: T, eps: Float64, acc: Int32) raises -> Owned:
    """batch_norm_calc_invstd: `rsqrt(var + eps)` in the running variance's
    accumulation dtype, stored in `acc`."""
    var racc = _acc_of(rv.stype)
    var v = own(T(retain(rv)))
    if rv.stype != racc:
        v = own(cast_to(rv, racc))
    var shifted_r = call_op(
        String("aten::add"),
        String("Scalar"),
        [
            tensor_arg(v.t),
            Value(TAG_SCALAR_DOUBLE, 0, f64_bits(eps), 0),
            Value(TAG_SCALAR_INT, 0, 1, 0),
        ],
        1,
    )
    var shifted = own(shifted_r.take_tensor(0))
    _ = v^
    var inv: Owned
    if racc == ST_FLOAT64:
        # No float64 rsqrt kernel here; `pow(-0.5)` is one.
        var r = call_op(
            String("aten::pow"),
            String("Tensor_Scalar"),
            [
                tensor_arg(shifted.t),
                Value(TAG_SCALAR_DOUBLE, 0, f64_bits(-0.5), 0),
            ],
            1,
        )
        inv = own(r.take_tensor(0))
    else:
        var r = call_op(
            String("aten::rsqrt"), String(""), [tensor_arg(shifted.t)], 1
        )
        inv = own(r.take_tensor(0))
    _ = shifted^
    if inv.t.stype == acc:
        return inv^
    var c = own(cast_to(inv.t, acc))
    _ = inv^
    return c^


def _bn_sync(
    args: Values,
    rets: Values,
    rm_i: Int,
    training: Bool,
    eps_i: Int,
    mom_i: Int,
) raises:
    """batch_norm_cuda on the batch_norm_sync kernels, for a float64 input
    (the normalization_forward kernels are float32 throughout; a
    channels-last one is read and written where it lies):
    Welford statistics with the running update and `rsqrt(var + eps)`
    (batch_norm_update_stats_and_invert / batch_norm_calc_invstd), or the
    running statistics copied and inverted, then batch_norm_elementwise."""
    var a = v_tensor(args[unsafe_offset=0])
    _loss_float(a)
    var p = Planes(a)
    var acc = _acc_of(a.stype)
    var eps = v_f64(args[unsafe_offset=eps_i])
    var momentum = v_f64(args[unsafe_offset=mom_i]) if mom_i >= 0 else 0.0
    var has_rm = _opt(args, rm_i)
    if has_rm != _opt(args, rm_i + 1):
        raise Error(
            "running_mean and running_var must either both be None or neither"
            " be None"
        )
    _affine_checks(a, args)
    var has_w = _opt(args, 1)
    var has_b = _opt(args, 2)
    var w = a.copy()
    var b = a.copy()
    var wd = Optional[Dense](None)
    var bd = Optional[Dense](None)
    if has_w:
        wd = _as_stype(_channel_vec(args, 1, a, p.c, "weight"), acc)
        w = wd.value().t.copy()
    if has_b:
        bd = _as_stype(_channel_vec(args, 2, a, p.c, "bias"), acc)
        b = bd.value().t.copy()
    var out = _out_like(p, a)
    var mean = _vec(p.c, acc, a.device)
    var invstd = _vec(p.c, acc, a.device)
    if training:
        var running: Optional[Tuple[T, T]] = None
        if has_rm:
            var rm = v_tensor(args[unsafe_offset=rm_i])
            var rv = v_tensor(args[unsafe_offset=rm_i + 1])
            _same_device(a, rm)
            _same_device(a, rv)
            _loss_float(rm)
            _expect_dtype(rv, rm)
            if (
                not rm.contig
                or not rv.contig
                or rm.numel != p.c
                or rv.numel != p.c
            ):
                raise Error("running statistics must be contiguous [C] tensors")
            running = (rm.copy(), rv.copy())
        if p.c > 0:
            _stats_launch(p, a, mean.t, invstd.t, running, 2, eps, momentum)
    else:
        if not has_rm:
            raise Error(
                "Expected has_running_mean to be true, but got false.  (Could"
                " this error message be improved?  If so, please report an"
                " enhancement request to PyTorch.)"
            )
        var rm = _channel_vec(args, rm_i, a, p.c, "running_mean")
        var rv = _channel_vec(args, rm_i + 1, a, p.c, "running_var")
        # `save_mean.copy_(running_mean)` into a fresh acc_type vector.
        if rm.t.stype == acc:
            copy_strided_into(mean.t, rm.t)
        else:
            mean = own(cast_to(rm.t, acc))
        invstd = _invstd_of(rv.t, eps, acc)
        _ = rm^
        _ = rv^
    if a.numel > 0:
        _elemt_launch(out.t, p, a, w, b, has_w, has_b, mean.t, invstd.t)
    _ = p^
    _ = wd^
    _ = bd^
    # `empty_like(self)`: any other dense permutation keeps its strides.
    var res = like_layout(out^, a, True)
    ret_owned(rets, 0, res)
    ret_owned(rets, 1, mean)
    ret_owned(rets, 2, invstd)


def _sync_route(a: T) raises -> Bool:
    """float64, which the normalization_forward kernels (float32
    throughout) do not take; every other dtype, channels-last included,
    stays on them."""
    return a.on_mojo() and a.dtype == DType.float64


def _relayout_ret0(rets: Values, a: T) raises:
    """Result 0 (a fresh contiguous output) with `a`'s strides when `a` is
    a dense permutation: CUDA allocates it with `empty_like(input)`. A
    contiguous input costs nothing."""
    if a.contig:
        return
    var cur = own(T(Int(rets[unsafe_offset=0].a)))
    rets[unsafe_offset=0] = none_arg()
    var o = like_layout(cur^, a, True)
    ret_owned(rets, 0, o)


# aten::native_batch_norm (and `_native_batch_norm_legit`, the same schema)
def op_native_batch_norm_any(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    if _sync_route(a):
        _bn_sync(args, rets, 3, v_bool(args[unsafe_offset=5]), 7, 6)
        return
    op_native_batch_norm(args, n_args, rets, n_rets)
    _relayout_ret0(rets, a)


# aten::_native_batch_norm_legit_no_training(Tensor input, Tensor? weight,
#   Tensor? bias, Tensor running_mean, Tensor running_var, float momentum,
#   float eps) -> (Tensor, Tensor, Tensor)
def op_batch_norm_legit_no_training_any(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    if _sync_route(a):
        _bn_sync(args, rets, 3, False, 6, -1)
        return
    op_batch_norm_legit_no_training(args, n_args, rets, n_rets)
    _relayout_ret0(rets, a)


def register_batch_norm(site: Site) raises:
    impl[op_native_batch_norm_out, "native_batch_norm.out"](site)
    impl[op_native_batch_norm_any, "native_batch_norm"](site)
    impl[op_native_batch_norm_any, "_native_batch_norm_legit"](site)
    impl[
        op_batch_norm_legit_no_training_any,
        "_native_batch_norm_legit_no_training",
    ](site)
    impl[op_native_batch_norm_out, "_native_batch_norm_legit.out"](site)
    impl[op_batch_norm_legit_no_stats, "_native_batch_norm_legit.no_stats"](
        site
    )
    impl[
        op_batch_norm_legit_no_stats_out,
        "_native_batch_norm_legit.no_stats_out",
    ](site)
    impl[op_batch_norm_with_update, "_batch_norm_with_update"](site)
    impl[op_batch_norm_with_update_out, "_batch_norm_with_update.out"](site)
    impl[op_batch_norm_backward, "batch_norm_backward"](site)
    impl[op_batch_norm_stats, "batch_norm_stats"](site)
    impl[op_batch_norm_update_stats, "batch_norm_update_stats"](site)
    impl[op_batch_norm_elemt, "batch_norm_elemt"](site)
    impl[op_batch_norm_elemt_out, "batch_norm_elemt.out"](site)
    impl[op_batch_norm_gather_stats, "batch_norm_gather_stats"](site)
    impl[
        op_batch_norm_gather_stats_with_counts,
        "batch_norm_gather_stats_with_counts",
    ](site)
    impl[op_batch_norm_backward_reduce, "batch_norm_backward_reduce"](site)
    impl[op_batch_norm_backward_elemt, "batch_norm_backward_elemt"](site)
