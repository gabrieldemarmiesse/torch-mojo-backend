"""aten ops: nn group (see docs/native_backend.md).

Softmax family, normalization (layer / batch / group), NLL loss, embedding,
2-D pooling and bilinear upsampling. Ported from the old Python fast path
(`eager_kernels/aten_fast.py`): same dtype gating, same route cascade (which
kernel for which dtype / layout / device), same output allocation and the same
kernel slot lists.

Families used: nn_ops (classic pointer ABI + SoftmaxSpec),
normalization_forward_ops, normalization_backward_ops, softmax_backward_ops,
loss_ops, embedding_backward_ops, reduction_ops (LogSoftmaxSpec).
"""
from std.utils import IndexList

from abi import (
    IntList,
    Owned,
    ST_FLOAT32,
    ST_INT64,
    T,
    TAG_BOOL,
    TAG_BOOL_LIST,
    TAG_INT_LIST,
    TAG_NONE,
    TAG_SCALAR_DOUBLE,
    TAG_SCALAR_INT,
    TAG_TENSOR,
    Value,
    Values,
    bits_f64,
    contiguous_strides,
    dtype_code,
    f64_bits,
    new_like,
    new_tensor,
    own,
    release,
    retain,
    ret_owned,
    ret_ref,
    ret_tensor,
    view_strided,
    unsupported,
    v_bool,
    v_dtype_or,
    v_f64,
    v_int,
    v_is_none,
    v_tensor,
)
from device import ctx_for, ctx_ptr, dev
from kernels import KernelCall
from op_utils import MAX_RANK, _f64_slot
from ops_common import (
    call_op,
    cast_to,
    contiguous,
    copy_strided_into,
    fill_value,
    resize_out,
)
from registry import Site, impl, op_address_of

# The three dtypes every nn kernel family is instantiated for
# (`op_utils.FLOAT_DTYPES` / `aten_fast._FLOAT_DTYPES`).
comptime NAN_BITS = Int64(0x7FF8000000000000)


def _is_float(dt: DType) -> Bool:
    return dt == DType.float32 or dt == DType.bfloat16 or dt == DType.float16


def _on_gpu(t: T) raises -> Bool:
    """True on an accelerator; the MAX CPU device is the last mojo index."""
    return t.on_mojo() and not dev(t.device)[].is_cpu


def _require_mojo(t: T, what: StaticString) raises:
    if not t.on_mojo():
        unsupported(String(what) + ": operand is not on the mojo device")


def _swapped(
    src: IndexList[MAX_RANK], rank: Int, a: Int, b: Int
) -> IndexList[MAX_RANK]:
    """`src` with the two logical axes `a` and `b` exchanged."""
    var out = IndexList[MAX_RANK](0)
    for k in range(MAX_RANK):
        out[k] = src[k]
    var pa = MAX_RANK - rank + a
    var pb = MAX_RANK - rank + b
    out[pa] = src[pb]
    out[pb] = src[pa]
    return out


def _bool_list(v: Value) raises -> List[Bool]:
    """A borrowed `bool[]` argument (uint8 per element in the call arena)."""
    var out = List[Bool]()
    if v.tag == TAG_NONE:
        return out^
    if v.tag != TAG_BOOL_LIST:
        raise Error("expected a bool[] argument, got record tag ", v.tag)
    var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(v.a))
    for i in range(Int(v.len)):
        out.append(p[unsafe_offset=i] != 0)
    return out^


def _pair(l: IntList, what: StaticString) raises -> Tuple[Int, Int]:
    if len(l) == 1:
        return (l[0], l[0])
    if len(l) == 2:
        return (l[0], l[1])
    raise Error(what, " must have one or two entries, got ", len(l))


struct Held(Movable):
    """A tensor that is either a borrowed input or a copy this op made.

    `contiguous`/`cast_to` hand back the input itself when nothing had to
    change, so the release has to be conditional; owning that decision here
    keeps every early `raise` (a declined route, a failed kernel build) from
    leaking the copy.
    """

    var t: T
    var mine: Bool

    def __init__(out self, var t: T, mine: Bool):
        self.t = t^
        self.mine = mine

    def take(mut self) -> T:
        self.mine = False
        return self.t.copy()

    def __deinit__(deinit self):
        if self.mine:
            release(self.t.h)


def _mat(t: T) raises -> Held:
    """`t` materialized contiguous (borrowed when it already is)."""
    var c = contiguous(t)
    var mine = c.h != t.h
    return Held(c^, mine)


def _mat16(t: T) raises -> Held:
    """`t` contiguous AND 16-byte aligned (the vectorized kernels' contract).

    An offset view can be contiguous and still land mid-vector, so alignment
    is checked on the runtime pointer rather than inferred from the shape.
    """
    if t.contig and t.ptr % 16 == 0:
        return Held(t.copy(), False)
    var out = new_like(t)
    copy_strided_into(out, t)
    return Held(out^, True)


def _cast(t: T, stype: Int32) raises -> Held:
    var c = cast_to(t, stype)
    var mine = c.h != t.h
    return Held(c^, mine)


def _keep_ptr(mut keep: List[Held], t: T) raises -> Int:
    """Materialize `t` contiguous, park it in `keep` for the call, return its
    data pointer (the classic kernel ABI takes raw pointers)."""
    keep.append(_mat(t))
    return keep[len(keep) - 1].t.ptr


def _same_dims(t: T, sizes: IntList) -> Bool:
    if t.rank != len(sizes):
        return False
    for i in range(t.rank):
        if t.dim(i) != sizes[i]:
            return False
    return True


def _stat_shape(t: T, k: Int) -> IndexList[MAX_RANK]:
    """`t`'s shape with the trailing `k` dims collapsed to 1 (layer norm's
    mean / rstd shape)."""
    var out = IndexList[MAX_RANK](1)
    for i in range(MAX_RANK):
        out[i] = t.shape[i]
    for i in range(k):
        out[MAX_RANK - 1 - i] = 1
    return out


# ---------------------------------------------------------------------------
# Softmax / log-softmax
# ---------------------------------------------------------------------------


def _one_device(a: T, b: T) raises:
    """Both operands of a raw-pointer launch on the same mojo device.

    A kernel gets bare pointers and one stream: a pointer belonging to
    another device -- or to no mojo device at all -- would be dereferenced
    against the wrong context. The fields are cached on `T`, so this costs
    nothing. Private to this file until the port is merged; it belongs in
    ops_common.mojo.
    """
    if not a.on_mojo() or not b.on_mojo() or a.device != b.device:
        raise Error("expected every operand on the same mojo device")


def _spec_unary(family: String, op: String, src: T, dst: T) raises:
    _one_device(src, dst)
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall(family, op)
    call.arg_dtype(0, src.dtype)
    call.out_dtype(dst.dtype)
    call.spec(src.spec(cp))
    call.spec(dst.spec(cp))
    call.run()
    _ = ctx


def _softmax_family(args: Values, rets: Values, log_variant: Bool) raises:
    """`aten::_softmax` / `aten::_log_softmax`.

    The kernels reduce the trailing dim of a contiguous operand, so any other
    dim goes through the transpose identity
    `softmax(x, d) = softmax(x.swap(d, -1), -1).swap(d, -1)`. Both swaps are
    zero-copy views over the two ends: the operand is materialized once and
    the result is written back through a permuted view of a fresh contiguous
    output (ATen promises a contiguous result, and `_copy_from` mojo->host
    cannot take a strided host destination).
    """
    var self = v_tensor(args[unsafe_offset=0])
    var dim = v_int(args[unsafe_offset=1])
    var half_to_float = v_bool(args[unsafe_offset=2])
    _require_mojo(self, "softmax")
    if not _is_float(self.dtype):
        unsupported("softmax of dtype " + String(self.dtype))
    if self.numel == 0:
        unsupported("softmax of an empty tensor")
    var work_stype = self.stype
    if half_to_float:
        # torch computes a half input in fp32 and returns fp32. The old path
        # took bfloat16 for softmax but only float16 for log_softmax; keep
        # both gates (on this device ATen casts on the host instead, so
        # half_to_float is in practice never set).
        if log_variant:
            if self.dtype != DType.float16:
                unsupported(
                    "_log_softmax half_to_float from " + String(self.dtype)
                )
        elif self.dtype != DType.float16 and self.dtype != DType.bfloat16:
            unsupported("_softmax half_to_float from " + String(self.dtype))
        work_stype = ST_FLOAT32
    var work = _cast(self, work_stype)
    var rank = work.t.rank
    if rank == 0:
        if log_variant and dim != -1 and dim != 0:
            unsupported("log_softmax dim out of range for a 0-d tensor")
        var flat = own(new_like(work.t))
        fill_value(flat.t, 0.0 if log_variant else 1.0)
        ret_owned(rets, 0, flat)
        return
    if dim < -rank or dim >= rank:
        unsupported("softmax dim out of range")
    if dim < 0:
        dim += rank
    var family = String("nn_ops")
    var op = String("SoftmaxSpec")
    if log_variant:
        family = String("reduction_ops")
        op = String("LogSoftmaxSpec")
    if dim == rank - 1:
        var src = _mat(work.t)
        var out = own(
            new_tensor(work.t.shape, rank, work.t.stype, work.t.device)
        )
        _spec_unary(family, op, src.t, out.t)
        _ = work.t.ptr  # src may borrow work's storage; keep it past the call
        ret_owned(rets, 0, out)
        return
    var vshape = _swapped(work.t.shape, rank, dim, rank - 1)
    var vstrides = _swapped(work.t.strides, rank, dim, rank - 1)
    var view = own(view_strided(work.t, vshape, vstrides, rank, work.t.offset))
    var src = _mat(view.t)
    var tmp = own(new_tensor(vshape, rank, work.t.stype, work.t.device))
    _spec_unary(family, op, src.t, tmp.t)
    # ATen hands back a contiguous result, so the second swap is a strided
    # write into a fresh contiguous output rather than a returned view.
    var out = own(new_tensor(work.t.shape, rank, work.t.stype, work.t.device))
    var oview = own(
        view_strided(
            out.t,
            vshape,
            _swapped(
                contiguous_strides(work.t.shape, rank), rank, dim, rank - 1
            ),
            rank,
            0,
        )
    )
    copy_strided_into(oview.t, tmp.t)
    _ = work.t.ptr  # src may borrow work's storage; keep it past the call
    ret_owned(rets, 0, out)


# aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
def op_softmax(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _softmax_family(args, rets, False)


# aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
def op_log_softmax(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _softmax_family(args, rets, True)


def _lsm_backward(dst: T, grad: T, output: T, rows: Int, cols: Int) raises:
    _one_device(grad, dst)
    _one_device(output, dst)
    var ctx = ctx_for(dst.device)
    var call = KernelCall("softmax_backward_ops", "LogSoftmaxBackwardData")
    call.arg_dtype(0, grad.dtype)
    call.arg_dtype(1, output.dtype)
    call.out_dtype(dst.dtype)
    call.int(dst.ptr)
    call.int(grad.ptr)
    call.int(output.ptr)
    call.int(rows)
    call.int(cols)
    call.int(dtype_code(grad.dtype))
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx


# aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim,
#   ScalarType input_dtype) -> Tensor
def op_log_softmax_backward_data(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var output = v_tensor(args[unsafe_offset=1])
    var dim = v_int(args[unsafe_offset=2])
    var target = v_dtype_or(args[unsafe_offset=3], grad.stype)
    _require_mojo(grad, "_log_softmax_backward_data")
    if not _on_gpu(grad):
        unsupported(
            "_log_softmax_backward_data needs an accelerator (the fused kernel"
            " has no CPU route)"
        )
    if (
        not grad.same_shape(output)
        or grad.stype != output.stype
        or grad.device != output.device
    ):
        unsupported(
            "_log_softmax_backward_data: grad and output must match in shape,"
            " dtype and device"
        )
    if not _is_float(grad.dtype):
        unsupported("_log_softmax_backward_data of dtype " + String(grad.dtype))
    if target != grad.stype:
        # f32 grad -> f16 input_dtype went through the composed
        # sum/exp/addcmul chain in Python; the fused kernel is single-dtype.
        unsupported(
            "_log_softmax_backward_data with an input_dtype different from"
            " the gradient's"
        )
    var rank = grad.rank
    if rank == 0:
        unsupported("_log_softmax_backward_data of a 0-d tensor")
    if dim < -rank or dim >= rank:
        unsupported("_log_softmax_backward_data dim out of range")
    if dim < 0:
        dim += rank
    if grad.numel == 0:
        # ATen promises a fresh contiguous result and no kernel is needed.
        ret_tensor(rets, 0, new_tensor(grad.shape, rank, target, grad.device))
        return
    if dim == rank - 1:
        var gm = _mat16(grad)
        var om = _mat16(output)
        var out = own(new_tensor(grad.shape, rank, target, grad.device))
        _check_rows(out.t, grad.dim(rank - 1), grad.numel)
        _lsm_backward(
            out.t,
            gm.t,
            om.t,
            grad.numel // grad.dim(rank - 1),
            grad.dim(rank - 1),
        )
        ret_owned(rets, 0, out)
        return
    var vshape = _swapped(grad.shape, rank, dim, rank - 1)
    var gview = own(
        view_strided(
            grad,
            vshape,
            _swapped(grad.strides, rank, dim, rank - 1),
            rank,
            grad.offset,
        )
    )
    var oview = own(
        view_strided(
            output,
            vshape,
            _swapped(output.strides, rank, dim, rank - 1),
            rank,
            output.offset,
        )
    )
    var gm = _mat16(gview.t)
    var om = _mat16(oview.t)
    var tmp = own(new_tensor(vshape, rank, target, grad.device))
    var cols = vshape[MAX_RANK - 1]
    _check_rows(tmp.t, cols, grad.numel)
    _lsm_backward(tmp.t, gm.t, om.t, grad.numel // cols, cols)
    var out = own(new_tensor(grad.shape, rank, target, grad.device))
    var back = own(
        view_strided(
            out.t,
            vshape,
            _swapped(contiguous_strides(grad.shape, rank), rank, dim, rank - 1),
            rank,
            0,
        )
    )
    copy_strided_into(back.t, tmp.t)
    ret_owned(rets, 0, out)


def _check_rows(dst: T, cols: Int, numel: Int) raises:
    if dst.ptr % 16 != 0:
        unsupported(
            "_log_softmax_backward_data: the output allocation is not 16-byte"
            " aligned"
        )
    if numel // cols >= 2147483648:
        unsupported("_log_softmax_backward_data: more than 2^31 rows")


# ---------------------------------------------------------------------------
# Layer norm
# ---------------------------------------------------------------------------


# aten::native_layer_norm(Tensor input, SymInt[] normalized_shape,
#   Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)
def op_native_layer_norm(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var ns = IntList(args[unsafe_offset=1])
    var has_w = not v_is_none(args[unsafe_offset=2])
    var has_b = not v_is_none(args[unsafe_offset=3])
    var eps = v_f64(args[unsafe_offset=4])
    _require_mojo(a, "native_layer_norm")
    if a.numel == 0 or not _is_float(a.dtype):
        unsupported("native_layer_norm of dtype " + String(a.dtype))
    var k = len(ns)
    if k < 1 or a.rank < k:
        unsupported("native_layer_norm: bad normalized_shape rank")
    var cols = 1
    for i in range(k):
        if a.dim(a.rank - k + i) != ns[i]:
            unsupported("native_layer_norm: normalized_shape mismatch")
        cols *= ns[i]
    if cols <= 0:
        unsupported("native_layer_norm: empty normalized_shape")
    var rows = a.numel // cols
    # `native_layer_norm_backward` here covers float32 only, so a
    # grad-requiring reduced-precision input is refused in the FORWARD, where
    # the traceback still names the op and the user's own frame, rather than
    # succeeding and failing later inside the autograd engine with nothing in
    # the message pointing at the layer norm.
    #
    # Only the INPUT is asked, although ATen records the node when the weight
    # or the bias requires grad too: grad mode is unreachable from inside a
    # backend kernel (see `_needs_grad` in ops_attention.mojo), and an
    # `nn.LayerNorm`'s Parameters require grad even under `torch.no_grad()` --
    # asking them would refuse every reduced-precision INFERENCE forward. An
    # activation, by contrast, requires grad exactly when a graph is being
    # built. The residual hole is a reduced-precision layer norm applied to a
    # non-grad input with grad-requiring parameters, which still fails in the
    # backward as it did before.
    if a.stype != ST_FLOAT32 and a.requires_grad():
        unsupported(
            "aten::native_layer_norm on a "
            + String(a.dtype)
            + " input that requires grad: this device implements"
            " aten::native_layer_norm_backward for float32 only, so the"
            " backward would fail. Run the forward under torch.no_grad(), or"
            " keep the layer norm in float32 (autocast already does: its"
            " policy runs normalization in float32)."
        )
    var keep = List[Held]()
    var gamma_ptr = 0
    var beta_ptr = 0
    var gamma_dtype = a.dtype
    var beta_dtype = a.dtype
    if has_w:
        var w = v_tensor(args[unsafe_offset=2])
        if w.stype != a.stype or w.device != a.device or not _same_dims(w, ns):
            unsupported("native_layer_norm: unsupported weight")
        gamma_dtype = w.dtype
        gamma_ptr = _keep_ptr(keep, w)
    if has_b:
        var b = v_tensor(args[unsafe_offset=3])
        if b.stype != a.stype or b.device != a.device or not _same_dims(b, ns):
            unsupported("native_layer_norm: unsupported bias")
        beta_dtype = b.dtype
        beta_ptr = _keep_ptr(keep, b)
    var am = _mat(a)
    var out = own(new_like(a))
    # The device kernels write float32 into the two statistics whatever the
    # input dtype is; ATen's `param_scalar_type` is the input's own dtype
    # here (weight/bias of a different dtype are declined above), so both are
    # cast back down once at the end.
    var stats = _stat_shape(a, k)
    var mean = own(new_tensor(stats, a.rank, ST_FLOAT32, a.device))
    var rstd = own(new_tensor(stats, a.rank, ST_FLOAT32, a.device))
    var ctx = ctx_for(a.device)
    if _on_gpu(a):
        var call = KernelCall("normalization_forward_ops", "LayerNormForward")
        call.arg_dtype(0, a.dtype)
        call.arg_dtype(1, gamma_dtype)
        call.arg_dtype(2, beta_dtype)
        call.out_dtype_i(0, out.t.dtype)
        call.out_dtype_i(1, DType.float32)
        call.out_dtype_i(2, DType.float32)
        call.flag("HAS_WEIGHT", 1 if has_w else 0)
        call.flag("HAS_BIAS", 1 if has_b else 0)
        call.int(out.t.ptr)
        call.int(mean.t.ptr)
        call.int(rstd.t.ptr)
        call.int(am.t.ptr)
        call.int(gamma_ptr)
        call.int(beta_ptr)
        call.int(rows)
        call.int(cols)
        call.f64(eps)
        call.int(1 if has_w else 0)
        call.int(1 if has_b else 0)
        # hxw / cpg / group: read only by the group-norm affine.
        call.int(1)
        call.int(1)
        call.int(1)
        call.int(ctx_ptr(ctx))
        call.run()
        _ = am.t.ptr
        _ = len(keep)
    else:
        # The classic nn_ops route has no optional-affine ABI: synthesize the
        # neutral parameters the same way the Python path did.
        var ones = own(_filled(a, cols, 1.0))
        var zeros = own(_filled(a, cols, 0.0))
        if not has_w:
            gamma_ptr = ones.t.ptr
        if not has_b:
            beta_ptr = zeros.t.ptr
        # `fill_value` writes the two synthesized parameters with
        # `DeviceContext.enqueue_memset`, which the MAX *CPU* context queues,
        # while the nn_ops kernel below runs inline on this thread: without
        # this drain the kernel reads uninitialized gamma/beta about one call
        # in ten (measured). Accelerators order both on the same stream and
        # never reach this branch.
        ctx.synchronize()
        var call = KernelCall("nn_ops", "LayerNorm")
        call.arg_dtype(0, a.dtype)
        call.arg_dtype(1, gamma_dtype)
        call.arg_dtype(2, beta_dtype)
        call.out_dtype_i(0, out.t.dtype)
        call.out_dtype_i(1, DType.float32)
        call.out_dtype_i(2, DType.float32)
        call.flag("HAS_WEIGHT", 1 if has_w else 0)
        call.flag("HAS_BIAS", 1 if has_b else 0)
        call.int(out.t.ptr)
        call.int(mean.t.ptr)
        call.int(rstd.t.ptr)
        call.int(am.t.ptr)
        call.int(gamma_ptr)
        call.int(beta_ptr)
        var params = List[Int]()
        params.append(_f64_slot(eps))
        params.append(rows)
        params.append(cols)
        call.tuple(params)
        call.int(dtype_code(a.dtype))
        call.int(ctx_ptr(ctx))
        call.run()
        _ = am.t.ptr
        _ = len(keep)
        _ = ones.t.ptr
        _ = zeros.t.ptr
    _ = ctx
    ret_owned(rets, 0, out)
    # float32 statistics whatever the input dtype is, matching the CUDA
    # kernel (`at::toAccumulateType(input.scalar_type(), true)`). The CPU
    # kernel returns them in the input dtype instead; the accelerator
    # contract is the one to keep, because the backward that consumes them
    # is an accelerator kernel and reduced-precision statistics would lose
    # the precision the forward accumulated.
    ret_owned(rets, 1, mean)
    ret_owned(rets, 2, rstd)


def _filled(like: T, n: Int, value: Float64) raises -> T:
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 1] = n
    var t = new_tensor(shape, 1, like.stype, like.device)
    fill_value(t, value)
    return t^


# aten::native_layer_norm_backward(Tensor grad_out, Tensor input,
#   SymInt[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight,
#   Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
def op_native_layer_norm_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var a = v_tensor(args[unsafe_offset=1])
    var ns = IntList(args[unsafe_offset=2])
    var saved_mean = v_tensor(args[unsafe_offset=3])
    var saved_rstd = v_tensor(args[unsafe_offset=4])
    var has_w = not v_is_none(args[unsafe_offset=5])
    var has_b = not v_is_none(args[unsafe_offset=6])
    var mask = _bool_list(args[unsafe_offset=7])
    if len(mask) != 3:
        raise Error(
            "native_layer_norm_backward: output_mask must have 3 entries"
        )
    _require_mojo(a, "native_layer_norm_backward")
    if not _on_gpu(a):
        unsupported(
            "native_layer_norm_backward needs an accelerator (the kernel has"
            " no CPU route)"
        )
    if (
        a.dtype != DType.float32
        or grad.dtype != DType.float32
        or saved_mean.dtype != DType.float32
        or saved_rstd.dtype != DType.float32
    ):
        unsupported("native_layer_norm_backward covers float32 only")
    if (
        grad.device != a.device
        or saved_mean.device != a.device
        or saved_rstd.device != a.device
        or not grad.same_shape(a)
    ):
        unsupported("native_layer_norm_backward: mismatched operands")
    var k = len(ns)
    if k < 1 or a.rank < k:
        unsupported("native_layer_norm_backward: bad normalized_shape rank")
    var cols = 1
    for i in range(k):
        if a.dim(a.rank - k + i) != ns[i]:
            unsupported("native_layer_norm_backward: normalized_shape mismatch")
        cols *= ns[i]
    if cols <= 0:
        unsupported("native_layer_norm_backward: empty normalized_shape")
    var rows = a.numel // cols
    if saved_mean.numel != rows or saved_rstd.numel != rows:
        unsupported("native_layer_norm_backward: wrong saved-statistic size")
    if has_w:
        var w = v_tensor(args[unsafe_offset=5])
        if (
            w.dtype != DType.float32
            or w.device != a.device
            or not _same_dims(w, ns)
        ):
            unsupported("native_layer_norm_backward: unsupported weight")
    if has_b:
        var b = v_tensor(args[unsafe_offset=6])
        if (
            b.dtype != DType.float32
            or b.device != a.device
            or not _same_dims(b, ns)
        ):
            unsupported("native_layer_norm_backward: unsupported bias")
    if (mask[1] and not has_w) or (mask[2] and not has_b):
        unsupported(
            "native_layer_norm_backward: an affine gradient was requested"
            " without the parameter"
        )
    var pshape = IndexList[MAX_RANK](1)
    for i in range(k):
        pshape[MAX_RANK - k + i] = ns[i]
    if rows == 0:
        # ATen defines the two affine reductions over an empty outer extent as
        # zero; avoid a zero-grid launch.
        _ret_masked(rets, 0, a.shape, a.rank, a, mask[0], False)
        _ret_masked(rets, 1, pshape, k, a, mask[1], True)
        _ret_masked(rets, 2, pshape, k, a, mask[2], True)
        return
    var keep = List[Held]()
    var grad_p = _keep_ptr(keep, grad)
    var a_p = _keep_ptr(keep, a)
    var mean_p = _keep_ptr(keep, saved_mean)
    var rstd_p = _keep_ptr(keep, saved_rstd)
    # The affine reductions do not consume weight: materialize it only for the
    # grad-input kernel, which needs gamma in the dx formula.
    var gamma_p = 0
    if has_w and mask[0]:
        gamma_p = _keep_ptr(keep, v_tensor(args[unsafe_offset=5]))
    var gi = own(_masked_alloc(a.shape, a.rank, a, mask[0]))
    var gw = own(_masked_alloc(pshape, k, a, mask[1]))
    var gb = own(_masked_alloc(pshape, k, a, mask[2]))
    var bits = (
        (1 if mask[0] else 0) | (2 if mask[1] else 0) | (4 if mask[2] else 0)
    )
    var ctx = ctx_for(a.device)
    var call = KernelCall("normalization_backward_ops", "LayerNormBackwardF32")
    call.arg_dtype(0, DType.float32)
    call.arg_dtype(1, DType.float32)
    call.arg_dtype(2, DType.float32)
    call.arg_dtype(3, DType.float32)
    call.arg_dtype(4, DType.float32)
    call.out_dtype_i(0, DType.float32)
    call.out_dtype_i(1, DType.float32)
    call.out_dtype_i(2, DType.float32)
    call.flag("OUTPUT_MASK", bits)
    call.int(gi.t.ptr if mask[0] else 0)
    call.int(gw.t.ptr if mask[1] else 0)
    call.int(gb.t.ptr if mask[2] else 0)
    call.int(grad_p)
    call.int(a_p)
    call.int(mean_p)
    call.int(rstd_p)
    call.int(gamma_p)
    call.int(rows)
    call.int(cols)
    call.int(bits)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = len(keep)
    _ = ctx
    # an output autograd did not ask for is an undefined Tensor (None record)
    if mask[0]:
        ret_owned(rets, 0, gi)
    else:
        rets[unsafe_offset=0] = Value(TAG_NONE, 0, 0, 0)
    if mask[1]:
        ret_owned(rets, 1, gw)
    else:
        rets[unsafe_offset=1] = Value(TAG_NONE, 0, 0, 0)
    if mask[2]:
        ret_owned(rets, 2, gb)
    else:
        rets[unsafe_offset=2] = Value(TAG_NONE, 0, 0, 0)


def _masked_alloc(
    shape: IndexList[MAX_RANK], rank: Int, like: T, wanted: Bool
) raises -> T:
    """The requested gradient, or the empty stand-in this ABI returns for one
    autograd did not ask for (there is no way to build an undefined
    `at::Tensor` from Mojo; the engine never reads a masked-off result)."""
    if wanted:
        return new_tensor(shape, rank, ST_FLOAT32, like.device)
    return new_tensor(IndexList[MAX_RANK](0), 1, ST_FLOAT32, like.device)


def _ret_masked(
    rets: Values,
    i: Int,
    shape: IndexList[MAX_RANK],
    rank: Int,
    like: T,
    wanted: Bool,
    zero: Bool,
) raises:
    var t = own(_masked_alloc(shape, rank, like, wanted))
    if wanted and zero:
        fill_value(t.t, 0.0)
    ret_owned(rets, i, t)


# ---------------------------------------------------------------------------
# Batch norm
# ---------------------------------------------------------------------------


def _bn_channels(a: T) raises -> Int:
    if a.rank < 2 or a.numel == 0:
        unsupported("batch norm needs a non-empty tensor of rank >= 2")
    return a.dim(1)


def _bn_param(t: T, a: T, channels: Int, what: StaticString) raises:
    """A per-channel parameter the kernels read (and, for the running
    statistics, write) in place: contiguous, 1-D of `channels`, float."""
    if (
        t.device != a.device
        or not t.contig
        or t.rank != 1
        or t.dim(0) != channels
        or not _is_float(t.dtype)
    ):
        unsupported(
            String(what) + " must be a contiguous float[C] on the same device"
        )


def _nn_arg(t: T) -> Value:
    return Value(TAG_TENSOR, 0, Int64(t.h), 0)


def _nn_call1(
    name: StaticString, overload: StaticString, var args: List[Value]
) raises -> T:
    """One aten op through the dispatcher, one owned Tensor result.

    Mojo destroys a value right after its last use and reading `.t` off an
    `Owned` ends the borrow, so an intermediate handed in here must be kept
    alive past the call by the caller (`_ = x.t.h` after it) -- same rule as
    ops_composed.mojo, where the composed formulas live.
    """
    var rets = call_op(String(name), String(overload), args^, 1)
    return rets.take_tensor(0)


def _nn_hold(t: T) raises -> Owned:
    """A second owned handle to `t`, so a value that is sometimes a borrowed
    input and sometimes a fresh allocation is handled uniformly."""
    return own(T(retain(t)))


def _nn_cast(t: Owned, stype: Int32) raises -> Owned:
    """`t` in `stype`, always as a handle the caller owns: `cast_to` returns
    the input itself when the dtype already matches."""
    if t.t.stype == stype:
        return _nn_hold(t.t)
    return own(cast_to(t.t, stype))


def _nn_mul(a: Owned, b: Owned) raises -> Owned:
    return own(_nn_call1("aten::mul", "Tensor", [_nn_arg(a.t), _nn_arg(b.t)]))


def _nn_add(a: Owned, b: Owned) raises -> Owned:
    return own(
        _nn_call1(
            "aten::add",
            "Tensor",
            [_nn_arg(a.t), _nn_arg(b.t), Value(TAG_SCALAR_INT, 0, 1, 0)],
        )
    )


def _nn_sub(a: Owned, b: Owned) raises -> Owned:
    return own(
        _nn_call1(
            "aten::sub",
            "Tensor",
            [_nn_arg(a.t), _nn_arg(b.t), Value(TAG_SCALAR_INT, 0, 1, 0)],
        )
    )


def _nn_add_scalar(a: Owned, v: Float64) raises -> Owned:
    return own(
        _nn_call1(
            "aten::add",
            "Scalar",
            [
                _nn_arg(a.t),
                Value(TAG_SCALAR_DOUBLE, 0, f64_bits(v), 0),
                Value(TAG_SCALAR_INT, 0, 1, 0),
            ],
        )
    )


def _nn_div_scalar(a: Owned, v: Float64) raises -> Owned:
    return own(
        _nn_call1(
            "aten::div",
            "Scalar",
            [_nn_arg(a.t), Value(TAG_SCALAR_DOUBLE, 0, f64_bits(v), 0)],
        )
    )


def _nn_rsqrt(a: Owned) raises -> Owned:
    return own(_nn_call1("aten::rsqrt", "", [_nn_arg(a.t)]))


def _nn_sum_dims(x: Owned, dims: List[Int64]) raises -> Owned:
    """`x.sum(dims)` without keepdim: for batch norm's reduce set, a
    per-channel vector."""
    return own(
        _nn_call1(
            "aten::sum",
            "dim_IntList",
            [
                _nn_arg(x.t),
                Value(
                    TAG_INT_LIST,
                    Int32(len(dims)),
                    Int64(Int(dims.unsafe_ptr())),
                    0,
                ),
                Value(TAG_BOOL, 0, 0, 0),
                Value(TAG_NONE, 0, 0, 0),
            ],
        )
    )


def _nn_channel_view(vec: Owned, rank: Int) raises -> Owned:
    """A rank-`rank` `[1, C, 1, ...]` view of a per-channel vector, which is
    what lines it up with dim 1 of the input for the broadcast binary
    kernels. Zero-copy: only the C axis carries a real stride."""
    var shape = IndexList[MAX_RANK](1)
    var strides = IndexList[MAX_RANK](1)
    shape[MAX_RANK - rank + 1] = vec.t.dim(0)
    strides[MAX_RANK - rank + 1] = vec.t.stride(0)
    return own(view_strided(vec.t, shape, strides, rank, vec.t.offset))


def _bn_set_saved_stats(rets: Values, mean: T, var_t: T, eps: Float64) raises:
    """Results 1 and 2 of an inference batch norm: a copy of `running_mean`
    and `rsqrt(running_var + eps)`.

    ATen does not leave these empty (`batch_norm_cuda_out`,
    ATen/native/cuda/Normalization.cu): the autograd formula of
    `_native_batch_norm_legit_no_training` forwards both into
    `native_batch_norm_backward`, which is how a frozen BatchNorm still
    yields gradients. The accelerator kernel emits them as part of its pass;
    the CPU one does not, so they are composed through the dispatcher.
    """
    var save_mean = own(
        _nn_call1("aten::clone", "", [_nn_arg(mean), Value(TAG_NONE, 0, 0, 0)])
    )
    var shifted = own(
        _nn_call1(
            "aten::add",
            "Scalar",
            [
                _nn_arg(var_t),
                Value(TAG_SCALAR_DOUBLE, 0, f64_bits(eps), 0),
                Value(TAG_SCALAR_INT, 0, 1, 0),
            ],
        )
    )
    var save_invstd = own(_nn_call1("aten::rsqrt", "", [_nn_arg(shifted.t)]))
    _ = shifted.t.h
    ret_owned(rets, 1, save_mean)
    ret_owned(rets, 2, save_invstd)


def _bn_inference_cpu(
    rets: Values, a: T, gamma: T, beta: T, mean: T, var_t: T, eps: Float64
) raises:
    """The MAX CPU device's inference route: nn_ops `BatchNormSpec`, one
    elementwise pass over the whole tensor."""
    if (
        mean.stype != a.stype
        or var_t.stype != a.stype
        or gamma.stype != a.stype
        or beta.stype != a.stype
    ):
        # `_batch_norm_spec_into_go` carries a single dtype for the input and
        # every parameter; only the accelerator kernel takes them apart.
        unsupported(
            "batch norm on the CPU device needs the running statistics and"
            " the affine parameters in the input's own dtype"
        )
    var am = _mat(a)
    var out = own(new_like(a))
    var ctx = ctx_for(a.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("nn_ops", "BatchNormSpec")
    call.arg_dtype(0, a.dtype)
    call.arg_dtype(1, mean.dtype)
    call.arg_dtype(2, var_t.dtype)
    call.arg_dtype(3, gamma.dtype)
    call.arg_dtype(4, beta.dtype)
    call.out_dtype(out.t.dtype)
    call.spec(am.t.spec(cp))
    call.spec(mean.spec(cp))
    call.spec(var_t.spec(cp))
    call.spec(gamma.spec(cp))
    call.spec(beta.spec(cp))
    call.f64(eps)
    call.spec(out.t.spec(cp))
    call.run()
    _ = am.t.ptr
    _ = ctx
    ret_owned(rets, 0, out)
    _bn_set_saved_stats(rets, mean, var_t, eps)


def _bn_inference(args: Values, rets: Values, base: Int, eps_i: Int) raises:
    """`training=False` batch norm: one elementwise pass that also emits the
    two saved statistics torch's CUDA path fills (Normalization.cu:454)."""
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a, "batch norm")
    if not _is_float(a.dtype):
        unsupported("batch norm of dtype " + String(a.dtype))
    var channels = _bn_channels(a)
    if (
        v_is_none(args[unsafe_offset=1])
        or v_is_none(args[unsafe_offset=2])
        or v_is_none(args[unsafe_offset=base])
        or v_is_none(args[unsafe_offset=base + 1])
    ):
        unsupported(
            "inference batch norm needs weight, bias and both running"
            " statistics"
        )
    var gamma = v_tensor(args[unsafe_offset=1])
    var beta = v_tensor(args[unsafe_offset=2])
    var mean = v_tensor(args[unsafe_offset=base])
    var var_t = v_tensor(args[unsafe_offset=base + 1])
    var eps = v_f64(args[unsafe_offset=eps_i])
    _bn_param(gamma, a, channels, "batch norm weight")
    _bn_param(beta, a, channels, "batch norm bias")
    _bn_param(mean, a, channels, "batch norm running_mean")
    _bn_param(var_t, a, channels, "batch norm running_var")
    if mean.stype != var_t.stype or gamma.stype != beta.stype:
        unsupported("batch norm: running statistics (and affine) must pair up")
    if not _on_gpu(a):
        _bn_inference_cpu(rets, a, gamma, beta, mean, var_t, eps)
        return
    var inner = 1
    for i in range(2, a.rank):
        inner *= a.dim(i)
    var planes = a.dim(0) * channels
    if inner <= 0 or planes <= 0:
        unsupported("batch norm geometry must be positive")
    var am = _mat(a)
    var out = own(new_like(a))
    var save_mean = own(_channel_vec(channels, mean.stype, a.device))
    var save_invstd = own(_channel_vec(channels, mean.stype, a.device))
    var ctx = ctx_for(a.device)
    var call = KernelCall("normalization_forward_ops", "BatchNormInfer")
    call.arg_dtype(0, a.dtype)
    call.arg_dtype(1, mean.dtype)
    call.arg_dtype(2, gamma.dtype)
    call.out_dtype(out.t.dtype)
    call.int(out.t.ptr)
    call.int(am.t.ptr)
    call.int(mean.ptr)
    call.int(var_t.ptr)
    call.int(gamma.ptr)
    call.int(beta.ptr)
    var params = List[Int]()
    params.append(_f64_slot(eps))
    params.append(channels)
    params.append(inner)
    params.append(planes)
    params.append(1)
    params.append(1)
    params.append(save_mean.t.ptr)
    params.append(save_invstd.t.ptr)
    call.tuple(params)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = am.t.ptr
    _ = ctx
    ret_owned(rets, 0, out)
    ret_owned(rets, 1, save_mean)
    ret_owned(rets, 2, save_invstd)


def _channel_vec(channels: Int, stype: Int32, device: Int) raises -> T:
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 1] = channels
    return new_tensor(shape, 1, stype, device)


def _bn_affine_step(
    args: Values, body: Owned, i: Int, rank: Int, present: Bool, scale: Bool
) raises -> Owned:
    """`body * weight` (scale) or `body + bias` per channel, `body` untouched
    when that parameter is absent."""
    if not present:
        return _nn_hold(body.t)
    var p = _nn_hold(v_tensor(args[unsafe_offset=i]))
    var pf = _nn_cast(p, ST_FLOAT32)
    var pv = _nn_channel_view(pf, rank)
    if scale:
        return _nn_mul(body, pv)
    return _nn_add(body, pv)


def _bn_update_running(running: T, batch: Owned, momentum: Float64) raises:
    """ATen's in-place `running = (1 - momentum) * running + momentum * batch`,
    written as `running.add_(batch - running, alpha=momentum)`: the same
    value, and `mul_.Scalar` is not registered."""
    var cur = _nn_hold(running)
    var stat = _nn_cast(batch, running.stype)
    var delta = _nn_sub(stat, cur)
    _ = call_op(
        String("aten::add_"),
        String("Tensor"),
        [
            _nn_arg(running),
            _nn_arg(delta.t),
            Value(TAG_SCALAR_DOUBLE, 0, f64_bits(momentum), 0),
        ],
        1,
    )
    _ = delta.t.h


def _bn_training_cpu(
    args: Values,
    rets: Values,
    a: T,
    channels: Int,
    has_w: Bool,
    has_b: Bool,
    has_mean: Bool,
    momentum: Float64,
    eps: Float64,
) raises:
    """The MAX CPU device's training route.

    nn_ops has no training kernel, so the whole forward is composed through
    the dispatcher, the way ops_composed.mojo builds the batch norm BACKWARD:
    per-channel statistics over every dim but 1, then
    `(x - mean) * rsqrt(var + eps) * weight + bias`. Reduced precision
    accumulates in float32 (ATen's `opmath_t`, and what the accelerator
    kernel does) and rounds once, at the final cast.
    """
    var rank = a.rank
    if rank > 4:
        # The broadcast binary kernels stop at rank 4; the accelerator kernel
        # reduces any rank itself.
        unsupported(
            "training batch norm of rank > 4 on the CPU device (the"
            " accelerator kernel takes any rank)"
        )
    var n = a.numel // channels
    if n < 2:
        # ATen's unbiased running variance divides by N-1.
        unsupported("training batch norm over a single sample")
    var dims = List[Int64](capacity=rank - 1)
    dims.append(0)
    for i in range(2, rank):
        dims.append(Int64(i))

    var ain = _nn_hold(a)
    var af = _nn_cast(ain, ST_FLOAT32)
    var total = _nn_sum_dims(af, dims)
    var mean = _nn_div_scalar(total, Float64(n))
    var mean_b = _nn_channel_view(mean, rank)
    var centered = _nn_sub(af, mean_b)
    var squared = _nn_mul(centered, centered)
    var sq_total = _nn_sum_dims(squared, dims)
    # Biased (divided by N), which is what the normalization uses; the
    # running variance takes the unbiased one below.
    var variance = _nn_div_scalar(sq_total, Float64(n))
    var shifted = _nn_add_scalar(variance, eps)
    var invstd = _nn_rsqrt(shifted)
    var invstd_b = _nn_channel_view(invstd, rank)
    var normed = _nn_mul(centered, invstd_b)
    var scaled = _bn_affine_step(args, normed, 1, rank, has_w, True)
    var biased = _bn_affine_step(args, scaled, 2, rank, has_b, False)
    var out = _nn_cast(biased, a.stype)

    if has_mean:
        _bn_update_running(v_tensor(args[unsafe_offset=3]), mean, momentum)
        var unbiased = _nn_div_scalar(variance, Float64(n - 1) / Float64(n))
        _bn_update_running(v_tensor(args[unsafe_offset=4]), unbiased, momentum)
    # float32 statistics whatever the input dtype is, like the accelerator
    # kernel and `at::acc_type`.
    ret_owned(rets, 0, out)
    ret_owned(rets, 1, mean)
    ret_owned(rets, 2, invstd)


def _bn_training(args: Values, rets: Values) raises:
    """`aten::native_batch_norm` with `training=True`: per-channel statistics
    over N*HxW, ATen's running-stat update, then the elementwise pass."""
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a, "batch norm")
    if not _is_float(a.dtype):
        unsupported("training batch norm of dtype " + String(a.dtype))
    var channels = _bn_channels(a)
    var has_w = not v_is_none(args[unsafe_offset=1])
    var has_b = not v_is_none(args[unsafe_offset=2])
    var has_mean = not v_is_none(args[unsafe_offset=3])
    var has_var = not v_is_none(args[unsafe_offset=4])
    var momentum = v_f64(args[unsafe_offset=6])
    var eps = v_f64(args[unsafe_offset=7])
    if has_mean != has_var:
        unsupported(
            "training batch norm needs both running statistics or neither"
        )
    var gamma_ptr = 0
    var beta_ptr = 0
    var mean_ptr = 0
    var var_ptr = 0
    var stat_dtype = DType.float32
    var param_dtype = a.dtype
    if has_w:
        var w = v_tensor(args[unsafe_offset=1])
        _bn_param(w, a, channels, "batch norm weight")
        param_dtype = w.dtype
        gamma_ptr = w.ptr
    if has_b:
        var b = v_tensor(args[unsafe_offset=2])
        _bn_param(b, a, channels, "batch norm bias")
        if has_w and b.stype != v_tensor(args[unsafe_offset=1]).stype:
            unsupported("batch norm: weight and bias must share a dtype")
        if not has_w:
            param_dtype = b.dtype
        beta_ptr = b.ptr
    if has_mean:
        var m = v_tensor(args[unsafe_offset=3])
        var v = v_tensor(args[unsafe_offset=4])
        _bn_param(m, a, channels, "batch norm running_mean")
        _bn_param(v, a, channels, "batch norm running_var")
        if m.stype != v.stype:
            unsupported("batch norm: running statistics must share a dtype")
        stat_dtype = m.dtype
        mean_ptr = m.ptr
        var_ptr = v.ptr
    if not _on_gpu(a):
        _bn_training_cpu(
            args, rets, a, channels, has_w, has_b, has_mean, momentum, eps
        )
        return
    var hxw = 1
    for i in range(2, a.rank):
        hxw *= a.dim(i)
    var runs = a.dim(0)
    if runs * hxw < 2:
        # ATen's unbiased running variance divides by N-1.
        unsupported("training batch norm over a single sample")
    var am = _mat(a)
    var out = own(new_like(a))
    var save_mean = own(_channel_vec(channels, ST_FLOAT32, a.device))
    var save_invstd = own(_channel_vec(channels, ST_FLOAT32, a.device))
    var ctx = ctx_for(a.device)
    var call = KernelCall("normalization_forward_ops", "BatchNormTrain")
    call.arg_dtype(0, a.dtype)
    call.arg_dtype(1, stat_dtype)
    call.arg_dtype(2, param_dtype)
    call.out_dtype_i(0, out.t.dtype)
    call.out_dtype_i(1, DType.float32)
    call.out_dtype_i(2, DType.float32)
    call.flag("HAS_WEIGHT", 1 if has_w else 0)
    call.flag("HAS_BIAS", 1 if has_b else 0)
    call.flag("HAS_RUNNING", 1 if has_mean else 0)
    call.int(out.t.ptr)
    call.int(save_mean.t.ptr)
    call.int(save_invstd.t.ptr)
    call.int(am.t.ptr)
    call.int(gamma_ptr)
    call.int(beta_ptr)
    call.int(mean_ptr)
    call.int(var_ptr)
    var params = List[Int]()
    params.append(_f64_slot(eps))
    params.append(_f64_slot(momentum))
    params.append(channels)
    params.append(runs)
    params.append(hxw)
    params.append(1 if has_w else 0)
    params.append(1 if has_b else 0)
    params.append(1 if has_mean else 0)
    call.tuple(params)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = am.t.ptr
    _ = ctx
    ret_owned(rets, 0, out)
    ret_owned(rets, 1, save_mean)
    ret_owned(rets, 2, save_invstd)


# aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias,
#   Tensor? running_mean, Tensor? running_var, bool training, float momentum,
#   float eps) -> (Tensor, Tensor, Tensor)
def op_native_batch_norm(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    if v_bool(args[unsafe_offset=5]):
        _bn_training(args, rets)
        return
    _bn_inference(args, rets, 3, 7)


# aten::_native_batch_norm_legit_no_training(Tensor input, Tensor? weight,
#   Tensor? bias, Tensor running_mean, Tensor running_var, float momentum,
#   float eps) -> (Tensor, Tensor, Tensor)
def op_batch_norm_legit_no_training(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _bn_inference(args, rets, 3, 6)


# ---------------------------------------------------------------------------
# Group norm
# ---------------------------------------------------------------------------


# aten::native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N,
#   SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)
def op_native_group_norm(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var has_w = not v_is_none(args[unsafe_offset=1])
    var has_b = not v_is_none(args[unsafe_offset=2])
    var n = v_int(args[unsafe_offset=3])
    var c = v_int(args[unsafe_offset=4])
    var hxw = v_int(args[unsafe_offset=5])
    var group = v_int(args[unsafe_offset=6])
    var eps = v_f64(args[unsafe_offset=7])
    _require_mojo(a, "native_group_norm")
    if a.numel == 0 or not _is_float(a.dtype):
        unsupported("native_group_norm of dtype " + String(a.dtype))
    if group <= 0 or c % group != 0 or a.numel != n * c * hxw:
        unsupported("native_group_norm: bad group geometry")
    var cpg = c // group
    var cols = cpg * hxw
    var rows = n * group
    var keep = List[Held]()
    var gamma_ptr = 0
    var beta_ptr = 0
    var gamma_dtype = a.dtype
    var beta_dtype = a.dtype
    if has_w:
        var w = v_tensor(args[unsafe_offset=1])
        if (
            w.stype != a.stype
            or w.device != a.device
            or w.rank != 1
            or w.dim(0) != c
        ):
            unsupported("native_group_norm: unsupported weight")
        gamma_dtype = w.dtype
        gamma_ptr = _keep_ptr(keep, w)
    if has_b:
        var b = v_tensor(args[unsafe_offset=2])
        if (
            b.stype != a.stype
            or b.device != a.device
            or b.rank != 1
            or b.dim(0) != c
        ):
            unsupported("native_group_norm: unsupported bias")
        beta_dtype = b.dtype
        beta_ptr = _keep_ptr(keep, b)
    var am = _mat(a)
    var out = own(new_like(a))
    var stats = IndexList[MAX_RANK](1)
    stats[MAX_RANK - 2] = n
    stats[MAX_RANK - 1] = group
    var mean = own(new_tensor(stats, 2, ST_FLOAT32, a.device))
    var rstd = own(new_tensor(stats, 2, ST_FLOAT32, a.device))
    var ctx = ctx_for(a.device)
    if _on_gpu(a):
        var call = KernelCall("normalization_forward_ops", "GroupNormForward")
        call.arg_dtype(0, a.dtype)
        call.arg_dtype(1, gamma_dtype)
        call.arg_dtype(2, beta_dtype)
        call.out_dtype_i(0, out.t.dtype)
        call.out_dtype_i(1, DType.float32)
        call.out_dtype_i(2, DType.float32)
        call.flag("HAS_WEIGHT", 1 if has_w else 0)
        call.flag("HAS_BIAS", 1 if has_b else 0)
        call.int(out.t.ptr)
        call.int(mean.t.ptr)
        call.int(rstd.t.ptr)
        call.int(am.t.ptr)
        call.int(gamma_ptr)
        call.int(beta_ptr)
        call.int(rows)
        call.int(cols)
        call.f64(eps)
        call.int(1 if has_w else 0)
        call.int(1 if has_b else 0)
        call.int(hxw)
        call.int(cpg)
        call.int(group)
        call.int(ctx_ptr(ctx))
        call.run()
        _ = am.t.ptr
        _ = len(keep)
    else:
        var ones = own(_filled(a, c, 1.0))
        var zeros = own(_filled(a, c, 0.0))
        if not has_w:
            gamma_ptr = ones.t.ptr
        if not has_b:
            beta_ptr = zeros.t.ptr
        # `fill_value` writes the two synthesized parameters with
        # `DeviceContext.enqueue_memset`, which the MAX *CPU* context queues,
        # while the nn_ops kernel below runs inline on this thread: without
        # this drain the kernel reads uninitialized gamma/beta about one call
        # in ten (measured). Accelerators order both on the same stream and
        # never reach this branch.
        ctx.synchronize()
        var call = KernelCall("nn_ops", "GroupNorm")
        call.arg_dtype(0, a.dtype)
        call.arg_dtype(1, gamma_dtype)
        call.arg_dtype(2, beta_dtype)
        call.out_dtype_i(0, out.t.dtype)
        call.out_dtype_i(1, DType.float32)
        call.out_dtype_i(2, DType.float32)
        call.flag("HAS_WEIGHT", 1 if has_w else 0)
        call.flag("HAS_BIAS", 1 if has_b else 0)
        call.int(out.t.ptr)
        call.int(mean.t.ptr)
        call.int(rstd.t.ptr)
        call.int(am.t.ptr)
        call.int(gamma_ptr)
        call.int(beta_ptr)
        var params = List[Int]()
        params.append(_f64_slot(eps))
        params.append(rows)
        params.append(cols)
        params.append(hxw)
        params.append(group)
        params.append(cpg)
        call.tuple(params)
        call.int(dtype_code(a.dtype))
        call.int(ctx_ptr(ctx))
        call.run()
        _ = am.t.ptr
        _ = len(keep)
        _ = ones.t.ptr
        _ = zeros.t.ptr
    _ = ctx
    ret_owned(rets, 0, out)
    ret_owned(rets, 1, mean)
    ret_owned(rets, 2, rstd)


# ---------------------------------------------------------------------------
# NLL loss (the f32 / int64 kernels, `out=` ABI)
# ---------------------------------------------------------------------------


def _nll_inputs(
    args: Values, self_i: Int, target_i: Int, weight_i: Int, red_i: Int
) raises -> Tuple[Int, Int]:
    """Validate the two-dimensional f32/i64 contract; returns (rows, classes).

    Enqueue-only, so label values are never inspected on the host: each label
    must be in `[0, classes)` or exactly `ignore_index`.
    """
    if not v_is_none(args[unsafe_offset=weight_i]):
        unsupported("nll_loss with a class weight tensor")
    var reduction = v_int(args[unsafe_offset=red_i])
    if reduction < 0 or reduction > 2:
        raise Error("nll_loss: invalid reduction ", reduction)
    var log_probs = v_tensor(args[unsafe_offset=self_i])
    var labels = v_tensor(args[unsafe_offset=target_i])
    _require_mojo(log_probs, "nll_loss")
    if not _on_gpu(log_probs):
        unsupported("nll_loss needs an accelerator")
    if (
        log_probs.dtype != DType.float32
        or not log_probs.contig
        or log_probs.rank != 2
    ):
        unsupported("nll_loss covers a contiguous 2-D float32 input only")
    var rows = log_probs.dim(0)
    var classes = log_probs.dim(1)
    if (
        classes <= 0
        or labels.dtype != DType.int64
        or labels.device != log_probs.device
        or labels.rank != 1
        or labels.dim(0) != rows
    ):
        unsupported("nll_loss: target must be an int64[N] on the same device")
    return (rows, classes)


# aten::nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight,
#   int reduction, SymInt ignore_index, *, Tensor(a!) output,
#   Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))
def op_nll_loss_forward_output(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var geom = _nll_inputs(args, 0, 1, 2, 3)
    var rows = geom[0]
    var classes = geom[1]
    var reduction = v_int(args[unsafe_offset=3])
    var ignore_index = v_int(args[unsafe_offset=4])
    var log_probs = v_tensor(args[unsafe_offset=0])
    var labels = v_tensor(args[unsafe_offset=1])
    var output = v_tensor(args[unsafe_offset=5])
    var total_weight = v_tensor(args[unsafe_offset=6])
    if output.ptr == total_weight.ptr:
        unsupported("nll_loss_forward: output and total_weight alias")
    _nll_out_ok(output, log_probs, "output")
    _nll_out_ok(total_weight, log_probs, "total_weight")
    var out_shape = IndexList[MAX_RANK](1)
    var out_rank = 0
    if reduction == 0:
        out_shape[MAX_RANK - 1] = rows
        out_rank = 1
    var wo = _nll_dest(output, out_shape, out_rank)
    var wt = _nll_dest(total_weight, IndexList[MAX_RANK](1), 0)
    if rows == 0:
        # PyTorch defines the empty reduced loss as NaN for mean and zero for
        # sum; reduction=none already has no elements.
        fill_value(wt.t, 0.0)
        if reduction == 1:
            fill_value(wo.t, bits_f64(NAN_BITS))
        elif reduction == 2:
            fill_value(wo.t, 0.0)
    else:
        var lm = _mat(labels)
        var ctx = ctx_for(log_probs.device)
        var call = KernelCall("loss_ops", "NllLossForwardF32")
        call.arg_dtype(0, log_probs.dtype)
        call.arg_dtype(1, lm.t.dtype)
        call.out_dtype_i(0, DType.float32)
        call.out_dtype_i(1, DType.float32)
        call.flag("REDUCTION", reduction)
        call.int(wo.t.ptr)
        call.int(wt.t.ptr)
        call.int(log_probs.ptr)
        call.int(lm.t.ptr)
        call.int(rows)
        call.int(classes)
        call.int(reduction)
        call.int(ignore_index)
        call.int(ctx_ptr(ctx))
        call.run()
        _ = lm.t.ptr
        _ = ctx
    if wo.t.h != output.h:
        copy_strided_into(output, wo.t)
    if wt.t.h != total_weight.h:
        copy_strided_into(total_weight, wt.t)
    ret_ref(rets, 0, output)
    ret_ref(rets, 1, total_weight)


# aten::nll_loss_backward.grad_input(Tensor grad_output, Tensor self,
#   Tensor target, Tensor? weight, int reduction, SymInt ignore_index,
#   Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)
def op_nll_loss_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var geom = _nll_inputs(args, 1, 2, 3, 4)
    var rows = geom[0]
    var classes = geom[1]
    var reduction = v_int(args[unsafe_offset=4])
    var ignore_index = v_int(args[unsafe_offset=5])
    var grad = v_tensor(args[unsafe_offset=0])
    var log_probs = v_tensor(args[unsafe_offset=1])
    var labels = v_tensor(args[unsafe_offset=2])
    var weight_sum = v_tensor(args[unsafe_offset=6])
    var grad_input = v_tensor(args[unsafe_offset=7])
    if grad.dtype != DType.float32 or grad.device != log_probs.device:
        unsupported("nll_loss_backward: grad_output must be float32")
    if reduction == 0:
        if grad.rank != 1 or grad.dim(0) != rows:
            unsupported("nll_loss_backward: grad_output must be a [N] vector")
    elif grad.numel != 1:
        unsupported("nll_loss_backward: grad_output must be a scalar")
    if (
        weight_sum.dtype != DType.float32
        or weight_sum.device != log_probs.device
        or weight_sum.numel != 1
    ):
        unsupported("nll_loss_backward: total_weight must be a float32 scalar")
    _nll_out_ok(grad_input, log_probs, "grad_input")
    var wg = _nll_dest(grad_input, log_probs.shape, 2)
    if rows != 0:
        var lm = _mat(labels)
        var gm = _mat(grad)
        var wm = _mat(weight_sum)
        var ctx = ctx_for(log_probs.device)
        var call = KernelCall("loss_ops", "NllLossBackwardF32")
        call.arg_dtype(0, gm.t.dtype)
        call.arg_dtype(1, lm.t.dtype)
        call.arg_dtype(2, wm.t.dtype)
        call.out_dtype(DType.float32)
        call.flag("REDUCTION", reduction)
        call.int(wg.t.ptr)
        call.int(gm.t.ptr)
        call.int(lm.t.ptr)
        call.int(wm.t.ptr)
        call.int(rows)
        call.int(classes)
        call.int(reduction)
        call.int(ignore_index)
        call.int(ctx_ptr(ctx))
        call.run()
        _ = lm.t.ptr
        _ = gm.t.ptr
        _ = wm.t.ptr
        _ = ctx
    if wg.t.h != grad_input.h:
        copy_strided_into(grad_input, wg.t)
    ret_ref(rets, 0, grad_input)


def _nll_out_ok(dst: T, like: T, what: StaticString) raises:
    if dst.dtype != DType.float32 or dst.device != like.device:
        unsupported(
            String(what) + " must be a float32 tensor on the input's device"
        )


def _nll_dest(mut dst: T, shape: IndexList[MAX_RANK], rank: Int) raises -> Held:
    """Where the kernel writes for this `out=` argument.

    A wrong-shaped out is resized in place (the eager out= convention). A
    correctly-shaped but strided view keeps its storage and takes one ordered
    strided copy after the kernel; the common contiguous case is written
    directly.
    """
    var matches = dst.rank == rank
    if matches:
        for i in range(rank):
            if dst.dim(i) != shape[MAX_RANK - rank + i]:
                matches = False
                break
    if not matches:
        resize_out(dst, shape, rank)
        return Held(dst.copy(), False)
    if dst.contig:
        return Held(dst.copy(), False)
    return Held(new_tensor(shape, rank, dst.stype, dst.device), True)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


# aten::embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1,
#   bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
def op_embedding(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var table = v_tensor(args[unsafe_offset=0])
    var idx = v_tensor(args[unsafe_offset=1])
    _require_mojo(table, "embedding")
    if (
        table.device != idx.device
        or not _is_float(table.dtype)
        or table.rank != 2
        or (idx.dtype != DType.int32 and idx.dtype != DType.int64)
    ):
        unsupported(
            "embedding covers a float 2-D table with int32/int64 indices on"
            " one device"
        )
    var row_len = table.dim(1)
    var rank = idx.rank + 1
    if rank > MAX_RANK:
        raise Error("embedding: index rank ", idx.rank, " is too large")
    var shape = IndexList[MAX_RANK](1)
    for i in range(idx.rank):
        shape[MAX_RANK - rank + i] = idx.dim(i)
    shape[MAX_RANK - 1] = row_len
    var tm = _mat(table)
    var im = _mat(idx)
    var out = own(new_tensor(shape, rank, table.stype, table.device))
    if out.t.numel > 0:
        var ctx = ctx_for(table.device)
        var call = KernelCall("nn_ops", "Gather0")
        call.arg_dtype(0, table.dtype)
        call.arg_dtype(1, idx.dtype)
        call.out_dtype(out.t.dtype)
        call.int(out.t.ptr)
        call.int(tm.t.ptr)
        call.int(im.t.ptr)
        call.int(dtype_code(idx.dtype))
        call.int(idx.numel)
        call.int(row_len)
        call.int(table.dim(0))
        call.int(dtype_code(table.dtype))
        call.int(ctx_ptr(ctx))
        call.run()
        _ = tm.t.ptr
        _ = im.t.ptr
        _ = ctx
    ret_owned(rets, 0, out)


# aten::embedding_dense_backward(Tensor grad_output, Tensor indices,
#   SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq) -> Tensor
def op_embedding_dense_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var idx = v_tensor(args[unsafe_offset=1])
    var num_weights = v_int(args[unsafe_offset=2])
    var padding_idx = v_int(args[unsafe_offset=3])
    if v_bool(args[unsafe_offset=4]):
        unsupported(
            "embedding_dense_backward does not support scale_grad_by_freq=True"
        )
    _require_mojo(grad, "embedding_dense_backward")
    if not _on_gpu(grad):
        unsupported("embedding_dense_backward needs an accelerator")
    if (
        grad.device != idx.device
        or grad.dtype != DType.float32
        or idx.dtype != DType.int64
        or grad.rank < 1
        or num_weights < 0
    ):
        unsupported(
            "embedding_dense_backward covers a float32 gradient with int64"
            " indices"
        )
    var embedding_dim = grad.dim(grad.rank - 1)
    if grad.numel != idx.numel * embedding_dim:
        unsupported("embedding_dense_backward: index / gradient size mismatch")
    var gm = _mat(grad)
    var im = _mat(idx)
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 2] = num_weights
    shape[MAX_RANK - 1] = embedding_dim
    var out = own(new_tensor(shape, 2, ST_FLOAT32, grad.device))
    if out.t.numel > 0:
        # The kernel zeroes the whole output and accumulates into it.
        var ctx = ctx_for(grad.device)
        var call = KernelCall(
            "embedding_backward_ops", "EmbeddingDenseBackwardF32I64"
        )
        call.arg_dtype(0, grad.dtype)
        call.arg_dtype(1, idx.dtype)
        call.out_dtype(DType.float32)
        call.int(out.t.ptr)
        call.int(gm.t.ptr)
        call.int(im.t.ptr)
        call.int(idx.numel)
        call.int(embedding_dim)
        call.int(num_weights)
        call.int(padding_idx)
        call.int(0)
        call.int(ctx_ptr(ctx))
        call.run()
        _ = gm.t.ptr
        _ = im.t.ptr
        _ = ctx
    ret_owned(rets, 0, out)


# ---------------------------------------------------------------------------
# Pooling and bilinear upsampling (nn_ops, NCHW)
# ---------------------------------------------------------------------------


def _nchw(t: T, what: StaticString) raises:
    _require_mojo(t, what)
    if t.numel == 0 or not _is_float(t.dtype) or t.rank != 4:
        unsupported(String(what) + " covers a non-empty float NCHW tensor")


def _pool_shape(n: Int, c: Int, h: Int, w: Int) -> IndexList[MAX_RANK]:
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 4] = n
    shape[MAX_RANK - 3] = c
    shape[MAX_RANK - 2] = h
    shape[MAX_RANK - 1] = w
    return shape


# aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size,
#   int[2] stride=[], int[2] padding=0, int[2] dilation=1,
#   bool ceil_mode=False) -> (Tensor, Tensor)
def op_max_pool2d_with_indices(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    _nchw(a, "max_pool2d")
    var kernel = IntList(args[unsafe_offset=1])
    var stride = IntList(args[unsafe_offset=2])
    var k = _pair(kernel, "kernel_size")
    var s = k
    if len(stride) > 0:
        s = _pair(stride, "stride")
    var p = _pair(IntList(args[unsafe_offset=3]), "padding")
    var d = _pair(IntList(args[unsafe_offset=4]), "dilation")
    if v_bool(args[unsafe_offset=5]):
        unsupported("max_pool2d with ceil_mode=True")
    var in_h = a.dim(2)
    var in_w = a.dim(3)
    var out_h = (in_h + 2 * p[0] - (d[0] * (k[0] - 1) + 1)) // s[0] + 1
    var out_w = (in_w + 2 * p[1] - (d[1] * (k[1] - 1) + 1)) // s[1] + 1
    if out_h <= 0 or out_w <= 0:
        unsupported("max_pool2d: empty output")
    var planes = a.dim(0) * a.dim(1)
    var shape = _pool_shape(a.dim(0), a.dim(1), out_h, out_w)
    var am = _mat(a)
    var out = own(new_tensor(shape, 4, a.stype, a.device))
    var idx = own(new_tensor(shape, 4, ST_INT64, a.device))
    var ctx = ctx_for(a.device)
    var call = KernelCall("nn_ops", "MaxPool2dWithIndices")
    call.arg_dtype(0, a.dtype)
    call.out_dtype_i(0, out.t.dtype)
    call.out_dtype_i(1, DType.int64)
    call.int(out.t.ptr)
    call.int(idx.t.ptr)
    call.int(am.t.ptr)
    var params = List[Int]()
    params.append(in_h)
    params.append(in_w)
    params.append(out_h)
    params.append(out_w)
    params.append(k[0])
    params.append(k[1])
    params.append(s[0])
    params.append(s[1])
    params.append(p[0])
    params.append(p[1])
    params.append(d[0])
    params.append(d[1])
    params.append(planes)
    call.tuple(params)
    call.int(dtype_code(a.dtype))
    call.int(ctx_ptr(ctx))
    call.run()
    _ = am.t.ptr
    _ = ctx
    ret_owned(rets, 0, out)
    ret_owned(rets, 1, idx)


# aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[],
#   int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True,
#   int? divisor_override=None) -> Tensor
def op_avg_pool2d(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    _nchw(a, "avg_pool2d")
    var kernel = IntList(args[unsafe_offset=1])
    var stride = IntList(args[unsafe_offset=2])
    var k = _pair(kernel, "kernel_size")
    var s = k
    if len(stride) > 0:
        s = _pair(stride, "stride")
    var p = _pair(IntList(args[unsafe_offset=3]), "padding")
    if v_bool(args[unsafe_offset=4]):
        unsupported("avg_pool2d with ceil_mode=True")
    var count_include_pad = v_bool(args[unsafe_offset=5])
    var has_div = not v_is_none(args[unsafe_offset=6])
    var div = 0
    if has_div:
        div = v_int(args[unsafe_offset=6])
    if has_div and div == 0:
        raise Error("avg_pool2d: divisor_override must not be zero")
    var in_h = a.dim(2)
    var in_w = a.dim(3)
    var out_h = (in_h + 2 * p[0] - k[0]) // s[0] + 1
    var out_w = (in_w + 2 * p[1] - k[1]) // s[1] + 1
    if out_h <= 0 or out_w <= 0:
        unsupported("avg_pool2d: empty output")
    var shape = _pool_shape(a.dim(0), a.dim(1), out_h, out_w)
    var am = _mat(a)
    var out = own(new_tensor(shape, 4, a.stype, a.device))
    var ctx = ctx_for(a.device)
    var call = KernelCall("nn_ops", "AvgPool2d")
    call.arg_dtype(0, a.dtype)
    call.out_dtype(out.t.dtype)
    call.flag("COUNT_INCLUDE_PAD", 1 if count_include_pad else 0)
    call.flag("HAS_DIVISOR_OVERRIDE", 1 if has_div else 0)
    call.int(out.t.ptr)
    call.int(am.t.ptr)
    var params = List[Int]()
    params.append(in_h)
    params.append(in_w)
    params.append(out_h)
    params.append(out_w)
    params.append(k[0])
    params.append(k[1])
    params.append(s[0])
    params.append(s[1])
    params.append(p[0])
    params.append(p[1])
    params.append(1 if count_include_pad else 0)
    params.append(div)
    params.append(a.dim(0) * a.dim(1))
    call.tuple(params)
    call.int(dtype_code(a.dtype))
    call.int(ctx_ptr(ctx))
    call.run()
    _ = am.t.ptr
    _ = ctx
    ret_owned(rets, 0, out)


# aten::_adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor
def op_adaptive_avg_pool2d(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    _nchw(a, "_adaptive_avg_pool2d")
    var osize = _pair(IntList(args[unsafe_offset=1]), "output_size")
    if osize[0] <= 0 or osize[1] <= 0:
        unsupported("_adaptive_avg_pool2d: empty output")
    var shape = _pool_shape(a.dim(0), a.dim(1), osize[0], osize[1])
    var am = _mat(a)
    var out = own(new_tensor(shape, 4, a.stype, a.device))
    var ctx = ctx_for(a.device)
    var call = KernelCall("nn_ops", "AdaptiveAvgPool2d")
    call.arg_dtype(0, a.dtype)
    call.out_dtype(out.t.dtype)
    call.int(out.t.ptr)
    call.int(am.t.ptr)
    var params = List[Int]()
    params.append(a.dim(2))
    params.append(a.dim(3))
    params.append(osize[0])
    params.append(osize[1])
    params.append(a.dim(0) * a.dim(1))
    call.tuple(params)
    call.int(dtype_code(a.dtype))
    call.int(ctx_ptr(ctx))
    call.run()
    _ = am.t.ptr
    _ = ctx
    ret_owned(rets, 0, out)


def _area_pixel_scale(
    in_size: Int, out_size: Int, align_corners: Bool, scale: Float64
) -> Float64:
    """torch's `area_pixel_compute_scale` for one axis (scale <= 0 = unset)."""
    if align_corners:
        if out_size <= 1:
            return 0.0
        return Float64(in_size - 1) / Float64(out_size - 1)
    if scale > 0.0:
        return 1.0 / scale
    return Float64(in_size) / Float64(out_size)


# aten::upsample_bilinear2d(Tensor self, SymInt[2] output_size,
#   bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
def op_upsample_bilinear2d(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    _nchw(a, "upsample_bilinear2d")
    var osize = _pair(IntList(args[unsafe_offset=1]), "output_size")
    var align_corners = v_bool(args[unsafe_offset=2])
    var scale_h = -1.0
    var scale_w = -1.0
    if not v_is_none(args[unsafe_offset=3]):
        scale_h = v_f64(args[unsafe_offset=3])
    if not v_is_none(args[unsafe_offset=4]):
        scale_w = v_f64(args[unsafe_offset=4])
    if osize[0] <= 0 or osize[1] <= 0:
        unsupported("upsample_bilinear2d: empty output")
    var in_h = a.dim(2)
    var in_w = a.dim(3)
    var ratio_h = _area_pixel_scale(in_h, osize[0], align_corners, scale_h)
    var ratio_w = _area_pixel_scale(in_w, osize[1], align_corners, scale_w)
    var shape = _pool_shape(a.dim(0), a.dim(1), osize[0], osize[1])
    var am = _mat(a)
    var out = own(new_tensor(shape, 4, a.stype, a.device))
    var ctx = ctx_for(a.device)
    var call = KernelCall("nn_ops", "UpsampleBilinear2d")
    call.arg_dtype(0, a.dtype)
    call.out_dtype(out.t.dtype)
    call.flag("ALIGN_CORNERS", 1 if align_corners else 0)
    call.int(out.t.ptr)
    call.int(am.t.ptr)
    var params = List[Int]()
    params.append(_f64_slot(ratio_h))
    params.append(_f64_slot(ratio_w))
    params.append(in_h)
    params.append(in_w)
    params.append(osize[0])
    params.append(osize[1])
    params.append(a.dim(0) * a.dim(1))
    params.append(1 if align_corners else 0)
    call.tuple(params)
    call.int(dtype_code(a.dtype))
    call.int(ctx_ptr(ctx))
    call.run()
    _ = am.t.ptr
    _ = ctx
    ret_owned(rets, 0, out)


def register_nn(site: Site) raises:
    impl[op_adaptive_avg_pool2d, "_adaptive_avg_pool2d"](site)
    impl[op_log_softmax, "_log_softmax"](site)
    impl[op_log_softmax_backward_data, "_log_softmax_backward_data"](site)
    impl[
        op_batch_norm_legit_no_training, "_native_batch_norm_legit_no_training"
    ](site)
    impl[op_softmax, "_softmax"](site)
    impl[op_avg_pool2d, "avg_pool2d"](site)
    impl[op_embedding, "embedding"](site)
    impl[op_embedding_dense_backward, "embedding_dense_backward"](site)
    impl[op_max_pool2d_with_indices, "max_pool2d_with_indices"](site)
    impl[op_native_batch_norm, "native_batch_norm"](site)
    impl[op_native_group_norm, "native_group_norm"](site)
    impl[op_native_layer_norm, "native_layer_norm"](site)
    impl[op_native_layer_norm_backward, "native_layer_norm_backward"](site)
    impl[op_nll_loss_backward_grad_input, "nll_loss_backward.grad_input"](site)
    impl[op_nll_loss_forward_output, "nll_loss_forward.output"](site)
    impl[op_upsample_bilinear2d, "upsample_bilinear2d"](site)


@export
def tmb_op_address() abi("C") -> Int:
    """Entry of this file's one-op extension: the address of the op the
    TMB_OP define selected (registry.mojo)."""
    return op_address_of[register_nn]()
