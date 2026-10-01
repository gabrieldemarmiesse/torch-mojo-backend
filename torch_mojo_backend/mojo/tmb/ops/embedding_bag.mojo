"""ATen ops: embedding_bag group (see agents_docs/native_backend.md).

* _embedding_bag / _embedding_bag_forward_only -- the `embedding_bag`
  family's `EmbeddingBagForward`, a port of ATen's
  EmbeddingBag_updateOutputKernel_sum_mean / _max (native/cuda/
  EmbeddingBag.cu): one thread per (bag, feature) walks the bag in order,
  so the output is deterministic and reduced in CUDA's order.
* _embedding_bag_backward -- ATen's generic `_embedding_bag_backward_symint`
  (native/EmbeddingBag.cpp): index promotion, offset2bag rebuilt when the
  forward did not return one, then the dense backward.
* _embedding_bag_dense_backward / _embedding_bag_per_sample_weights_backward
  -- compositions of ops this device already runs: sum / mean gather the
  bag gradient of every index (index_select), scale it (per-sample weight,
  bag size, frequency) in the accumulator dtype and index_add it into the
  weight gradient; max scatter_adds into the rows max_indices names. The
  float sums are atomic (CUDA sorts and segment-reduces instead), so they
  can differ from CUDA in the last bits.
* embedding_renorm_ -- the indices are made unique on the host (one read,
  CUDA reads none), then `EmbeddingRenorm` rescales each row in place.
"""
from std.utils import IndexList

from tmb.backend.abi import (
    Owned,
    ST_FLOAT32,
    ST_FLOAT64,
    ST_INT32,
    ST_INT64,
    T,
    TAG_BOOL,
    TAG_INT,
    TAG_INT_LIST,
    TAG_NONE,
    TAG_SCALAR_INT,
    Value,
    Values,
    alert_not_deterministic,
    cpu_empty,
    dtype_code,
    index_error,
    new_tensor,
    own,
    own_if_new,
    ret_owned,
    ret_ref,
    tensor_arg,
    unsupported,
    v_bool,
    v_f64,
    v_int,
    v_opt_tensor,
    v_tensor,
)
from tmb.backend.device import (
    copy_from_host,
    copy_to_host,
    ctx_for,
    ctx_ptr,
)
from tmb.backend.kernel_call import KernelCall
from tmb.backend.registry import Site, impl
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.common import call_op, cast_to, contiguous, fill_value
from tmb.ops.data_movement import _scalar_type_name

comptime _MODE_SUM = 0
comptime _MODE_MEAN = 1
comptime _MODE_MAX = 2


def _t(t: T) -> Value:
    return tensor_arg(t)


def _int(x: Int) -> Value:
    return Value(TAG_INT, 0, Int64(x), 0)


def _scalar(x: Int) -> Value:
    return Value(TAG_SCALAR_INT, 0, Int64(x), 0)


def _op(name: String, overload: String, var args: List[Value]) raises -> Owned:
    """One aten op through the dispatcher, its one Tensor result (owned)."""
    var r = call_op(name, overload, args^, 1)
    return own(r.take_tensor(0))


def _vec(n: Int) -> IndexList[MAX_RANK]:
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 1] = n
    return shape


def _mat(rows: Int, cols: Int) -> IndexList[MAX_RANK]:
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 2] = rows
    shape[MAX_RANK - 1] = cols
    return shape


def _zeros(
    shape: IndexList[MAX_RANK], rank: Int, stype: Int32, device: Int
) raises -> Owned:
    var z = own(new_tensor(shape, rank, stype, device))
    fill_value(z.t, 0.0)
    return z^


def _check_index_dtype(t: T, name: String, arg: Int, what: String) raises:
    """`checkScalarTypes(what, t, {kLong, kInt})`."""
    if t.dtype != DType.int64 and t.dtype != DType.int32:
        raise Error(
            "Expected tensor for argument #",
            arg,
            " '",
            name,
            "' to have one of the following scalar types: Long, Int; but got ",
            _scalar_type_name(t.dtype),
            " instead (while checking arguments for ",
            what,
            ")",
        )


def _index_stype(indices: T, offsets: T) -> Int32:
    """`promoteIndicesAndOffsets`: the common index dtype."""
    if indices.dtype == DType.int64 or offsets.dtype == DType.int64:
        return ST_INT64
    return ST_INT32


def _as_int64(t: T) raises -> Owned:
    """`t` as a contiguous int64 tensor (itself when it already is one)."""
    var c = own_if_new(cast_to(t, ST_INT64), t)
    if c.t.contig:
        return c^
    var d = own(contiguous(c.t))
    _ = c^
    return d^


def _acc_stype(t: T) -> Int32:
    """ATen's `acc_type<scalar_t, true>`: float for the half types."""
    return ST_FLOAT64 if t.dtype == DType.float64 else ST_FLOAT32


def _is_bag_dtype(dt: DType) -> Bool:
    return (
        dt == DType.float32
        or dt == DType.float16
        or dt == DType.bfloat16
        or dt == DType.float64
    )


def _check_bag_dtype(t: T, what: String) raises:
    if not _is_bag_dtype(t.dtype):
        raise Error(
            '"',
            what,
            "\" not implemented for '",
            _scalar_type_name(t.dtype),
            "'",
        )
    if t.dtype == DType.float64 and ctx_for(t.device).api() == "metal":
        unsupported("aten::" + what + " of float64 on Apple GPU")


def _read_flag(flag: T) raises -> Int:
    var host = own(cpu_empty(flag.shape, 1, ST_INT32))
    var ctx = ctx_for(flag.device)
    copy_to_host(ctx, flag.ptr, host.t.ptr, 4)
    _ = ctx
    var v = Int(
        Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=host.t.ptr)[]
    )
    _ = host^
    return v


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


# aten::_embedding_bag(Tensor weight, Tensor indices, Tensor offsets,
#   bool scale_grad_by_freq=False, int mode=0, bool sparse=False,
#   Tensor? per_sample_weights=None, bool include_last_offset=False,
#   int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
# (and _embedding_bag_forward_only, the same schema)
def op_embedding_bag(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var weight = v_tensor(args[unsafe_offset=0])
    var indices = v_tensor(args[unsafe_offset=1])
    var offsets = v_tensor(args[unsafe_offset=2])
    var mode = v_int(args[unsafe_offset=4])
    var psw = v_opt_tensor(args[unsafe_offset=6])
    var include_last_offset = v_bool(args[unsafe_offset=7])
    var padding_idx = v_int(args[unsafe_offset=8])
    if indices.rank != 1 and indices.rank != 2:
        raise Error(
            "input has to be a 1D or 2D Tensor, but got Tensor of dimension ",
            indices.rank,
        )
    if indices.rank == 1 and offsets.rank != 1:
        raise Error(
            "offsets has to be a 1D Tensor, but got Tensor of dimension ",
            offsets.rank,
        )
    if weight.rank != 2:
        raise Error(
            "weight has to be a 2D Tensor, but got Tensor of dimension ",
            weight.rank,
        )
    if indices.rank != 1:
        unsupported(
            "_embedding_bag with 2-D indices (F.embedding_bag flattens them)"
        )
    _check_index_dtype(indices, "indices", 1, "embedding_bag_cuda")
    _check_index_dtype(offsets, "offsets", 1, "embedding_bag_cuda")
    if indices.device != weight.device or offsets.device != weight.device:
        raise Error(
            "Expected all tensors to be on the same device (while checking"
            " arguments for embedding_bag_cuda)"
        )
    _check_bag_dtype(weight, "embedding_bag_cuda")
    var num_indices = indices.dim(0)
    var num_bags = offsets.dim(0)
    if include_last_offset:
        if num_bags < 1:
            raise Error("include_last_offset: numBags should be at least 1")
        num_bags -= 1
    var has_psw = False
    var psw_ptr = 0
    var psw_stride = 0
    if psw and mode != _MODE_MAX:
        var p = psw.value().copy()
        if p.dtype != weight.dtype:
            raise Error(
                "expected scalar type ",
                _scalar_type_name(weight.dtype),
                " but found ",
                _scalar_type_name(p.dtype),
            )
        if p.rank != 1 or p.numel != num_indices or p.device != weight.device:
            raise Error(
                "per_sample_weights must be a 1-D tensor with one weight per"
                " index, on the weight's device"
            )
        has_psw = True
        psw_ptr = p.ptr
        psw_stride = p.stride(0)
    var idx_st = _index_stype(indices, offsets)
    var features = weight.dim(1)
    var idx = _as_int64(indices)
    var offs = _as_int64(offsets)
    var device = weight.device
    var output = own(
        new_tensor(_mat(num_bags, features), 2, weight.stype, device)
    )
    var offset2bag = _zeros(_vec(num_indices), 1, ST_INT64, device)
    var bag_size = _zeros(_vec(offsets.dim(0)), 1, ST_INT64, device)
    var max_indices = own(
        new_tensor(
            _mat(num_bags, features) if mode == _MODE_MAX else _vec(0),
            2 if mode == _MODE_MAX else 1,
            ST_INT64,
            device,
        )
    )
    if num_bags * features > 0:
        var flag = _zeros(_vec(1), 1, ST_INT32, device)
        var ctx = ctx_for(device)
        var call = KernelCall("embedding_bag", "EmbeddingBagForward")
        call.arg_dtype(0, weight.dtype)
        call.int(output.t.ptr)
        call.int(offset2bag.t.ptr)
        call.int(bag_size.t.ptr)
        call.int(max_indices.t.ptr)
        call.int(psw_ptr)
        call.int(idx.t.ptr)
        call.int(offs.t.ptr)
        call.int(weight.ptr)
        call.int(flag.t.ptr)
        call.tuple(
            [
                num_indices,
                num_bags,
                features,
                weight.stride(0),
                weight.stride(1),
                mode,
                padding_idx,
                weight.dim(0),
                psw_stride,
                1 if has_psw else 0,
            ]
        )
        call.int(dtype_code(weight.dtype))
        call.int(ctx_ptr(ctx))
        call.run()
        _ = ctx
        var bad = _read_flag(flag.t)
        _ = flag^
        if bad == 1:
            index_error(
                "Invalid input index in EmbeddingBag: index out of range [0, "
                + String(weight.dim(0))
                + ")"
            )
        if bad == 2:
            raise Error("embedding_bag: offsets must be non-decreasing")
    _ = idx^
    _ = offs^
    ret_owned(rets, 0, output)
    if idx_st == ST_INT64:
        ret_owned(rets, 1, offset2bag)
        ret_owned(rets, 2, bag_size)
        ret_owned(rets, 3, max_indices)
    else:
        var a = own(cast_to(offset2bag.t, idx_st))
        var b = own(cast_to(bag_size.t, idx_st))
        var c = own(cast_to(max_indices.t, idx_st))
        ret_owned(rets, 1, a)
        ret_owned(rets, 2, b)
        ret_owned(rets, 3, c)


# ---------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------


def _offset2bag(offsets: T, num_indices: Int, stype: Int32) raises -> Owned:
    """ATen's `make_offset2bag`: ones scattered at the offsets, minus one,
    cumulatively summed -- the bag of every index."""
    var z = _zeros(_vec(num_indices + 1), 1, stype, offsets.device)
    var ones = own(
        new_tensor(offsets.shape, offsets.rank, stype, offsets.device)
    )
    fill_value(ones.t, 1.0)
    var added = _op(
        "aten::index_add",
        "",
        [_t(z.t), _int(0), _t(offsets), _t(ones.t), _scalar(1)],
    )
    var csum = _op(
        "aten::cumsum", "", [_t(added.t), _int(0), Value(TAG_NONE, 0, 0, 0)]
    )
    var shifted = _op(
        "aten::sub", "Scalar", [_t(csum.t), _scalar(1), _scalar(1)]
    )
    # A leading narrow of a contiguous vector: contiguous, as offset2bag
    # must be.
    var out = _op(
        "aten::narrow", "", [_t(shifted.t), _int(0), _int(0), _int(num_indices)]
    )
    _ = shifted^
    _ = csum^
    _ = added^
    _ = ones^
    _ = z^
    return out^


# aten::_embedding_bag_backward(Tensor grad, Tensor indices, Tensor offsets,
#   Tensor offset2bag, Tensor bag_size, Tensor maximum_indices,
#   SymInt num_weights, bool scale_grad_by_freq, int mode, bool sparse,
#   Tensor? per_sample_weights, int padding_idx=-1) -> Tensor
def op_embedding_bag_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var indices_in = v_tensor(args[unsafe_offset=1])
    var offsets_in = v_tensor(args[unsafe_offset=2])
    var offset2bag = v_tensor(args[unsafe_offset=3])
    var st = _index_stype(indices_in, offsets_in)
    var indices = own_if_new(cast_to(indices_in, st), indices_in)
    var offsets = own_if_new(cast_to(offsets_in, st), offsets_in)
    _check_index_dtype(indices.t, "indices", 1, "embedding_bag")
    _check_index_dtype(offsets.t, "offsets", 1, "embedding_bag")
    if not indices.t.contig or not offsets.t.contig:
        raise Error(
            "Expected contiguous tensor, but got non-contiguous tensor for"
            " argument #1 'indices' (while checking arguments for"
            " embedding_bag)"
        )
    if v_bool(args[unsafe_offset=9]):
        unsupported(
            "_embedding_bag_backward with sparse=True (no sparse tensors on"
            " the mojo device)"
        )
    var o2b = Optional[Owned](None)
    var o2b_t = offset2bag.copy()
    if indices.t.numel != 0 and offset2bag.numel == 0:
        o2b = _offset2bag(offsets.t, indices.t.dim(0), st)
        o2b_t = o2b.value().t.copy()
    else:
        _check_index_dtype(offset2bag, "offset2bag", 1, "embedding_bag")
    var out = _op(
        "aten::_embedding_bag_dense_backward",
        "",
        [
            args[unsafe_offset=0].copy(),
            _t(indices.t),
            _t(o2b_t),
            args[unsafe_offset=4].copy(),
            args[unsafe_offset=5].copy(),
            args[unsafe_offset=6].copy(),
            args[unsafe_offset=7].copy(),
            args[unsafe_offset=8].copy(),
            args[unsafe_offset=10].copy(),
            args[unsafe_offset=11].copy(),
        ],
    )
    _ = o2b^
    _ = indices^
    _ = offsets^
    ret_owned(rets, 0, out)


def _unsqueeze1(t: T) raises -> Owned:
    return _op("aten::unsqueeze", "", [_t(t), _int(1)])


# aten::_embedding_bag_dense_backward(Tensor grad, Tensor indices,
#   Tensor offset2bag, Tensor bag_size, Tensor maximum_indices,
#   SymInt num_weights, bool scale_grad_by_freq, int mode,
#   Tensor? per_sample_weights, int padding_idx=-1) -> Tensor
def op_embedding_bag_dense_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var indices = v_tensor(args[unsafe_offset=1])
    var offset2bag = v_tensor(args[unsafe_offset=2])
    var bag_size = v_tensor(args[unsafe_offset=3])
    var max_indices = v_tensor(args[unsafe_offset=4])
    var num_weights = v_int(args[unsafe_offset=5])
    var scale_grad_by_freq = v_bool(args[unsafe_offset=6])
    var mode = v_int(args[unsafe_offset=7])
    var psw = v_opt_tensor(args[unsafe_offset=8])
    var padding_idx = v_int(args[unsafe_offset=9])
    _check_bag_dtype(grad, "embedding_bag_backward_cuda")
    if grad.rank != 2:
        raise Error("embedding_bag_backward: grad must be 2-D")
    if indices.device != grad.device:
        raise Error(
            "Expected all tensors to be on the same device (while checking"
            " arguments for embedding_bag_cuda)"
        )
    var features = grad.dim(1)
    var device = grad.device
    if mode == _MODE_MAX:
        if psw:
            raise Error("embedding_bag: per_sample_weights is unused by max")
        alert_not_deterministic("embedding_bag_backward_cuda_max")
        # One spare row takes the empty bags (-1) and padding_idx, then is
        # dropped: no contribution is masked by a multiply (0 * inf).
        var gw = _zeros(_mat(num_weights + 1, features), 2, grad.stype, device)
        var neg = _op("aten::lt", "Scalar", [_t(max_indices), _scalar(0)])
        var mi = _op(
            "aten::masked_fill",
            "Scalar",
            [_t(max_indices), _t(neg.t), _scalar(num_weights)],
        )
        if padding_idx >= 0:
            var pad = _op(
                "aten::eq", "Scalar", [_t(mi.t), _scalar(padding_idx)]
            )
            var mi2 = _op(
                "aten::masked_fill",
                "Scalar",
                [_t(mi.t), _t(pad.t), _scalar(num_weights)],
            )
            _ = pad^
            _ = mi^  # alive until the call above has read it
            mi = mi2^
        if max_indices.numel > 0:
            var g = own_if_new(contiguous(grad), grad)
            _ = call_op(
                "aten::scatter_add_",
                "",
                [_t(gw.t), _int(0), _t(mi.t), _t(g.t)],
                1,
            )
            _ = g^
        var res = _op(
            "aten::narrow", "", [_t(gw.t), _int(0), _int(0), _int(num_weights)]
        )
        _ = mi^
        _ = neg^
        _ = gw^
        ret_owned(rets, 0, res)
        return
    if mode == _MODE_MEAN and psw:
        raise Error(
            "embedding_bag: per_sample_weights only supported with mode='sum'"
        )
    var acc = _acc_stype(grad)
    if indices.numel == 0:
        var z = _zeros(_mat(num_weights, features), 2, grad.stype, device)
        ret_owned(rets, 0, z)
        return
    var g = own_if_new(cast_to(grad, acc), grad)
    var rows = _op("aten::index_select", "", [_t(g.t), _int(0), _t(offset2bag)])
    if psw:
        var w = own_if_new(cast_to(psw.value(), acc), psw.value())
        var w2 = _unsqueeze1(w.t)
        var scaled = _op("aten::mul", "Tensor", [_t(rows.t), _t(w2.t)])
        _ = rows^  # alive until the call above has read it
        rows = scaled^
        _ = w2^
        _ = w^
    if mode == _MODE_MEAN:
        var bs = _op(
            "aten::index_select", "", [_t(bag_size), _int(0), _t(offset2bag)]
        )
        var bsf = own_if_new(cast_to(bs.t, acc), bs.t)
        var bs2 = _unsqueeze1(bsf.t)
        var averaged = _op("aten::div", "Tensor", [_t(rows.t), _t(bs2.t)])
        _ = rows^
        rows = averaged^
        _ = bs2^
        _ = bsf^
        _ = bs^
    if scale_grad_by_freq:
        var counts = _zeros(_vec(num_weights), 1, acc, device)
        var ones = own(new_tensor(_vec(indices.numel), 1, acc, device))
        fill_value(ones.t, 1.0)
        var c = _op(
            "aten::index_add",
            "",
            [_t(counts.t), _int(0), _t(indices), _t(ones.t), _scalar(1)],
        )
        var per = _op("aten::index_select", "", [_t(c.t), _int(0), _t(indices)])
        var per2 = _unsqueeze1(per.t)
        var freq = _op("aten::div", "Tensor", [_t(rows.t), _t(per2.t)])
        _ = rows^
        rows = freq^
        _ = per2^
        _ = per^
        _ = c^
        _ = ones^
        _ = counts^
    # padding_idx rows land in a spare row that is dropped afterwards.
    var target = indices.copy()
    var remapped = Optional[Owned](None)
    if padding_idx >= 0:
        var pad = _op("aten::eq", "Scalar", [_t(indices), _scalar(padding_idx)])
        remapped = _op(
            "aten::masked_fill",
            "Scalar",
            [_t(indices), _t(pad.t), _scalar(num_weights)],
        )
        target = remapped.value().t.copy()
        _ = pad^
    var gw = _zeros(_mat(num_weights + 1, features), 2, acc, device)
    var summed = _op(
        "aten::index_add",
        "",
        [_t(gw.t), _int(0), _t(target), _t(rows.t), _scalar(1)],
    )
    var top = _op(
        "aten::narrow", "", [_t(summed.t), _int(0), _int(0), _int(num_weights)]
    )
    _ = summed^
    _ = gw^
    _ = remapped^
    _ = rows^
    _ = g^
    if acc == grad.stype:
        ret_owned(rets, 0, top)  # a leading narrow: contiguous
        return
    var res = own(cast_to(top.t, grad.stype))
    _ = top^
    ret_owned(rets, 0, res)


# aten::_embedding_bag_per_sample_weights_backward(Tensor grad, Tensor weight,
#   Tensor indices, Tensor offsets, Tensor offset2bag, int mode,
#   int padding_idx=-1) -> Tensor
def op_embedding_bag_per_sample_weights_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var weight = v_tensor(args[unsafe_offset=1])
    var indices = v_tensor(args[unsafe_offset=2])
    var offset2bag = v_tensor(args[unsafe_offset=4])
    var mode = v_int(args[unsafe_offset=5])
    var padding_idx = v_int(args[unsafe_offset=6])
    if mode != _MODE_SUM:
        raise Error(
            "embedding_bag_backward: per_sample_weights only supported for"
            " mode='sum'"
        )
    _check_bag_dtype(grad, "_embedding_bag_per_sample_weights_backward_cuda")
    var n = indices.numel
    if n == 0:
        var e = own(new_tensor(_vec(0), 1, grad.stype, grad.device))
        ret_owned(rets, 0, e)
        return
    var acc = _acc_stype(grad)
    var g = own_if_new(cast_to(grad, acc), grad)
    var w = own_if_new(cast_to(weight, acc), weight)
    var gr = _op("aten::index_select", "", [_t(g.t), _int(0), _t(offset2bag)])
    var wr = _op("aten::index_select", "", [_t(w.t), _int(0), _t(indices)])
    var prod = _op("aten::mul", "Tensor", [_t(gr.t), _t(wr.t)])
    var dims = List[Int64]()
    dims.append(Int64(1))
    var s = _op(
        "aten::sum",
        "dim_IntList",
        [
            _t(prod.t),
            Value(TAG_INT_LIST, Int32(1), Int64(Int(dims.unsafe_ptr())), 0),
            Value(TAG_BOOL, 0, 0, 0),
            Value(TAG_NONE, 0, 0, 0),
        ],
    )
    _ = dims^
    if padding_idx >= 0:
        var pad = _op("aten::eq", "Scalar", [_t(indices), _scalar(padding_idx)])
        var masked = _op(
            "aten::masked_fill", "Scalar", [_t(s.t), _t(pad.t), _scalar(0)]
        )
        _ = pad^
        _ = s^  # alive until the call above has read it
        s = masked^
    _ = prod^
    _ = wr^
    _ = gr^
    _ = w^
    _ = g^
    if acc == grad.stype:
        ret_owned(rets, 0, s)
        return
    var res = own(cast_to(s.t, grad.stype))
    _ = s^
    ret_owned(rets, 0, res)


# ---------------------------------------------------------------------------
# embedding_renorm_
# ---------------------------------------------------------------------------


# aten::embedding_renorm_(Tensor(a!) self, Tensor indices, float max_norm,
#   float norm_type) -> Tensor(a!)
def op_embedding_renorm_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var indices = v_tensor(args[unsafe_offset=1])
    var max_norm = v_f64(args[unsafe_offset=2])
    var norm_type = v_f64(args[unsafe_offset=3])
    if t.rank != 2:
        raise Error(
            "Expected 2-dimensional tensor, but got ",
            t.rank,
            (
                "-dimensional tensor for argument #1 'self' (while checking"
                " arguments for embedding_renorm_)"
            ),
        )
    _check_index_dtype(indices, "indices", 1, "embedding_renorm_")
    if indices.device != t.device:
        raise Error(
            "Expected all tensors to be on the same device (while checking"
            " arguments for embedding_renorm)"
        )
    _check_bag_dtype(t, "embedding_renorm_cuda_")
    if indices.numel == 0:
        ret_ref(rets, 0, t)
        return
    # Wrap, range-check and deduplicate on the host: two threads renorming
    # one row would race (CUDA sorts and uniques on the device).
    var idx = _as_int64(indices)
    var n = idx.t.numel
    var host = own(cpu_empty(idx.t.shape, idx.t.rank, ST_INT64))
    var ctx = ctx_for(t.device)
    copy_to_host(ctx, idx.t.ptr, host.t.ptr, n * 8)
    _ = idx^
    var p = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=host.t.ptr)
    var num_weights = t.dim(0)
    var seen = List[Bool](capacity=num_weights)
    for _ in range(num_weights):
        seen.append(False)
    var count = 0
    for i in range(n):
        var r = Int(p[unsafe_offset=i])
        if r < -num_weights or r >= num_weights:
            index_error("embedding_renorm_: index out of bounds")
        if r < 0:
            r += num_weights
        if not seen[r]:
            seen[r] = True
            p[unsafe_offset=count] = Int64(r)  # compacted in place
            count += 1
    var dev_rows = own(new_tensor(_vec(count), 1, ST_INT64, t.device))
    copy_from_host(t.device, ctx, dev_rows.t.ptr, host.t.ptr, count * 8)
    _ = host^
    if t.dim(1) > 0:
        var call = KernelCall("embedding_bag", "EmbeddingRenorm")
        call.arg_dtype(0, t.dtype)
        call.int(t.ptr)
        call.int(dev_rows.t.ptr)
        call.int(dev_rows.t.numel)
        call.int(t.dim(1))
        call.int(t.stride(0))
        call.int(t.stride(1))
        call.f64(max_norm)
        call.f64(norm_type)
        call.int(dtype_code(t.dtype))
        call.int(ctx_ptr(ctx))
        call.run()
    _ = dev_rows^
    _ = ctx
    ret_ref(rets, 0, t)


def register_embedding_bag(site: Site) raises:
    impl[op_embedding_bag, "_embedding_bag"](site)
    impl[op_embedding_bag, "_embedding_bag_forward_only"](site)
    impl[op_embedding_bag_backward, "_embedding_bag_backward"](site)
    impl[op_embedding_bag_dense_backward, "_embedding_bag_dense_backward"](site)
    impl[
        op_embedding_bag_per_sample_weights_backward,
        "_embedding_bag_per_sample_weights_backward",
    ](site)
    impl[op_embedding_renorm_, "embedding_renorm_"](site)
