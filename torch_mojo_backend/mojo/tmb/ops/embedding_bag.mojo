"""ATen ops: embedding_bag group (see agents_docs/native_backend.md).

* _embedding_bag / _embedding_bag_forward_only -- the `embedding_bag`
  family's `EmbeddingBagForward`, a port of ATen's
  EmbeddingBag_updateOutputKernel_sum_mean / _max (native/cuda/
  EmbeddingBag.cu): one thread per (bag, feature) walks the bag in order,
  so the output is deterministic and reduced in CUDA's order.
* _embedding_bag_backward -- ATen's generic `_embedding_bag_backward_symint`
  (native/EmbeddingBag.cpp): index promotion, offset2bag rebuilt when the
  forward did not return one, then the dense backward.
* _embedding_bag_dense_backward -- sum / mean: by default
  `EmbeddingBagBackwardAtomic` (atomics into an accumulator-dtype buffer,
  CUDA's fused atomic route); under deterministic algorithms a stable sort
  of the indices, their runs, then `EmbeddingBagBackwardSorted`, CUDA's
  chunked two-pass segment sum in CUDA's summation order; max
  scatter_adds into the rows max_indices names (atomic, alerting like
  CUDA's embedding_bag_backward_cuda_max).
* _embedding_bag_per_sample_weights_backward -- index_select / mul / sum:
  each product rounded to the dtype, summed in the accumulator, as CUDA.
* embedding_renorm_ -- the indices are wrapped and made unique on the
  device (the unique group's sort route, one read of the count), then
  the ends of the sorted unique rows are read to reject an index out of
  range before any row changes (CUDA device-asserts instead), then
  `EmbeddingRenorm` rescales each row in place.
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
    deterministic_algorithms,
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
from tmb.ops.common import call_op, cast_into, cast_to, contiguous, fill_value
from tmb.ops.data_movement import _scalar_type_name
from tmb.ops.unique import run_bounds, unique_flat

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


def _read_flags(flag: T) raises -> List[Bool]:
    """The int32 flags of a launch, read back (one blocking copy)."""
    var host = own(cpu_empty(flag.shape, 1, ST_INT32))
    var ctx = ctx_for(flag.device)
    copy_to_host(ctx, flag.ptr, host.t.ptr, flag.numel * 4)
    _ = ctx
    var p = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=host.t.ptr)
    var out = List[Bool]()
    for i in range(flag.numel):
        out.append(p[unsafe_offset=i] != 0)
    _ = host^
    return out^


def _read_int(t: T, i: Int) raises -> Int:
    """Element `i` of the contiguous int64 device tensor `t`."""
    var host = own(cpu_empty(_vec(1), 1, ST_INT64))
    var ctx = ctx_for(t.device)
    copy_to_host(ctx, t.ptr + i * 8, host.t.ptr, 8)
    _ = ctx
    var v = Int(
        Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=host.t.ptr)[]
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
        var flag = _zeros(_vec(4), 1, ST_INT32, device)
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
                offsets.dim(0),
            ]
        )
        call.int(dtype_code(weight.dtype))
        call.int(ctx_ptr(ctx))
        call.run()
        _ = ctx
        var bad = _read_flags(flag.t)
        _ = flag^
        # check_arguments' order (EmbeddingBag.cpp), then the kernel's.
        if bad[3]:
            raise Error(
                (
                    "offsets[0] has to be 0, i.e., the first sequence in the"
                    " mini-batch has to start from position 0. However, got "
                ),
                _read_int(offs.t, 0),
            )
        if bad[2]:
            raise Error(
                "offsets[-1] can not be greater than input's length ",
                num_indices,
                " but got offsets[-1] of ",
                _read_int(offs.t, offs.t.numel - 1),
            )
        if bad[0]:
            index_error(
                "Invalid input index in EmbeddingBag: index out of range [0, "
                + String(weight.dim(0))
                + ")"
            )
        if bad[1]:
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
    # Checked before the promotion casts, which would launder a float.
    _check_index_dtype(indices_in, "indices", 1, "embedding_bag")
    _check_index_dtype(offsets_in, "offsets", 1, "embedding_bag")
    var st = _index_stype(indices_in, offsets_in)
    var indices = own_if_new(cast_to(indices_in, st), indices_in)
    var offsets = own_if_new(cast_to(offsets_in, st), offsets_in)
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
    var gw = own(new_tensor(_mat(num_weights, features), 2, grad.stype, device))
    if indices.numel == 0 or gw.t.numel == 0:
        fill_value(gw.t, 0.0)
        ret_owned(rets, 0, gw)
        return
    var n = indices.numel
    var idx = _as_int64(indices)
    var o2b = _as_int64(offset2bag)
    var bs = _as_int64(bag_size)
    var g = own_if_new(contiguous(grad), grad)
    var psw_ptr = 0
    var psw_stride = 0
    if psw:
        if psw.value().dtype != grad.dtype:
            raise Error(
                "expected scalar type ",
                _scalar_type_name(grad.dtype),
                " but found ",
                _scalar_type_name(psw.value().dtype),
            )
        psw_ptr = psw.value().ptr
        psw_stride = psw.value().stride(0)
    var mean = 1 if mode == _MODE_MEAN else 0
    var ctx = ctx_for(device)
    if not deterministic_algorithms():
        # The default route: atomics into an accumulator-dtype buffer.
        var acc = _acc_stype(grad)
        var buf = _zeros(_mat(num_weights, features), 2, acc, device)
        var counts = Optional[Owned](None)
        if scale_grad_by_freq:
            var z = _zeros(_vec(num_weights), 1, ST_INT64, device)
            var ones = own(new_tensor(_vec(n), 1, ST_INT64, device))
            fill_value(ones.t, 1.0)
            counts = _op(
                "aten::index_add",
                "",
                [_t(z.t), _int(0), _t(idx.t), _t(ones.t), _scalar(1)],
            )
            _ = ones^
            _ = z^
        var call = KernelCall("embedding_bag", "EmbeddingBagBackwardAtomic")
        call.arg_dtype(0, grad.dtype)
        call.int(buf.t.ptr)
        call.int(g.t.ptr)
        call.int(idx.t.ptr)
        call.int(o2b.t.ptr)
        call.int(bs.t.ptr)
        call.int(psw_ptr)
        call.int(counts.value().t.ptr if counts else 0)
        call.tuple(
            [
                n,
                features,
                mean,
                1 if psw else 0,
                psw_stride,
                1 if scale_grad_by_freq else 0,
                padding_idx,
            ]
        )
        call.int(dtype_code(grad.dtype))
        call.int(ctx_ptr(ctx))
        call.run()
        _ = counts^
        if acc == grad.stype:
            _ = gw^
            ret_owned(rets, 0, buf)
        else:
            cast_into(gw.t, buf.t)
            _ = buf^
            ret_owned(rets, 0, gw)
        _ = g^
        _ = bs^
        _ = o2b^
        _ = idx^
        _ = ctx
        return
    # Deterministic: a stable sort of the indices (CUDA's
    # radix_sort_pairs), the runs of equal indices (the unique family), then
    # CUDA's chunked two-pass segment sum.
    fill_value(gw.t, 0.0)
    var r = call_op(
        "aten::sort",
        "stable",
        [
            _t(idx.t),
            Value(TAG_BOOL, 0, 1, 0),
            _int(0),
            Value(TAG_BOOL, 0, 0, 0),
        ],
        2,
    )
    var sorted = own(r.take_tensor(0))
    var perm = own(r.take_tensor(1))
    var runs = run_bounds(sorted.t)
    var partials = own(
        new_tensor(_mat(n, features), 2, _acc_stype(grad), device)
    )
    for phase in range(2):
        var call = KernelCall("embedding_bag", "EmbeddingBagBackwardSorted")
        call.arg_dtype(0, grad.dtype)
        call.tuple(
            [
                gw.t.ptr,
                partials.t.ptr,
                g.t.ptr,
                sorted.t.ptr,
                perm.t.ptr,
                runs.gid.t.ptr,
                runs.first.t.ptr,
                runs.last.t.ptr,
                o2b.t.ptr,
                bs.t.ptr,
                psw_ptr,
            ]
        )
        call.tuple(
            [
                n,
                features,
                mean,
                1 if psw else 0,
                psw_stride,
                1 if scale_grad_by_freq else 0,
                padding_idx,
                phase,
            ]
        )
        call.int(dtype_code(grad.dtype))
        call.int(ctx_ptr(ctx))
        call.run()
    _ = ctx
    _ = partials^
    _ = runs^
    _ = g^
    _ = bs^
    _ = o2b^
    _ = perm^
    _ = sorted^
    _ = idx^
    ret_owned(rets, 0, gw)


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
    # CUDA's kernel does `result += grad[..] * weight[..]` with scalar_t
    # operands: each product is rounded to the dtype (a half product can
    # overflow to inf), then summed in the accumulator.
    var gr = _op("aten::index_select", "", [_t(grad), _int(0), _t(offset2bag)])
    var wr = _op("aten::index_select", "", [_t(weight), _int(0), _t(indices)])
    var prod_dt = _op("aten::mul", "Tensor", [_t(gr.t), _t(wr.t)])
    var prod = own_if_new(cast_to(prod_dt.t, acc), prod_dt.t)
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
    _ = prod_dt^
    _ = wr^
    _ = gr^
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
    # Wrap the negative indices (embedding_renorm_wrap_indices_kernel),
    # then deduplicate on the device: two threads renorming one row would
    # race. An index outside [-W, W) stays outside [0, W) and is reported by
    # the kernel.
    var num_weights = t.dim(0)
    var idx = _as_int64(indices)
    var neg = _op("aten::lt", "Scalar", [_t(idx.t), _scalar(0)])
    var wrapped = _op(
        "aten::add",
        "Tensor",
        [_t(idx.t), _t(neg.t), _scalar(num_weights)],
    )
    _ = neg^
    _ = idx^
    var u = unique_flat(wrapped.t, False, False, False)
    _ = wrapped^
    var dev_rows = own(u[0].take())
    # Every index is validated before any row is touched: the unique rows
    # are sorted, so their ends are the extremes (two small reads).
    var m = dev_rows.t.numel
    if (
        _read_int(dev_rows.t, 0) < 0
        or _read_int(dev_rows.t, m - 1) >= num_weights
    ):
        index_error("embedding_renorm_: index out of bounds")
    var ctx = ctx_for(t.device)
    if t.dim(1) > 0:
        # The kernel's own bounds flag is a guard only: never set here.
        var flag = _zeros(_vec(1), 1, ST_INT32, t.device)
        var call = KernelCall("embedding_bag", "EmbeddingRenorm")
        call.arg_dtype(0, t.dtype)
        call.int(t.ptr)
        call.int(dev_rows.t.ptr)
        call.int(flag.t.ptr)
        call.int(m)
        call.int(num_weights)
        call.int(t.dim(1))
        call.int(t.stride(0))
        call.int(t.stride(1))
        call.f64(max_norm)
        call.f64(norm_type)
        call.int(dtype_code(t.dtype))
        call.int(ctx_ptr(ctx))
        call.run()
        _ = flag^
    _ = u^
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
