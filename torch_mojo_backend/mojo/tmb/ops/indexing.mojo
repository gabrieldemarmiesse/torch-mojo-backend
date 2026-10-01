"""ATen ops: indexing group (see agents_docs/native_backend.md).

Reordering and index-driven data movement that needs no kernel of its own:
every op here is one or a few launches of kernels the data_movement and
memory families already have.

* flip / roll / channel_shuffle -- `CopyStrided` over a re-described
  source: negative strides (flip), 2^k shifted rectangles (roll), a swapped
  pair of axes (channel_shuffle). `rot90`, `fliplr`, `flipud` and
  `fft_fftshift` / `fft_ifftshift` are ATen composites over flip and roll.
* unfold -- a metadata-only view (the storage-sharing `tmb_as_strided`,
  like ATen's own kernel); `unfold_copy` is its composite clone.
* take / put_ -- the flattened tensor through `GatherDim` / `ScatterDim` /
  `ScatterAddDim`.
* index_fill_ / index_copy -- `ScatterDim` with the 1-D index broadcast
  over every other axis (value mode for fill).
* masked_scatter_ -- the mask's prefix sum (`cumsum`) is the source
  position of every selected element: `GatherDim` from source, then
  `WhereSelect`.
* repeat_interleave.Tensor -- cumsum, then `searchsorted(cumsum, arange,
  right=True)`. Its output length is data-dependent: like ATen's CUDA
  kernel it reads the total (and the negativity check) back to the host
  unless `output_size` is given.
* scatter(reduce=) / scatter.value(_out) / scatter_reduce / index_reduce --
  `ScatterAddDim` for sum and mean, `ScatterReduceDim` (compare-and-swap)
  for prod / amax / amin, `ScatterDim` for the `include_self=False`
  identity fill.
* nonzero.out / nonzero_static / index.Tensor_out / narrow_copy.out /
  fill_.Tensor -- the functional op through the dispatcher, written into
  the caller's tensor.

A "raw view" below is a `T` whose shape/strides/pointer were rewritten for
a kernel launch: it still carries its base's handle, so it is only ever
handed to kernels, never to the dispatcher.
"""
from std.utils import IndexList

from tmb.backend.abi import (
    IntList,
    Owned,
    ST_BOOL,
    ST_INT64,
    T,
    TAG_MEMORY_FORMAT,
    TAG_SCALAR_INT,
    Value,
    Values,
    bool_arg,
    cpu_empty,
    int_arg,
    new_like,
    new_tensor,
    none_arg,
    own,
    own_if_new,
    ret_owned,
    ret_ref,
    alert_not_deterministic,
    deterministic_algorithms,
    index_error,
    tensor_arg,
    unsupported,
    v_bool,
    v_f64,
    v_int,
    v_is_none,
    v_string,
    v_tensor,
    v_tensor_list,
    view_strided,
)
from tmb.backend.device import (
    copy_d2d,
    copy_from_host,
    copy_to_host,
    ctx_for,
)
from tmb.backend.registry import Site, impl
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.common import (
    assert_no_internal_overlap,
    assert_no_overlap,
    call_op,
    check_out,
    contiguous,
    copy_strided_into,
    cast_to,
    device_str,
    fill_value,
    is_int_stype,
    resize_out,
    same_view,
    scalar_to_float,
    scalar_to_int,
    shape_str,
)
from tmb.ops.compare import _where_select, scalar_as_fill
from tmb.ops.data_movement import (
    _dim_or1,
    _dims_of,
    _gather_dim_launch,
    _is_scatter_add_dtype,
    _is_scatter_dtype,
    _norm_dim,
    _scalar_type_name,
    _materialize_contiguous,
    _scatter_into,
    _scatter_launch,
    _scatter_validate,
    index_put_slice,
    scatter_add_sorted,
    scatter_scalar,
    _strides_of,
)
from tmb.ops.factories import _arange_fill


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _raw_view(
    base: T, shape: List[Int], strides: List[Int], elem_offset: Int
) -> T:
    """`base` re-described for a kernel: `shape`/`strides` (element units)
    starting `elem_offset` elements past its first element."""
    var v = base.copy()
    var rank = len(shape)
    var pad = MAX_RANK - rank
    v.rank = rank
    v.shape = IndexList[MAX_RANK](1)
    v.strides = IndexList[MAX_RANK](0)
    var n = 1
    for i in range(rank):
        v.shape[pad + i] = shape[i]
        v.strides[pad + i] = strides[i]
        n *= shape[i]
    v.numel = n
    v.ptr = base.ptr + elem_offset * base.itemsize
    v.offset = base.offset + elem_offset
    var expect = 1
    var contig = True
    for i in range(rank - 1, -1, -1):
        if shape[i] != 1 and strides[i] != expect:
            contig = False
        expect *= shape[i]
    v.contig = contig
    return v^


def _flat(t: T) -> T:
    """A contiguous `t` as a 1-D raw view of all its elements."""
    return _raw_view(t, [t.numel], [1], 0)


def _call1(
    name: StaticString, overload: StaticString, var args: List[Value]
) raises -> T:
    """One aten op through the dispatcher, its one Tensor result (owned)."""
    var rets = call_op(String(name), String(overload), args^, 1)
    return rets.take_tensor(0)


def _scalar_int(x: Int) -> Value:
    return Value(TAG_SCALAR_INT, 0, Int64(x), 0)


def _wrapped_index(index: T, n: Int) raises -> Owned:
    """A fresh contiguous int64 copy of `index` with negative entries
    wrapped by `n` (`idx < 0 ? idx + n : idx`, what ATen's take/put/
    index_fill kernels do); anything still out of range stays so, for the
    kernel's own bounds handling."""
    var neg = own(
        _call1("aten::lt", "Scalar", [tensor_arg(index), _scalar_int(0)])
    )
    var w = own(
        _call1(
            "aten::add",
            "Tensor",
            [tensor_arg(index), tensor_arg(neg.t), _scalar_int(n)],
        )
    )
    _ = neg^
    if w.t.contig:
        return w^
    var c = own(contiguous(w.t))
    _ = w^
    return c^


def _wrap_dim(dim: Int, rank: Int) raises -> Int:
    """`maybe_wrap_dim` (scalars wrap as rank 1)."""
    if rank == 0 and dim != 0 and dim != -1:
        raise Error(
            (
                "Dimension out of range (expected to be in range of [-1, 0],"
                " but got "
            ),
            dim,
            ")",
        )
    var r = max(rank, 1)
    var d = dim + r if dim < 0 else dim
    if d < 0 or d >= r:
        raise Error(
            "Dimension out of range (expected to be in range of [",
            -r,
            ", ",
            r - 1,
            "], but got ",
            dim,
            ")",
        )
    return d


def _read_int_at(t: T, element: Int) raises -> Int:
    """Element `element` of the contiguous 1-D integer tensor `t`, read back
    to the host (one sync)."""
    var one = own(
        view_strided(
            t,
            IndexList[MAX_RANK](1),
            IndexList[MAX_RANK](0),
            0,
            t.offset + element,
        )
    )
    var r = call_op("aten::_local_scalar_dense", "", [tensor_arg(one.t)], 1)
    _ = one^  # its handle was read by the call
    return v_int(r[0])


def _same_view(a: T, b: T) -> Bool:
    """`a` and `b` describe exactly the same elements (ATen's `is_same` up
    to the TensorImpl: same data pointer, shape and strides)."""
    if a.ptr != b.ptr or a.rank != b.rank:
        return False
    for i in range(a.rank):
        if a.dim(i) != b.dim(i) or a.stride(i) != b.stride(i):
            return False
    return True


def _same_device(a: T, b: T) -> Bool:
    return a.device_type == b.device_type and a.device == b.device


# ---------------------------------------------------------------------------
# flip -- ATen's `flip` (native/TensorTransformations.cpp): the source read
# backwards along every flipped dim, i.e. through a negative stride from its
# last element, into a fresh contiguous tensor.
# ---------------------------------------------------------------------------


def _flip_into(dst: T, src: T, flipped: List[Bool]) raises:
    var v = src.copy()
    var pad = MAX_RANK - src.rank
    for d in range(src.rank):
        if flipped[d] and src.dim(d) > 1:
            v.ptr += (src.dim(d) - 1) * src.stride(d) * src.itemsize
            v.strides[pad + d] = -src.stride(d)
    copy_strided_into(dst, v)


# aten::flip(Tensor self, int[] dims) -> Tensor
def op_flip(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var dims = IntList(args[unsafe_offset=1])
    var flipped = List[Bool](capacity=max(a.rank, 1))
    for _ in range(max(a.rank, 1)):
        flipped.append(False)
    for i in range(len(dims)):
        var d = _wrap_dim(dims[i], a.rank)
        if flipped[d]:
            raise Error(
                "dim ", d, " appears multiple times in the list of dims"
            )
        flipped[d] = True
    var out = own(new_like(a))
    if a.numel > 0:
        _flip_into(out.t, a, flipped)
    ret_owned(rets, 0, out)


# ---------------------------------------------------------------------------
# roll -- ATen's `roll_cuda` / `roll_common` (native/cuda/TensorTransformations.cu,
# native/TensorTransformations.h): no dims rolls the flattened tensor; one
# (shift, dim) pair per dim otherwise, a dim listed twice summing its
# shifts. Each rolled dim splits into two rectangles (out[s:] = in[:n-s],
# out[:s] = in[n-s:]), so k rolled dims are 2^k strided copies.
# ---------------------------------------------------------------------------


def _roll_into(dst: T, src: T, starts: List[Int]) raises:
    """dst = src rolled by `starts[d]` (already in [0, size)) along every
    dim d; dst and src have the same shape."""
    var rolled = List[Int]()
    for d in range(src.rank):
        if starts[d] != 0:
            rolled.append(d)
    var k = len(rolled)
    var pad = MAX_RANK - src.rank
    for mask in range(1 << k):
        var dv = dst.copy()
        var sv = src.copy()
        for j in range(k):
            var d = rolled[j]
            var n = src.dim(d)
            var s = starts[d]
            var length: Int
            var dst_off: Int
            var src_off: Int
            if (mask >> j) & 1 == 0:
                length = n - s
                dst_off = s
                src_off = 0
            else:
                length = s
                dst_off = 0
                src_off = n - s
            dv.shape[pad + d] = length
            sv.shape[pad + d] = length
            dv.ptr += dst_off * dst.stride(d) * dst.itemsize
            sv.ptr += src_off * src.stride(d) * src.itemsize
        var numel = 1
        for d in range(src.rank):
            numel *= dv.shape[pad + d]
        dv.numel = numel
        sv.numel = numel
        copy_strided_into(dv, sv)


# aten::roll(Tensor self, SymInt[1] shifts, int[1] dims=[]) -> Tensor
def op_roll(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var shifts = IntList(args[unsafe_offset=1])
    var dims = IntList(args[unsafe_offset=2])
    if len(shifts) == 0:
        raise Error("`shifts` required")
    if len(dims) == 0 and len(shifts) == 1:
        # Roll of the flattened tensor, reshaped back.
        var out = own(new_like(a))
        if a.numel > 0:
            var src = own_if_new(contiguous(a), a)
            _roll_into(_flat(out.t), _flat(src.t), [shifts[0] % a.numel])
            _ = src^  # alive past the launch
        ret_owned(rets, 0, out)
        return
    if len(shifts) != len(dims):
        raise Error(
            "shifts and dimensions must align. shifts: ",
            len(shifts),
            ", dims:",
            len(dims),
        )
    if a.rank == 0:
        raise Error(
            "Dimension specified as ", dims[0], " but tensor has no dimensions"
        )
    var totals = List[Int](capacity=a.rank)
    for _ in range(a.rank):
        totals.append(0)
    for i in range(len(dims)):
        var d = _wrap_dim(dims[i], a.rank)
        totals[d] += shifts[i]
    var out = own(new_like(a))
    if a.numel > 0:
        var starts = List[Int](capacity=a.rank)
        for d in range(a.rank):
            starts.append(totals[d] % a.dim(d))
        _roll_into(out.t, a, starts)
    ret_owned(rets, 0, out)


# ---------------------------------------------------------------------------
# unfold -- ATen's `unfold` (native/TensorShape.cpp): every window of `size`
# elements `step` apart along `dim` as a new trailing dim. A view: the result
# shares self's storage (`view_strided` is the storage-sharing
# `tmb_as_strided`), and ADInplaceOrView records it as one.
# ---------------------------------------------------------------------------


# aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)
def op_unfold(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var d = _wrap_dim(v_int(args[unsafe_offset=1]), a.rank)
    var size = v_int(args[unsafe_offset=2])
    var step = v_int(args[unsafe_offset=3])
    var max_size = 1 if a.rank == 0 else a.dim(d)
    if size > max_size:
        raise Error(
            "maximum size for tensor at dimension ",
            d,
            " is ",
            max_size,
            " but size is ",
            size,
        )
    if size < 0:
        raise Error("size is ", size, " but must be >= 0")
    if step <= 0:
        raise Error("step is ", step, " but must be > 0")
    var rank = a.rank + 1
    if rank > MAX_RANK:
        unsupported("unfold: the result rank exceeds the mojo device limit")
    var shape = IndexList[MAX_RANK](1)
    var strides = IndexList[MAX_RANK](0)
    var pad = MAX_RANK - rank
    for i in range(a.rank):
        shape[pad + i] = a.dim(i)
        strides[pad + i] = a.stride(i)
    shape[MAX_RANK - 1] = size
    strides[MAX_RANK - 1] = 1 if a.rank == 0 else a.stride(d)
    if a.rank > 0:
        shape[pad + d] = (a.dim(d) - size) // step + 1
        strides[pad + d] = a.stride(d) * step
    var out = own(view_strided(a, shape, strides, rank, a.offset))
    ret_owned(rets, 0, out)


# aten::unfold_backward(Tensor grad_in, SymInt[] input_sizes, int dim, int size, int step) -> Tensor
# ATen's `unfold_backward` (native/UnfoldBackward.cpp): every window's
# gradient summed back into the elements it read. Windows `g = ceil(size /
# step)` apart never overlap, so the windows split into `g` interleaved
# groups, each written through one strided view of the result: the first
# group is a copy into the zeroed result, the others accumulate (add_).
def op_unfold_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var sizes = IntList(args[unsafe_offset=1])
    var size = v_int(args[unsafe_offset=3])
    var step = v_int(args[unsafe_offset=4])
    var rank = len(sizes)
    if rank + 1 > MAX_RANK:
        unsupported(
            "unfold_backward: the gradient rank exceeds the device limit"
        )
    var shape = IndexList[MAX_RANK](1)
    for i in range(rank):
        shape[MAX_RANK - rank + i] = sizes[i]
    var out = own(new_tensor(shape, rank, grad.stype, grad.device))
    fill_value(out.t, 0.0)
    if rank == 0:
        if size == 1 and grad.numel == 1:
            copy_strided_into(out.t, _raw_view(grad, [], [], 0))
        ret_owned(rets, 0, out)
        return
    var d = _wrap_dim(v_int(args[unsafe_offset=2]), rank)
    var windows = grad.dim(d)
    if grad.numel == 0 or windows == 0:
        ret_owned(rets, 0, out)
        return
    var groups = 1 if step >= size else (size + step - 1) // step
    var od = out.t.stride(d)
    for r in range(min(groups, windows)):
        var count = (windows - r + groups - 1) // groups
        var vshape = List[Int](capacity=rank + 1)
        var ostrides = List[Int](capacity=rank + 1)
        var gstrides = List[Int](capacity=rank + 1)
        for i in range(rank):
            vshape.append(count if i == d else sizes[i])
            ostrides.append(od * step * groups if i == d else out.t.stride(i))
            gstrides.append(
                grad.stride(i) * groups if i == d else grad.stride(i)
            )
        vshape.append(size)
        ostrides.append(od)
        gstrides.append(grad.stride(rank))
        if r == 0:
            copy_strided_into(
                _raw_view(out.t, vshape, ostrides, 0),
                _raw_view(grad, vshape, gstrides, 0),
            )
            continue
        var oshape = IndexList[MAX_RANK](1)
        var ostr = IndexList[MAX_RANK](0)
        var gstr = IndexList[MAX_RANK](0)
        var pad = MAX_RANK - (rank + 1)
        for i in range(rank + 1):
            oshape[pad + i] = vshape[i]
            ostr[pad + i] = ostrides[i]
            gstr[pad + i] = gstrides[i]
        var ov = own(
            view_strided(
                out.t, oshape, ostr, rank + 1, out.t.offset + r * step * od
            )
        )
        var gv = own(
            view_strided(
                grad, oshape, gstr, rank + 1, grad.offset + r * grad.stride(d)
            )
        )
        _ = call_op(
            "aten::add_",
            "Tensor",
            [tensor_arg(ov.t), tensor_arg(gv.t), _scalar_int(1)],
            1,
        )
        _ = ov^  # both read by the add
        _ = gv^
    ret_owned(rets, 0, out)


# ---------------------------------------------------------------------------
# channel_shuffle -- ATen's `channel_shuffle` (native/ChanelShuffle.cpp):
# (N, C, *) viewed as (N, g, C/g, R), axes 1 and 2 swapped, back to (N, C, *).
# ---------------------------------------------------------------------------


# aten::channel_shuffle(Tensor self, SymInt groups) -> Tensor
def op_channel_shuffle(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var groups = v_int(args[unsafe_offset=1])
    if a.rank <= 2:
        var sizes = String("[")
        for i in range(a.rank):
            if i:
                sizes += ", "
            sizes += String(a.dim(i))
        raise Error(
            (
                "channel_shuffle expects input to have at least 3 dimensions,"
                " but got input with sizes "
            ),
            sizes + "]",
        )
    var c = a.dim(1)
    if groups <= 0:
        raise Error(
            "Number of groups to divide channels in must be positive.",
            " Value of groups:",
            groups,
        )
    if c % groups != 0:
        raise Error(
            "Number of channels must be divisible by groups. Got ",
            c,
            " channels and ",
            groups,
            " groups.",
        )
    var out = own(new_like(a))
    if a.numel > 0:
        var n = a.dim(0)
        var cg = c // groups
        var r = a.numel // (n * c)
        var src = own_if_new(contiguous(a), a)
        # out[n, i, j, r] = in[n, j, i, r] over (N, C/g, g, R).
        var sv = _raw_view(src.t, [n, cg, groups, r], [c * r, r, cg * r, 1], 0)
        var dv = _raw_view(
            out.t, [n, cg, groups, r], [c * r, groups * r, r, 1], 0
        )
        copy_strided_into(dv, sv)
        _ = src^  # alive past the launch
    ret_owned(rets, 0, out)


# ---------------------------------------------------------------------------
# take / put_ -- ATen's `take_out` / `put_` (native/TensorAdvancedIndexing.cpp)
# over the flattened tensor. Negative indices wrap once, as the CUDA kernels
# do. take gathers through GatherDim, which clamps a still out-of-range index
# exactly as index_select / gather / embedding do here (the family's policy:
# no bounds report, which would cost a device sync per call). Known gap:
# CUDA device-asserts and CPU raises IndexError on such an index. put_
# scatters through ScatterDim / ScatterAddDim, which skip it and raise
# afterwards.
# ---------------------------------------------------------------------------


def _take_checks(a: T, index: T, dest: T) raises:
    if index.dtype != DType.int64:
        raise Error(
            "take(): Expected a long tensor for index, but got ",
            _scalar_type_name(index.dtype),
        )
    if a.stype != dest.stype:
        raise Error(
            (
                "take(): self and out expected to have the same dtype, but got"
                " self.dtype = "
            ),
            _scalar_type_name(a.dtype),
            " and dest.dtype = ",
            _scalar_type_name(dest.dtype),
        )
    if not _same_device(a, dest) or not _same_device(a, index):
        raise Error(
            "take(): self, index and out expected to be in the same device"
        )
    if a.numel == 0 and index.numel != 0:
        raise Error("take(): tried to take from an empty tensor")


def _take_into(dst: T, a: T, index: T) raises:
    """dst (contiguous, index's shape) = a.flatten()[index]."""
    if index.numel == 0:
        return
    var idx = _wrapped_index(index, a.numel)
    var src = own_if_new(contiguous(a), a)
    _gather_dim_launch(
        _flat(dst),
        _flat(src.t),
        _flat(idx.t),
        [index.numel],
        [1],
        [1],
        [1],
        0,
        a.numel,
    )
    _ = src^  # alive past the launch
    _ = idx^


# aten::take(Tensor self, Tensor index) -> Tensor
def op_take(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var index = v_tensor(args[unsafe_offset=1])
    var out = own(new_tensor(index.shape, index.rank, a.stype, a.device))
    _take_checks(a, index, out.t)
    _take_into(out.t, a, index)
    ret_owned(rets, 0, out)


# aten::take.out(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
def op_take_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var index = v_tensor(args[unsafe_offset=1])
    var out = v_tensor(args[unsafe_offset=2])
    _take_checks(a, index, out)
    assert_no_internal_overlap(out)
    assert_no_overlap(out, index)
    assert_no_overlap(out, a)
    resize_out(out, index.shape, index.rank)
    # The resize may have moved a storage `out` shares with an input:
    # re-read their pointers before any kernel reads them.
    a = T(a.h)
    index = T(index.h)
    if out.contig:
        _take_into(out, a, index)
    else:
        var tmp = own(new_like(out))
        _take_into(tmp.t, a, index)
        copy_strided_into(out, tmp.t)
        _ = tmp^  # alive past the launch
    ret_ref(rets, 0, out)


# aten::put_(Tensor(a!) self, Tensor index, Tensor source, bool accumulate=False) -> Tensor(a!)
def op_put_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var index = v_tensor(args[unsafe_offset=1])
    var source = v_tensor(args[unsafe_offset=2])
    var accumulate = v_bool(args[unsafe_offset=3])
    # put_ (TensorAdvancedIndexing.cpp) alerts on a CUDA tensor whether it
    # accumulates (atomics) or not (duplicate indices race).
    alert_not_deterministic("put_")
    if index.dtype != DType.int64:
        raise Error(
            "put_(): Expected a long tensor for index, but got ",
            _scalar_type_name(index.dtype),
        )
    if a.stype != source.stype:
        raise Error(
            (
                "put_(): self and source expected to have the same dtype, but"
                " got self.dtype = "
            ),
            _scalar_type_name(a.dtype),
            " and source.dtype = ",
            _scalar_type_name(source.dtype),
        )
    if not _same_device(a, source) or not _same_device(a, index):
        raise Error(
            "put_(): self, index and source expected to be in the same device"
        )
    if source.numel != index.numel:
        raise Error(
            (
                "put_(): Expected source and index to have the same number of"
                " elements, but got source.numel() = "
            ),
            source.numel,
            ", index.numel() = ",
            index.numel,
        )
    if a.numel == 0 and index.numel != 0:
        raise Error("put_(): Tried to put elements into an empty tensor")
    assert_no_internal_overlap(a)
    assert_no_overlap(a, index)
    assert_no_overlap(a, source)
    if index.numel == 0:
        ret_ref(rets, 0, a)
        return
    if not _is_scatter_dtype(a.dtype):
        unsupported("put_ of dtype " + String(a.dtype))
    if accumulate and not _is_scatter_add_dtype(a.dtype):
        unsupported(
            "put_(accumulate=True) of dtype "
            + String(a.dtype)
            + " (no atomic add for it on the device)"
        )
    if accumulate and a.itemsize != 4 and a.dtype != DType.bool:
        var ctx = ctx_for(a.device)
        var metal = ctx.api() == "metal"
        _ = ctx
        if metal:
            # Apple GPUs have no 16- or 64-bit atomic add.
            unsupported(
                "put_(accumulate=True) of " + String(a.dtype) + " on Apple GPU"
            )
    var idx = _wrapped_index(index, a.numel)
    var src = own_if_new(contiguous(source), source)
    var tmp = own_if_new(contiguous(a), a)
    _scatter_launch(
        _flat(tmp.t),
        [1],
        idx.t,
        [1],
        src.t.ptr,
        src.t.dtype,
        [1],
        [index.numel],
        0,
        a.numel,
        False,
        0.0,
        accumulate,
        "put_",
    )
    if not a.contig:
        copy_strided_into(a, tmp.t)
    _ = tmp^  # alive past the launch
    _ = src^
    _ = idx^
    ret_ref(rets, 0, a)


# ---------------------------------------------------------------------------
# index_fill_ / index_copy -- ATen's `index_fill_` / `index_copy_out`
# (native/TensorAdvancedIndexing.cpp). ScatterDim over the index space
# `self.shape` with `dim` resized to the index length, the 1-D index read
# with stride 0 on every other axis. A 0-d self counts as shape (1,).
# ---------------------------------------------------------------------------


def _index_strides(rank: Int, dim: Int, stride: Int) -> List[Int]:
    var out = List[Int](capacity=rank)
    for d in range(rank):
        out.append(stride if d == dim else 0)
    return out^


def _index_fill(a: T, dim_in: Int, index: T, value_v: Value) raises:
    if index.dtype != DType.int64:
        raise Error("index_fill_(): Expected dtype int64 for index.")
    assert_no_overlap(a, index)
    var dim = _norm_dim(dim_in, a.rank, "index_fill_")
    if index.rank > 1:
        raise Error("Index has to be a vector/scalar")
    if not _same_device(a, index):
        unsupported("index_fill_ with the index on a different device")
    # The value converts (and is range-checked) before any early return, as
    # ATen's `source.to<scalar_t>()` does even for an empty fill.
    var value = scalar_as_fill(value_v, a.dtype)
    if index.numel == 0:
        return
    if a.rank > 0 and a.dim(dim) == 0:
        # Every index is out of bounds for an empty dimension.
        var idx0 = own_if_new(contiguous(index), index)
        var first = _read_int_at(_flat(idx0.t), 0)
        _ = idx0^
        raise Error(
            "index_fill_(): index ",
            first,
            " is out of bounds for dimension ",
            dim,
            " with size 0",
        )
    if a.numel == 0:
        return
    if a.rank > 4:
        unsupported("index_fill_ of rank greater than 4")
    if not _is_scatter_dtype(a.dtype):
        unsupported("index_fill_ of dtype " + String(a.dtype))
    var dim_size = 1 if a.rank == 0 else a.dim(dim)
    var idx = _wrapped_index(index, dim_size)
    var dims = _dims_of(a)
    dims[dim] = index.numel
    var rank = len(dims)
    _scatter_launch(
        a,
        _strides_of(a),
        idx.t,
        _index_strides(rank, dim, 1),
        a.ptr,
        a.dtype,
        _index_strides(rank, dim, 0),
        dims,
        dim,
        dim_size,
        True,
        value,
        False,
        "index_fill_",
    )
    _ = idx^


# aten::index_fill_.int_Scalar(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)
def op_index_fill__int_scalar(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    _index_fill(
        a,
        v_int(args[unsafe_offset=1]),
        v_tensor(args[unsafe_offset=2]),
        args[unsafe_offset=3].copy(),
    )
    ret_ref(rets, 0, a)


# aten::index_fill_.int_Tensor(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)
def op_index_fill__int_tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var value = v_tensor(args[unsafe_offset=3])
    if value.rank != 0:
        raise Error(
            (
                "index_fill_ only supports a 0-dimensional value tensor, but"
                " got tensor with "
            ),
            value.rank,
            " dimension(s).",
        )
    # `source.item()`: one read of the 0-d value (on the host when it is a
    # device tensor, like ATen).
    var item = call_op("aten::_local_scalar_dense", "", [tensor_arg(value)], 1)
    _index_fill(
        a,
        v_int(args[unsafe_offset=1]),
        v_tensor(args[unsafe_offset=2]),
        item[0],
    )
    ret_ref(rets, 0, a)


def _index_copy_check(a: T, dim_in: Int, index: T, source: T) raises -> Int:
    """ATen's `index_copy` meta function; returns the wrapped dim."""
    var dim = _norm_dim(dim_in, a.rank, "index_copy_")
    if index.rank >= 2:
        raise Error(
            "index_copy_(): Index should have dimension 1 or 0 (got ",
            index.rank,
            ")",
        )
    var num = index.numel
    if source.rank == 0 and num != 1:
        raise Error(
            (
                "index_copy_(): When source is scalar, index should have one"
                " element (got "
            ),
            num,
            ")",
        )
    elif source.rank != a.rank and source.rank != 0 and a.rank != 0:
        raise Error(
            (
                "index_copy_(): When source and destination are not scalars,"
                " their dimensionality must match. Source dimensionality ("
            ),
            source.rank,
            "), destination dimensionality (",
            a.rank,
            ")",
        )
    if index.dtype != DType.int64:
        raise Error(
            "index_copy_(): Expected a long tensor for index, but got ",
            _scalar_type_name(index.dtype),
        )
    if a.stype != source.stype:
        raise Error(
            (
                "index_copy_(): self and source expected to have the same"
                " dtype, but got (self) "
            ),
            _scalar_type_name(a.dtype),
            " and (source) ",
            _scalar_type_name(source.dtype),
        )
    if not _same_device(a, source) or not _same_device(a, index):
        raise Error(
            "index_copy_(): self, index and source expected to be in the same"
            " device"
        )
    # Source/destination slices (every dim but `dim`) must match.
    var same = a.rank == source.rank or a.rank == 0 or source.rank == 0
    var sa = List[Int]()
    var ss = List[Int]()
    for d in range(a.rank):
        if d != dim:
            sa.append(a.dim(d))
    for d in range(source.rank):
        if d != dim:
            ss.append(source.dim(d))
    if len(sa) != len(ss):
        same = False
    else:
        for i in range(len(sa)):
            if sa[i] != ss[i]:
                same = False
    if not same:
        var msg = String(
            "index_copy_(): Source/destination tensor must have same slice"
            " shapes. Destination slice shape: "
        )
        msg += _list_str(sa) + " at dimension " + String(dim)
        msg += " and source slice shape: " + _list_str(ss) + " at dimension 0."
        raise Error(msg)
    if source.rank != 0 and num != source.dim(dim):
        raise Error(
            "index_copy_(): Number of indices (",
            num,
            ") should be equal to source.size(dim) (",
            source.dim(dim),
            ")",
        )
    return dim


def _list_str(xs: List[Int]) -> String:
    var s = String("[")
    for i in range(len(xs)):
        if i:
            s += ", "
        s += String(xs[i])
    return s + "]"


def _index_copy_into(dest: T, dim: Int, index: T, source: T) raises:
    """Scatter `source` into `dest` (already holding self's values)."""
    if index.numel == 0:
        return
    if dest.rank > 0 and dest.dim(dim) == 0:
        # Every index is out of bounds for an empty dimension (CPU reports
        # the first one; read back only in this degenerate case).
        var idx = own_if_new(contiguous(index), index)
        var first = _read_int_at(_flat(idx.t), 0)
        _ = idx^
        raise Error(
            "index_copy_(): index ",
            first,
            " is out of bounds for dimension ",
            dim,
            " with size 0",
        )
    if dest.numel == 0:
        return
    if dest.rank > 4:
        unsupported("index_copy of rank greater than 4")
    if not _is_scatter_dtype(dest.dtype):
        unsupported("index_copy of dtype " + String(dest.dtype))
    var dim_size = 1 if dest.rank == 0 else dest.dim(dim)
    # The index space is source's shape. A 0-d self reads as shape (1,) and
    # takes one element per index (a 1-D source of index's length, so every
    # index is bounds-checked); a 0-d source is one element.
    var dims = _dims_of(source)
    var src_strides = _strides_of(source)
    var dst_strides = _strides_of(dest)
    if dest.rank == 0:
        dims = [index.numel]
        src_strides = [source.stride(0) if source.rank == 1 else 0]
        dst_strides = [0]
    elif source.rank == 0:
        dims = [1]
        src_strides = [0]
    var rank = len(dims)
    var idx = own_if_new(contiguous(index), index)
    _scatter_launch(
        dest,
        dst_strides,
        idx.t,
        _index_strides(rank, dim, 1 if index.rank == 1 else 0),
        source.ptr,
        source.dtype,
        src_strides,
        dims,
        dim,
        dim_size,
        False,
        0.0,
        False,
        "index_copy_",
    )
    _ = idx^


# aten::index_copy.out(Tensor self, int dim, Tensor index, Tensor source, *, Tensor(a!) out) -> Tensor(a!)
def op_index_copy_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var index = v_tensor(args[unsafe_offset=2])
    var source = v_tensor(args[unsafe_offset=3])
    var out = v_tensor(args[unsafe_offset=4])
    var dim = _index_copy_check(a, v_int(args[unsafe_offset=1]), index, source)
    check_out(out, a)
    # The structured meta's order: resize first, then the overlap checks. The
    # resize may move a storage `out` shares with an input, so every input
    # is re-read after it.
    resize_out(out, a.shape, a.rank)
    a = T(a.h)
    index = T(index.h)
    source = T(source.h)
    assert_no_internal_overlap(out)
    assert_no_overlap(out, index)
    assert_no_overlap(out, source)
    if not _same_view(out, a):
        # `result.copy_(self)`: copy_'s assert_no_partial_overlap -- the
        # identical view is fine, any other shared memory is not.
        assert_no_overlap(out, a)
        copy_strided_into(out, a)
    _index_copy_into(out, dim, index, source)
    ret_ref(rets, 0, out)


# aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
def op_index_copy(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var index = v_tensor(args[unsafe_offset=2])
    var source = v_tensor(args[unsafe_offset=3])
    var dim = _index_copy_check(a, v_int(args[unsafe_offset=1]), index, source)
    var out = own(new_like(a))
    copy_strided_into(out.t, a)
    _index_copy_into(out.t, dim, index, source)
    ret_owned(rets, 0, out)


# aten::index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
def op_index_copy_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var index = v_tensor(args[unsafe_offset=2])
    var source = v_tensor(args[unsafe_offset=3])
    var dim = _index_copy_check(a, v_int(args[unsafe_offset=1]), index, source)
    assert_no_internal_overlap(a)
    assert_no_overlap(a, index)
    assert_no_overlap(a, source)
    _index_copy_into(a, dim, index, source)
    ret_ref(rets, 0, a)


# ---------------------------------------------------------------------------
# masked_scatter_ -- ATen's `masked_scatter__cuda` (native/cuda/IndexKernel.cpp):
# the i-th selected element of self (row-major order) takes source's i-th
# element. The inclusive prefix sum of the broadcast mask, minus one, is that
# i at every position; GatherDim reads source there (clamped, so unselected
# positions read something harmless) and WhereSelect keeps self elsewhere.
# ---------------------------------------------------------------------------


# aten::masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)
def op_masked_scatter_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var mask = v_tensor(args[unsafe_offset=1])
    var source = v_tensor(args[unsafe_offset=2])
    assert_no_internal_overlap(a)
    if a.stype != source.stype:
        raise Error(
            (
                "masked_scatter_: expected self and source to have same dtypes"
                " but got "
            ),
            _scalar_type_name(a.dtype),
            " and ",
            _scalar_type_name(source.dtype),
        )
    if mask.dtype != DType.bool:
        raise Error(
            (
                "masked_scatter_ only supports boolean masks, but got mask with"
                " dtype "
            ),
            _scalar_type_name(mask.dtype),
        )
    if not _same_device(a, mask) or not _same_device(a, source):
        unsupported("masked_scatter_ with operands on different devices")
    # expand_inplace(self, mask): mask must broadcast to self's shape.
    if mask.rank > a.rank:
        raise Error(
            "masked_scatter_: mask of rank ",
            mask.rank,
            " cannot be expanded to self of rank ",
            a.rank,
        )
    var mask_strides = IndexList[MAX_RANK](0)
    for i in range(MAX_RANK):
        if mask.shape[i] == a.shape[i]:
            mask_strides[i] = mask.strides[i]
        elif mask.shape[i] == 1:
            mask_strides[i] = 0
        else:
            raise Error(
                "The expanded size of the tensor (",
                a.shape[i],
                ") must match the existing size (",
                mask.shape[i],
                ") at non-singleton dimension ",
                i - (MAX_RANK - a.rank),
                ".",
            )
    if a.numel == 0:
        ret_ref(rets, 0, a)
        return
    var n = a.numel
    # The broadcast mask, dense in self's logical order.
    var m = own(new_tensor(a.shape, a.rank, ST_BOOL, a.device))
    var mview = mask.copy()
    mview.rank = a.rank
    mview.shape = a.shape
    mview.strides = mask_strides
    mview.numel = n
    copy_strided_into(m.t, mview)
    var flat_shape = IndexList[MAX_RANK](1)
    flat_shape[MAX_RANK - 1] = n
    var flat_strides = IndexList[MAX_RANK](0)
    flat_strides[MAX_RANK - 1] = 1
    var m_flat = own(view_strided(m.t, flat_shape, flat_strides, 1, 0))
    var csum = own(
        _call1(
            "aten::cumsum", "", [tensor_arg(m_flat.t), int_arg(0), none_arg()]
        )
    )
    _ = m_flat^
    # The number of selected elements, read back (one sync): too few source
    # elements is an error, as on CPU (CUDA device-asserts).
    var count = _read_int_at(csum.t, n - 1)
    if count > source.numel:
        raise Error(
            "masked_scatter_: Number of elements of source < number of ones"
            " in mask"
        )
    if count == 0:
        ret_ref(rets, 0, a)
        return
    var pos = own(
        _call1(
            "aten::sub",
            "Scalar",
            [tensor_arg(csum.t), _scalar_int(1), _scalar_int(1)],
        )
    )
    _ = csum^
    var src = own_if_new(contiguous(source), source)
    var gathered = own(new_tensor(a.shape, a.rank, a.stype, a.device))
    _gather_dim_launch(
        _flat(gathered.t),
        _flat(src.t),
        _flat(pos.t),
        [n],
        [1],
        [1],
        [1],
        0,
        source.numel,
    )
    _ = src^  # alive past the launch
    _ = pos^
    # self = mask ? gathered : self, element for element over flat views.
    var dst = own_if_new(contiguous(a), a)
    var flat_cond = _flat(m.t)
    var flat_dst = _flat(dst.t)
    _where_select(
        flat_cond,
        flat_cond.strides,
        _flat(gathered.t),
        flat_dst.strides,
        flat_dst,
        flat_dst.strides,
        flat_dst,
    )
    if not a.contig:
        copy_strided_into(a, dst.t)
    _ = dst^  # alive past the launch
    _ = gathered^
    _ = m^
    ret_ref(rets, 0, a)


# ---------------------------------------------------------------------------
# repeat_interleave.Tensor -- ATen's `repeat_interleave_cuda`
# (native/cuda/Repeat.cu via native/Repeat.h `repeat_interleave_common`):
# output[j] = the i with cumsum[i-1] <= j < cumsum[i], i.e. the right
# insertion point of j in the inclusive cumsum. Without `output_size` the
# total and the `repeats >= 0` check are read back to the host, as upstream.
# ---------------------------------------------------------------------------


# aten::repeat_interleave.Tensor(Tensor repeats, *, SymInt? output_size=None) -> Tensor
def op_repeat_interleave_tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var repeats = v_tensor(args[unsafe_offset=0])
    if repeats.rank != 1:
        raise Error("repeat_interleave only accept 1D vector as repeat")
    if repeats.dtype != DType.int64 and repeats.dtype != DType.int32:
        raise Error("repeats has to be Long or Int tensor")
    if repeats.numel == 0:
        var empty = own(new_like(repeats))
        ret_owned(rets, 0, empty)
        return
    var csum = own(
        _call1(
            "aten::cumsum", "", [tensor_arg(repeats), int_arg(0), none_arg()]
        )
    )
    # The total and the negativity check are read back even with
    # `output_size` (two syncs): a wrong `output_size` or a negative repeat
    # raises instead of truncating (CUDA device-asserts the former).
    var total = _read_int_at(csum.t, repeats.numel - 1)
    var mn = own(_call1("aten::min", "", [tensor_arg(repeats)]))
    var r2 = call_op("aten::_local_scalar_dense", "", [tensor_arg(mn.t)], 1)
    _ = mn^  # its handle was read by the call
    if v_int(r2[0]) < 0:
        raise Error("repeats can not be negative")
    if not v_is_none(args[unsafe_offset=1]):
        var output_size = v_int(args[unsafe_offset=1])
        if output_size != total:
            raise Error(
                (
                    "Invalid input! In `repeat_interleave`, the `output_size`"
                    " argument ("
                ),
                output_size,
                (
                    ") must be the same as the sum of the elements in the"
                    " `repeats` tensor ("
                ),
                total,
                ").",
            )
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 1] = total
    if total <= 0:
        var empty = own(new_tensor(shape, 1, repeats.stype, repeats.device))
        ret_owned(rets, 0, empty)
        return
    var positions = own(new_tensor(shape, 1, ST_INT64, repeats.device))
    _arange_fill(positions.t, 0.0, 1.0)
    var out = own(
        _call1(
            "aten::searchsorted",
            "Tensor",
            [
                tensor_arg(csum.t),
                tensor_arg(positions.t),
                bool_arg(repeats.dtype == DType.int32),
                bool_arg(True),
                none_arg(),
                none_arg(),
            ],
        )
    )
    _ = positions^
    _ = csum^
    ret_owned(rets, 0, out)


# ---------------------------------------------------------------------------
# scatter(reduce=) / scatter.value / scatter_reduce / index_reduce -- ATen's
# `scatter_impl` + `scatter_reduce_two` and `index_reduce_func_cuda_impl`
# (native/TensorAdvancedIndexing.cpp, native/cuda/ScatterGatherKernel.cu,
# native/cuda/Indexing.cu). Sum and mean ride ScatterAddDim's atomic add;
# prod / amax / amin are ScatterReduceDim's compare-and-swap, which
# computes them as ATen's gpuAtomicMul / gpuAtomicMax / gpuAtomicMin do
# (in the dtype, NaN wins). `include_self=False` first writes the
# reduction's identity into every scattered-to slot, and mean divides by a
# count scattered the same way (floor division for integers), as upstream.
# Duplicate indices combine in an unspecified order, like CUDA's atomics: a
# float sum or mean can differ from CPU in the last bits.
# ---------------------------------------------------------------------------

comptime _RED_SUM = 1
comptime _RED_PROD = 2  # the values the kernel family's ROP_* take
comptime _RED_MAX = 3
comptime _RED_MIN = 4
comptime _RED_MEAN = 5


def _reduction(reduce: String, new_options: Bool) raises -> Int:
    """`get_operator_enum` (native/ReductionType.h)."""
    if not new_options:
        if reduce == "add":
            return _RED_SUM
        if reduce == "multiply":
            return _RED_PROD
        raise Error("reduce argument must be either add or multiply.")
    if reduce == "sum":
        return _RED_SUM
    if reduce == "prod":
        return _RED_PROD
    if reduce == "mean":
        return _RED_MEAN
    if reduce == "amax" or reduce == "max":
        return _RED_MAX
    if reduce == "amin" or reduce == "min":
        return _RED_MIN
    raise Error(
        "reduce argument must be either sum, prod, mean, amax or amin, got ",
        reduce,
    )


def _reduce_dtype_check(a: T, red: Int, what: String) raises:
    if not _is_scatter_add_dtype(a.dtype):
        unsupported(
            "aten::"
            + what
            + " of dtype "
            + String(a.dtype)
            + " (no atomic read-modify-write for it on the device)"
        )
    if red == _RED_MEAN and a.dtype == DType.bool:
        unsupported("aten::" + what + " mean of a bool tensor")


def _reduce_launch(
    target: T,
    dim_size: Int,
    idx: T,
    idx_strides: List[Int],
    dims: List[Int],
    dim: Int,
    src: Optional[T],
    value: Float64,
    red: Int,
    include_self: Bool,
    what: String,
    ordered: Bool = False,
    int_value: Optional[Int] = None,
    sorted_sums: Bool = False,
    slice_size: Int = 1,
) raises:
    """Reduce src (or the scalar `value`) into `target`, which holds self's
    values, over the index space `dims` (int64 `idx` read through
    `idx_strides`). The bad-index report of `_scatter_launch` applies.
    `ordered` reduces each slot in index order (`_scatter_launch`).

    `sorted_sums` (scatter_reduce / scatter), under deterministic
    algorithms: a floating sum, and a mean's sum and count, take the sorted
    route, which accumulates in the opmath type and stores the dtype once
    -- CUDA's `_scatter_via_index_put` for both (`count.scatter_add_`
    included). Otherwise both are atomic adds in the dtype, as CUDA's
    default kernels: a half sum and its count saturate together (1024
    bfloat16 ones sum and count to 256, mean 1)."""
    var src_ptr = target.ptr
    var src_dtype = target.dtype
    var src_strides = List[Int]()
    for _ in range(len(dims)):
        src_strides.append(0)
    if src:
        src_ptr = src.value().ptr
        src_dtype = src.value().dtype
        src_strides = _strides_of(src.value())
    var add = red == _RED_SUM or red == _RED_MEAN
    var rop = 0 if add else red
    if not include_self:
        _scatter_launch(
            target,
            _strides_of(target),
            idx,
            idx_strides,
            src_ptr,
            src_dtype,
            src_strides,
            dims,
            dim,
            dim_size,
            True,
            0.0,
            False,
            what,
            rop,
            not add,
        )
    var floating = target.dtype.is_floating_point()
    # CUDA's deterministic route only under deterministic algorithms (its
    # default is dtype atomics: a half sum and count saturate together).
    var by_sort = sorted_sums and floating and deterministic_algorithms()
    if add and src and by_sort and not ordered:
        # The sorted, ordered route of `_scatter_via_index_put`.
        scatter_add_sorted(
            target,
            idx,
            idx_strides,
            src.value(),
            src_strides,
            dims,
            dim,
            dim_size,
            what,
            slice_size,
        )
    else:
        _scatter_launch(
            target,
            _strides_of(target),
            idx,
            idx_strides,
            src_ptr,
            src_dtype,
            src_strides,
            dims,
            dim,
            dim_size,
            not src,
            value,
            add,
            what,
            rop,
            False,
            ordered,
            int_value,
        )
    if red != _RED_MEAN:
        return
    var counts = own(new_like(target))
    fill_value(counts.t, 1.0 if include_self else 0.0)
    if by_sort:
        var one = own(
            new_tensor(
                IndexList[MAX_RANK](1), 1, counts.t.stype, counts.t.device
            )
        )
        fill_value(one.t, 1.0)
        var zero_strides = List[Int]()
        for _ in range(len(dims)):
            zero_strides.append(0)
        scatter_add_sorted(
            counts.t,
            idx,
            idx_strides,
            one.t,
            zero_strides,
            dims,
            dim,
            dim_size,
            what,
            slice_size,
        )
        _ = one^
    else:
        _scatter_launch(
            counts.t,
            _strides_of(counts.t),
            idx,
            idx_strides,
            counts.t.ptr,
            counts.t.dtype,
            src_strides,
            dims,
            dim,
            dim_size,
            True,
            1.0,
            True,
            what,
        )
    # count.masked_fill_(count == 0, 1): a count is never negative.
    var safe = own(
        _call1("aten::clamp_min", "", [tensor_arg(counts.t), _scalar_int(1)])
    )
    var r = call_op(
        String("aten::div") if floating else String("aten::floor_divide"),
        String("Tensor") if floating else String(""),
        [tensor_arg(target), tensor_arg(safe.t)],
        1,
    )
    var quotient = own(r.take_tensor(0))
    copy_strided_into(target, quotient.t)
    _ = quotient^
    _ = safe^
    _ = counts^


def _scatter_reduce_into(
    target: T,
    a: T,
    dim: Int,
    index: T,
    src: Optional[T],
    value: Float64,
    ivalue: Optional[Int],
    red: Int,
    include_self: Bool,
    what: String,
) raises:
    """`scatter_impl` on a validated call: `target` holds self's values."""
    if index.numel == 0:
        return
    var idx64 = own_if_new(cast_to(index, ST_INT64), index)
    var idx_c = own_if_new(contiguous(idx64.t), idx64.t)
    _reduce_launch(
        target,
        _dim_or1(a, dim),
        idx_c.t,
        _strides_of(idx_c.t),
        _dims_of(idx_c.t),
        dim,
        src,
        value,
        red,
        include_self,
        what,
        int_value=ivalue,
        sorted_sums=True,
        slice_size=index_put_slice(target, index, dim),
    )
    _ = idx_c^
    _ = idx64^


def _write_result(mut dest: T, res: T, what: String) raises:
    """Hand a computed result to a caller's `out=`: torch's `resize_output`,
    then a copy. The result was computed before `out` was touched, so an
    `out` sharing storage with an input cannot corrupt the read."""
    resize_out(dest, res.shape, res.rank)
    if res.numel > 0:
        copy_strided_into(dest, res)


def _check_out_of(dest: T, a: T, index: T, src: Optional[T]) raises:
    """The `out=` checks of `scatter_meta_impl`, before anything is written."""
    check_out(dest, a)
    assert_no_internal_overlap(dest)
    assert_no_overlap(dest, index)
    if src:
        assert_no_overlap(dest, src.value())


def _scatter_reduce_op(
    args: Values,
    rets: Values,
    src_pos: Int,
    is_scalar: Bool,
    reduce_pos: Int,
    new_options: Bool,
    include_self_pos: Int,
    out_pos: Int,
    in_place: Bool,
    what: String,
) raises:
    """Every scatter / scatter_reduce overload with a reduce or a scalar:
    `reduce_pos` < 0 is a plain scatter.value, `out_pos` >= 0 an `out=`."""
    var a = v_tensor(args[unsafe_offset=0])
    var dim_in = v_int(args[unsafe_offset=1])
    var index = v_tensor(args[unsafe_offset=2])
    var src = Optional[T](None)
    var value = 0.0
    var ivalue = Optional[Int](None)
    if is_scalar:
        var sv = scatter_scalar(a, args[unsafe_offset=src_pos])
        value = sv[0]
        ivalue = sv[1]
    else:
        src = v_tensor(args[unsafe_offset=src_pos])
    var red = 0
    if reduce_pos >= 0:
        red = _reduction(v_string(args[unsafe_offset=reduce_pos]), new_options)
    var include_self = True
    if include_self_pos >= 0:
        include_self = v_bool(args[unsafe_offset=include_self_pos])
    var dim = _scatter_validate(a, dim_in, index, src, False)
    if red != 0:
        _reduce_dtype_check(a, red, what)
        # CUDA's alerts: prod always; the legacy add / multiply kernel
        # unless a floating add takes the deterministic index_put route
        # (which it does whenever deterministic algorithms are on).
        if not is_scalar:
            if red == _RED_PROD and new_options:
                alert_not_deterministic("scatter_reduce_cuda_prod_")
            elif not new_options and (
                red == _RED_PROD or not a.dtype.is_floating_point()
            ):
                alert_not_deterministic("scatter_reduce_cuda_kernel")
    if out_pos >= 0:
        var out = v_tensor(args[unsafe_offset=out_pos])
        _check_out_of(out, a, index, src)
        if same_view(out, a):
            _scatter_red_or_fill(
                out, a, dim, index, src, value, ivalue, red, include_self, what
            )
        else:
            var res = own(_materialize_contiguous(a))
            _scatter_red_or_fill(
                res.t,
                a,
                dim,
                index,
                src,
                value,
                ivalue,
                red,
                include_self,
                what,
            )
            _write_result(out, res.t, what)
            _ = res^
        ret_ref(rets, 0, out)
        return
    if in_place:
        # scatter_meta_impl's overlap checks: the output is self.
        assert_no_internal_overlap(a)
        assert_no_overlap(a, index)
        if src:
            assert_no_overlap(a, src.value())
        _scatter_red_or_fill(
            a, a, dim, index, src, value, ivalue, red, include_self, what
        )
        ret_ref(rets, 0, a)
        return
    var res = own(_materialize_contiguous(a))
    _scatter_red_or_fill(
        res.t, a, dim, index, src, value, ivalue, red, include_self, what
    )
    ret_owned(rets, 0, res)


def _scatter_red_or_fill(
    target: T,
    a: T,
    dim: Int,
    index: T,
    src: Optional[T],
    value: Float64,
    ivalue: Optional[Int],
    red: Int,
    include_self: Bool,
    what: String,
) raises:
    if red == 0:
        _scatter_into(
            target,
            dim,
            _dim_or1(a, dim),
            index,
            src,
            value,
            not src,
            False,
            ivalue,
        )
    else:
        _scatter_reduce_into(
            target, a, dim, index, src, value, ivalue, red, include_self, what
        )


# aten::scatter.value_out(Tensor self, int dim, Tensor index, Scalar value, *,
#   Tensor(a!) out) -> Tensor(a!)
def op_scatter_value_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _scatter_reduce_op(args, rets, 3, True, -1, False, -1, 4, False, "scatter")


# aten::scatter_.value(Tensor(a!) self, int dim, Tensor index, Scalar value)
#   -> Tensor(a!)
def op_scatter__value(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _scatter_reduce_op(args, rets, 3, True, -1, False, -1, -1, True, "scatter")


# aten::scatter.reduce(Tensor self, int dim, Tensor index, Tensor src, *,
#   str reduce) -> Tensor
def op_scatter_reduce(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _scatter_reduce_op(args, rets, 3, False, 4, False, -1, -1, False, "scatter")


# aten::scatter_.reduce(Tensor(a!) self, int dim, Tensor index, Tensor src, *,
#   str reduce) -> Tensor(a!)
def op_scatter__reduce(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _scatter_reduce_op(args, rets, 3, False, 4, False, -1, -1, True, "scatter")


# aten::scatter.reduce_out(Tensor self, int dim, Tensor index, Tensor src, *,
#   str reduce, Tensor(a!) out) -> Tensor(a!)
def op_scatter_reduce_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _scatter_reduce_op(args, rets, 3, False, 4, False, -1, 5, False, "scatter")


# aten::scatter.value_reduce(Tensor self, int dim, Tensor index, Scalar value,
#   *, str reduce) -> Tensor
def op_scatter_value_reduce(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _scatter_reduce_op(args, rets, 3, True, 4, False, -1, -1, False, "scatter")


# aten::scatter_.value_reduce(Tensor(a!) self, int dim, Tensor index,
#   Scalar value, *, str reduce) -> Tensor(a!)
def op_scatter__value_reduce(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _scatter_reduce_op(args, rets, 3, True, 4, False, -1, -1, True, "scatter")


# aten::scatter.value_reduce_out(Tensor self, int dim, Tensor index,
#   Scalar value, *, str reduce, Tensor(a!) out) -> Tensor(a!)
def op_scatter_value_reduce_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _scatter_reduce_op(args, rets, 3, True, 4, False, -1, 5, False, "scatter")


# aten::scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src,
#   str reduce, *, bool include_self=True) -> Tensor
def op_scatter_reduce_two(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _scatter_reduce_op(
        args, rets, 3, False, 4, True, 5, -1, False, "scatter_reduce"
    )


# aten::scatter_reduce_.two(Tensor(a!) self, int dim, Tensor index,
#   Tensor src, str reduce, *, bool include_self=True) -> Tensor(a!)
def op_scatter_reduce__two(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _scatter_reduce_op(
        args, rets, 3, False, 4, True, 5, -1, True, "scatter_reduce"
    )


# aten::scatter_reduce.two_out(Tensor self, int dim, Tensor index,
#   Tensor src, str reduce, *, bool include_self=True, Tensor(a!) out)
#   -> Tensor(a!)
def op_scatter_reduce_two_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _scatter_reduce_op(
        args, rets, 3, False, 4, True, 5, 6, False, "scatter_reduce"
    )


def _index_reduce_check(
    a: T, dim_in: Int, index: T, source: T, reduce: String
) raises -> Tuple[Int, Int]:
    """`index_func_meta_impl` + the reduce check of index_reduce's meta;
    returns (dim, reduction)."""
    if (
        reduce != "prod"
        and reduce != "mean"
        and reduce != "amax"
        and reduce != "amin"
    ):
        raise Error(
            (
                "index_reduce(): Expected reduce to be one of prod, mean, amax"
                " or amin but got "
            ),
            reduce,
            ".",
        )
    var dim = _norm_dim(dim_in, a.rank, "index_reduce")
    if index.rank > 1:
        index_error(
            "index_reduce_(): Index is supposed to be a vector, but got dim: "
            + String(index.rank)
        )
    if index.dtype != DType.int64 and index.dtype != DType.int32:
        raise Error(
            "index_reduce_(): Expected dtype int32/int64 for index but got: ",
            _scalar_type_name(index.dtype),
        )
    if source.dtype != a.dtype:
        raise Error(
            "index_reduce_(): self (",
            _scalar_type_name(a.dtype),
            ") and source (",
            _scalar_type_name(source.dtype),
            ") must have the same scalar type",
        )
    if not (dim == 0 or dim < source.rank):
        raise Error(
            "index_reduce_(): Indexing dim ",
            dim,
            " is out of bounds of the source tensor with dim ",
            source.rank,
        )
    var src_n = 1 if source.rank == 0 else source.dim(dim)
    if index.numel != src_n:
        raise Error(
            "index_reduce_(): Number of indices (",
            index.numel,
            ") should be equal to source.size(dim): (",
            src_n,
            "), for dim: ",
            dim,
        )
    var same = True
    if source.rank != 0 and a.rank != 0:
        if source.rank != a.rank:
            same = False
        else:
            for d in range(a.rank):
                if d != dim and source.dim(d) != a.dim(d):
                    same = False
    elif source.rank != a.rank:
        same = False
    if not same:
        raise Error(
            (
                "source tensor shape must match self tensor shape, excluding"
                " the specified dimension. Got self.shape = "
            ),
            shape_str(a),
            " source.shape = ",
            shape_str(source),
        )
    if a.rank > 4:
        unsupported("aten::index_reduce with rank greater than 4")
    if index.device != a.device or source.device != a.device:
        unsupported("aten::index_reduce with operands on different devices")
    var red = _reduction(reduce, True)
    _reduce_dtype_check(a, red, "index_reduce")
    if a.dtype == DType.bool:
        unsupported("aten::index_reduce of a bool tensor")
    var ctx = ctx_for(a.device)
    var metal = ctx.api() == "metal"
    _ = ctx
    if metal and a.dtype == DType.float64:
        unsupported("aten::index_reduce of float64 on Apple GPU")
    return (dim, red)


def _index_reduce_into(
    target: T,
    a: T,
    dim: Int,
    index: T,
    source: T,
    red: Int,
    include_self: Bool,
) raises:
    """`target[..., index[i], ...] (red)= source[..., i, ...]` along `dim`:
    the reduce launch over source's index space with the 1-D index
    broadcast (stride 0) across every other coordinate."""
    alert_not_deterministic("index_reduce_cuda")
    if index.numel == 0:
        return
    # CUDA's indexFuncSmallIndex (<= 16 indices) reduces every slot in index
    # order; above that it is atomic (indexFuncLargeIndex).
    var ordered = index.numel <= 16
    var idx = own_if_new(cast_to(index, ST_INT64), index)
    var idx_stride = idx.t.stride(0) if idx.t.rank == 1 else 0
    var idx_strides = List[Int](capacity=max(a.rank, 1))
    for d in range(max(a.rank, 1)):
        idx_strides.append(idx_stride if d == dim else 0)
    _reduce_launch(
        target,
        _dim_or1(a, dim),
        idx.t,
        idx_strides,
        _dims_of(source),
        dim,
        source.copy(),
        0.0,
        red,
        include_self,
        "index_reduce",
        ordered,
    )
    _ = idx^


def _index_reduce_op(
    args: Values, rets: Values, out_pos: Int, in_place: Bool
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var index = v_tensor(args[unsafe_offset=2])
    var source = v_tensor(args[unsafe_offset=3])
    var checked = _index_reduce_check(
        a,
        v_int(args[unsafe_offset=1]),
        index,
        source,
        v_string(args[unsafe_offset=4]),
    )
    var dim = checked[0]
    var red = checked[1]
    var include_self = v_bool(args[unsafe_offset=5])
    if out_pos >= 0:
        var out = v_tensor(args[unsafe_offset=out_pos])
        check_out(out, a)
        assert_no_internal_overlap(out)
        assert_no_overlap(out, index)
        assert_no_overlap(out, source)
        if same_view(out, a):
            _index_reduce_into(out, a, dim, index, source, red, include_self)
        else:
            var res = own(_materialize_contiguous(a))
            _index_reduce_into(res.t, a, dim, index, source, red, include_self)
            _write_result(out, res.t, "index_reduce")
            _ = res^
        ret_ref(rets, 0, out)
        return
    if in_place:
        # index_func_meta_impl's overlap checks: the output is self.
        assert_no_internal_overlap(a)
        assert_no_overlap(a, index)
        assert_no_overlap(a, source)
        _index_reduce_into(a, a, dim, index, source, red, include_self)
        ret_ref(rets, 0, a)
        return
    var res = own(_materialize_contiguous(a))
    _index_reduce_into(res.t, a, dim, index, source, red, include_self)
    ret_owned(rets, 0, res)


# aten::index_reduce(Tensor self, int dim, Tensor index, Tensor source,
#   str reduce, *, bool include_self=True) -> Tensor
def op_index_reduce(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _index_reduce_op(args, rets, -1, False)


# aten::index_reduce_(Tensor(a!) self, int dim, Tensor index, Tensor source,
#   str reduce, *, bool include_self=True) -> Tensor(a!)
def op_index_reduce_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _index_reduce_op(args, rets, -1, True)


# aten::index_reduce.out(Tensor self, int dim, Tensor index, Tensor source,
#   str reduce, *, bool include_self=True, Tensor(a!) out) -> Tensor(a!)
def op_index_reduce_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _index_reduce_op(args, rets, 6, False)


# ---------------------------------------------------------------------------
# nonzero.out / nonzero_static / index.Tensor_out / narrow_copy.out /
# fill_.Tensor -- the functional op through the dispatcher, then written
# into the caller's tensor (ATen's `nonzero_out_cuda`,
# `nonzero_static_cuda`, `index_out`, `narrow_copy_dense_cpu_out`,
# `fill_`). The result is complete before `out` is resized, so an `out`
# aliasing an input cannot corrupt the read.
# ---------------------------------------------------------------------------


def _nonzero_out_checks(t: T, dest: T, what: String) raises:
    if dest.dtype != DType.int64:
        if what == "nonzero_static":
            raise Error(
                (
                    "nonzero_static: Expected out tensor to have scalar type"
                    " Long but got "
                ),
                _scalar_type_name(dest.dtype),
            )
        raise Error(
            "Expected object of scalar type Long as out, but got ",
            _scalar_type_name(dest.dtype),
        )
    if dest.device_type != t.device_type or dest.device != t.device:
        raise Error(
            "expected self and out to be on the same device, but got out on ",
            device_str(dest),
            " and self on ",
            device_str(t),
        )


# aten::nonzero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_nonzero_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dest = v_tensor(args[unsafe_offset=1])
    _nonzero_out_checks(t, dest, "nonzero")
    var res = own(_call1("aten::nonzero", "", [tensor_arg(t)]))
    _write_result(dest, res.t, "nonzero")
    _ = res^
    ret_ref(rets, 0, dest)


def _nonzero_static(t: T, size: Int, fill: Int) raises -> Owned:
    """The first `size` rows of `nonzero(t)`, the rest `fill_value`."""
    if size < 0:
        raise Error("nonzero_static: 'size' must be an non-negative integer")
    var nz = own(_call1("aten::nonzero", "", [tensor_arg(t)]))
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 2] = size
    shape[MAX_RANK - 1] = t.rank
    var res = own(new_tensor(shape, 2, ST_INT64, t.device))
    if res.t.numel > 0:
        fill_value(res.t, _scalar_int(fill))
        var rows = min(nz.t.dim(0), size)
        if rows > 0:
            var ctx = ctx_for(t.device)
            copy_d2d(ctx, res.t.ptr, nz.t.ptr, rows * t.rank * 8)
            _ = ctx
    _ = nz^
    return res^


# aten::nonzero_static(Tensor self, *, SymInt size, int fill_value=-1)
#   -> Tensor
def op_nonzero_static(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var res = _nonzero_static(
        t, v_int(args[unsafe_offset=1]), v_int(args[unsafe_offset=2])
    )
    ret_owned(rets, 0, res)


# aten::nonzero_static.out(Tensor self, *, SymInt size, int fill_value=-1,
#   Tensor(a!) out) -> Tensor(a!)
def op_nonzero_static_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dest = v_tensor(args[unsafe_offset=3])
    _nonzero_out_checks(t, dest, "nonzero_static")
    var res = _nonzero_static(
        t, v_int(args[unsafe_offset=1]), v_int(args[unsafe_offset=2])
    )
    _write_result(dest, res.t, "nonzero_static")
    _ = res^
    ret_ref(rets, 0, dest)


# aten::index.Tensor_out(Tensor self, Tensor?[] indices, *,
#   Tensor(a!) out) -> Tensor(a!)
def op_index_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dest = v_tensor(args[unsafe_offset=2])
    check_out(dest, t)
    assert_no_internal_overlap(dest)
    assert_no_overlap(dest, t)
    for idx in v_tensor_list(args[unsafe_offset=1]):
        assert_no_overlap(dest, idx)
    var res = own(
        _call1(
            "aten::index",
            "Tensor",
            [tensor_arg(t), args[unsafe_offset=1].copy()],
        )
    )
    _write_result(dest, res.t, "index")
    _ = res^
    ret_ref(rets, 0, dest)


# aten::narrow_copy.out(Tensor self, int dim, SymInt start, SymInt length, *,
#   Tensor(a!) out) -> Tensor(a!)
def op_narrow_copy_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var dest = v_tensor(args[unsafe_offset=4])
    check_out(dest, t)
    var res = own(
        _call1(
            "aten::narrow_copy",
            "",
            [
                tensor_arg(t),
                args[unsafe_offset=1].copy(),
                args[unsafe_offset=2].copy(),
                args[unsafe_offset=3].copy(),
            ],
        )
    )
    _write_result(dest, res.t, "narrow_copy")
    _ = res^
    ret_ref(rets, 0, dest)


# aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)
def op_fill__tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var value = v_tensor(args[unsafe_offset=1])
    if value.rank != 0:
        raise Error(
            "fill_ only supports 0-dimension value tensor but got tensor with ",
            value.rank,
            " dimensions.",
        )
    # The value is read once, on the host (a device value costs one sync,
    # where CUDA copies on the stream), then filled with fill_.Scalar's
    # conversion rules. Reading it first also covers a value aliasing self.
    var r = call_op("aten::_local_scalar_dense", "", [tensor_arg(value)], 1)
    if value.on_mojo() and value.device == t.device:
        # A same-device value is a `copy_`: it converts without a range
        # check, but keeps copy_'s overlap check on self.
        assert_no_internal_overlap(t)
    else:
        # A value on another device (the CPU or another GPU) goes through
        # `fill_(Scalar)`, whose `Scalar::to<scalar_t>()` raises on overflow.
        if is_int_stype(t.stype):
            _ = scalar_to_int(r[0], t.stype)
        elif t.dtype.is_floating_point():
            _ = scalar_to_float(r[0], t.stype)
    fill_value(t, r[0])
    ret_ref(rets, 0, t)


def register_indexing(site: Site) raises:
    impl[op_flip, "flip"](site)
    impl[op_roll, "roll"](site)
    impl[op_unfold, "unfold"](site)
    impl[op_unfold_backward, "unfold_backward"](site)
    impl[op_channel_shuffle, "channel_shuffle"](site)
    impl[op_take, "take"](site)
    impl[op_take_out, "take.out"](site)
    impl[op_put_, "put_"](site)
    impl[op_index_fill__int_scalar, "index_fill_.int_Scalar"](site)
    impl[op_index_fill__int_tensor, "index_fill_.int_Tensor"](site)
    impl[op_index_copy, "index_copy"](site)
    impl[op_index_copy_, "index_copy_"](site)
    impl[op_index_copy_out, "index_copy.out"](site)
    impl[op_masked_scatter_, "masked_scatter_"](site)
    impl[op_repeat_interleave_tensor, "repeat_interleave.Tensor"](site)
    impl[op_scatter_value_out, "scatter.value_out"](site)
    impl[op_scatter__value, "scatter_.value"](site)
    impl[op_scatter_reduce, "scatter.reduce"](site)
    impl[op_scatter__reduce, "scatter_.reduce"](site)
    impl[op_scatter_reduce_out, "scatter.reduce_out"](site)
    impl[op_scatter_value_reduce, "scatter.value_reduce"](site)
    impl[op_scatter__value_reduce, "scatter_.value_reduce"](site)
    impl[op_scatter_value_reduce_out, "scatter.value_reduce_out"](site)
    impl[op_scatter_reduce_two, "scatter_reduce.two"](site)
    impl[op_scatter_reduce__two, "scatter_reduce_.two"](site)
    impl[op_scatter_reduce_two_out, "scatter_reduce.two_out"](site)
    impl[op_index_reduce, "index_reduce"](site)
    impl[op_index_reduce_, "index_reduce_"](site)
    impl[op_index_reduce_out, "index_reduce.out"](site)
    impl[op_nonzero_out, "nonzero.out"](site)
    impl[op_nonzero_static, "nonzero_static"](site)
    impl[op_nonzero_static_out, "nonzero_static.out"](site)
    impl[op_index_tensor_out, "index.Tensor_out"](site)
    impl[op_narrow_copy_out, "narrow_copy.out"](site)
    impl[op_fill__tensor, "fill_.Tensor"](site)
