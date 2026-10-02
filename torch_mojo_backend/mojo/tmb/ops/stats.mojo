"""ATen ops: statistics that are neither a plain reduction nor a scan --
mode, histc, bincount, segment_reduce (and its backward), renorm, the
weight-norm interface (and its backward), the fused RMS-norm backward and
`_compute_linear_combination`.

mode, segment_reduce and the linear combination run on the stats kernel
family (tmb/kernels/stats/entry.mojo); everything else is composed from
registered ops through the dispatcher, in the order and the accumulation
dtype of the CUDA kernel it stands for (named next to each op).

histc and bincount need the data's extremes to size or bound the result,
exactly like CUDA's `_histc_cuda_template` / `_bincount_cuda_template`: one
`aminmax` read back to the host. Their counts are scattered with `index_add_`
(atomic adds, as CUDA's are), so float results raise in deterministic mode
the way CUDA's do.
"""
from std.utils import IndexList
from std.utils.numerics import isinf, isnan, max_or_inf, min_or_neg_inf

from tmb.backend.abi import (
    ST_BOOL,
    ST_FLOAT32,
    ST_FLOAT64,
    ST_INT64,
    TAG_DTYPE,
    TAG_INT_LIST,
    TAG_NONE,
    TAG_SCALAR_DOUBLE,
    TAG_SCALAR_INT,
    DoubleList,
    IntList,
    Owned,
    T,
    Value,
    Values,
    alert_not_deterministic,
    bool_arg,
    call_op,
    int_arg,
    contiguous_strides,
    dtype_code,
    f64_bits,
    index_error,
    max_dtype,
    new_scalar,
    new_tensor,
    none_arg,
    own,
    own_if_new,
    retain,
    ret_owned,
    ret_ref,
    ret_tensor_list,
    tensor_arg,
    unsupported,
    v_bool,
    v_bool_or,
    v_f64,
    v_int,
    v_int_or,
    v_is_none,
    v_scalar_is_integral,
    v_string,
    v_tensor,
    v_tensor_list,
    view_strided,
)
from tmb.backend.device import ctx_for, ctx_ptr, dev, read_bytes_sync
from tmb.backend.kernel_call import KernelCall
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.common import (
    assert_no_internal_overlap,
    cast_to,
    check_out,
    contiguous,
    copy_strided_into,
    fill_value,
    resize_out,
    scalar_to_float,
    scalar_to_int,
)
from tmb.ops.data_movement import _scalar_type_name
from tmb.ops.reductions import (
    _cast_any,
    _norm_dim,
    _order_stat_shape,
    _permuted_contiguous,
    _sort_kernel_dtype,
    _sort_rows_into,
    _zero_numel_check,
)
from tmb.backend.registry import Site, impl


# ---------------------------------------------------------------------------
# dispatcher helpers
# ---------------------------------------------------------------------------


def _call(op: String, overload: String, var args: List[Value]) raises -> Owned:
    """One aten op through the dispatcher, its single Tensor result owned."""
    var r = call_op(op, overload, args^, 1)
    return own(r.take_tensor(0))


def _t(t: T) -> Value:
    return tensor_arg(t)


def _dbl(v: Float64) -> Value:
    return Value(TAG_SCALAR_DOUBLE, 0, f64_bits(v), 0)


def _sint(v: Int) -> Value:
    return Value(TAG_SCALAR_INT, 0, Int64(v), 0)


def _dtype_v(st: Int32) -> Value:
    return Value(TAG_DTYPE, 0, Int64(st), 0)


def _ilist(values: List[Int64]) -> Value:
    """An `int[]` record over `values`' storage (keep `values` alive across
    the call)."""
    return Value(
        TAG_INT_LIST, Int32(len(values)), Int64(Int(values.unsafe_ptr())), 0
    )


def _cast(t: T, st: Int32) raises -> Owned:
    """`t` in dtype `st` as a handle the caller owns (a second reference when
    the dtype already matches)."""
    if t.stype == st:
        return own(T(retain(t)))
    return own(_cast_any(t, st))


def _mul(a: T, b: T) raises -> Owned:
    return _call("aten::mul", "Tensor", [_t(a), _t(b)])


def _sub(a: T, b: T) raises -> Owned:
    return _call("aten::sub", "Tensor", [_t(a), _t(b), _sint(1)])


def _reciprocal(a: T) raises -> Owned:
    return _call("aten::reciprocal", "", [_t(a)])


def _sum_dims(a: T, dims: List[Int64], keepdim: Bool) raises -> Owned:
    var r = _call(
        "aten::sum",
        "dim_IntList",
        [_t(a), _ilist(dims), bool_arg(keepdim), none_arg()],
    )
    _ = dims  # read by the call
    return r^


def _zeros(
    shape: IndexList[MAX_RANK], rank: Int, st: Int32, device: Int
) raises -> Owned:
    var z = own(new_tensor(shape, rank, st, device))
    fill_value(z.t, 0.0)
    return z^


def _shape1(n: Int) -> IndexList[MAX_RANK]:
    var s = IndexList[MAX_RANK](1)
    s[MAX_RANK - 1] = n
    return s


def _view(t: T, shape: IndexList[MAX_RANK], rank: Int) raises -> Owned:
    """A contiguous `t` viewed with another shape of the same element count."""
    return own(
        view_strided(t, shape, contiguous_strides(shape, rank), rank, t.offset)
    )


def _acc_stype(st: Int32) -> Int32:
    """CUDA's accumulate type of a floating dtype: float32 for the halves."""
    var dt: DType
    try:
        dt = max_dtype(st)
    except:
        return st
    if dt == DType.float16 or dt == DType.bfloat16:
        return ST_FLOAT32
    return st


def _decline_metal_f64(t: T, what: StaticString) raises:
    if t.dtype == DType.float64 and dev(t.device)[].api == "metal":
        unsupported(String(what) + ": float64 is unavailable on Apple GPUs")


def _read_scalar_f64(t: Owned) raises -> Float64:
    """The 0-d (or one-element contiguous) `t` read back to the host, as a
    float64 (exact for every floating dtype and every int up to 2**53).
    Borrows the owner, so the tensor outlives the read."""
    var r = call_op("aten::_local_scalar_dense", "", [_t(t.t)], 1)
    return v_f64(r[0])


def _read_scalar_int(t: Owned) raises -> Int:
    var r = call_op("aten::_local_scalar_dense", "", [_t(t.t)], 1)
    return v_int(r[0])


# ---------------------------------------------------------------------------
# mode (TensorModeKernel.cpp): the row sorted by the sort family, then the
# longest run (smallest value on a tie) and its largest index.
# ---------------------------------------------------------------------------


comptime MODE_FUSED_MAX_ROW = 2048


def _mode_into(a: T, dim: Int, values: T, indices: T) raises:
    """Along the normalized `dim` of `a` into the fresh contiguous `values`
    / `indices` (whose element order is the row order of `a` with `dim`
    moved last)."""
    var kdt = _sort_kernel_dtype(a, "mode")
    var n = a.dim(dim) if a.rank > 0 else 1
    if values.numel == 0:
        return
    var rows = a.numel // n
    if n == 1:
        # CUDA's slice-size-1 shortcut: the value itself at index 0.
        var src = own_if_new(contiguous(a), a)
        copy_strided_into(values, _view(src.t, values.shape, values.rank).t)
        _ = src^
        fill_value(indices, 0.0)
        return
    var last = a.rank <= 1 or dim == a.rank - 1
    var src = own_if_new(contiguous(a), a)
    if not last:
        var dims = List[Int]()
        dims.append(dim)
        src = own(_permuted_contiguous(a, dims))
    var sv = own(new_tensor(_shape1(rows * n), 1, a.stype, a.device))
    var si = own(new_tensor(_shape1(rows * n), 1, ST_INT64, a.device))
    _sort_rows_into(src.t, kdt, rows, n, n, False, False, sv.t, si.t)
    _ = src^  # alive past the sort
    var ctx = ctx_for(a.device)
    var call = KernelCall("stats", "ModeRuns")
    call.arg_dtype(0, kdt)
    call.int(values.ptr)
    call.int(indices.ptr)
    call.int(sv.t.ptr)
    call.int(si.t.ptr)
    call.int(rows)
    call.int(n)
    # TensorModeKernel.cpp: rows up to 2 * MAX_BLOCK_SIZE (1024 threads on
    # CUDA, 256 on ROCm) take the fused kernel (largest index of the mode),
    # longer ones the thrust fallback (smallest index).
    var fused_max = MODE_FUSED_MAX_ROW
    if dev(a.device)[].api == "hip":
        fused_max = 512
    call.int(1 if n > fused_max else 0)
    call.int(dtype_code(kdt))
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx
    _ = sv^  # alive past the launch
    _ = si^


def _mode_dim(a: T, dim_in: Int) raises -> Int:
    """`maybe_wrap_dim` plus `get_zero_numel_tensor_size`'s check."""
    if not a.on_mojo():
        raise Error("expected a tensor on the mojo device")
    var dim = _norm_dim(dim_in, a.rank)
    if a.numel == 0:
        _zero_numel_check(a, dim, "mode()")
    return dim


# aten::mode(Tensor self, int dim=-1, bool keepdim=False)
#   -> (Tensor values, Tensor indices)
def op_mode(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var dim = _mode_dim(a, v_int_or(args[unsafe_offset=1], -1))
    var keepdim = v_bool_or(args[unsafe_offset=2], False)
    var sr = _order_stat_shape(a, dim, keepdim)
    var values = own(new_tensor(sr[0], sr[1], a.stype, a.device))
    var indices = own(new_tensor(sr[0], sr[1], ST_INT64, a.device))
    _mode_into(a, dim, values.t, indices.t)
    ret_owned(rets, 0, values)
    ret_owned(rets, 1, indices)


# aten::mode.values(Tensor self, int dim=-1, bool keepdim=False, *,
#   Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
def op_mode_values(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var out_v = v_tensor(args[unsafe_offset=3])
    var out_i = v_tensor(args[unsafe_offset=4])
    var dim = _mode_dim(a, v_int_or(args[unsafe_offset=1], -1))
    if out_v.stype != a.stype:
        raise Error(
            "expected scalar type '",
            _scalar_type_name(a.dtype),
            "' but got '",
            _scalar_type_name(out_v.dtype),
            "' for values output",
        )
    if out_i.stype != ST_INT64:
        raise Error(
            "expected scalar type 'Long' but got '",
            _scalar_type_name(out_i.dtype),
            "' for indices output",
        )
    check_out(out_v, a)
    var keepdim = v_bool_or(args[unsafe_offset=2], False)
    var sr = _order_stat_shape(a, dim, keepdim)
    var values = own(new_tensor(sr[0], sr[1], a.stype, a.device))
    var indices = own(new_tensor(sr[0], sr[1], ST_INT64, a.device))
    _mode_into(a, dim, values.t, indices.t)
    if not out_v.same_shape(values.t):
        resize_out(out_v, sr[0], sr[1])
    if not out_i.same_shape(indices.t):
        resize_out(out_i, sr[0], sr[1])
    assert_no_internal_overlap(out_v)
    assert_no_internal_overlap(out_i)
    copy_strided_into(out_v, values.t)
    copy_strided_into(out_i, indices.t)
    _ = values^  # alive past the copies
    _ = indices^
    ret_ref(rets, 0, out_v)
    ret_ref(rets, 1, out_i)


# ---------------------------------------------------------------------------
# histc (SummaryOps.cu `_histc_cuda_template`) and bincount
# (`_bincount_cuda_template`)
# ---------------------------------------------------------------------------


def _fmt_bound(v: Float64) -> String:
    """A histc bound as `operator<<` prints a float: integral values without
    a fraction, `inf` / `-inf` / `nan` spelled out."""
    if isnan(v):
        return "nan"
    if isinf(v):
        return "-inf" if v < 0 else "inf"
    if v == Float64(Int(v)) and abs(v) < 1e15:
        return String(Int(v))
    return String(v)


def _aminmax_host(a: T) raises -> Tuple[Float64, Float64]:
    """`self.aminmax()` read back to the host."""
    var r = call_op(
        "aten::aminmax", "", [_t(a), none_arg(), bool_arg(False)], 2
    )
    var mn = own(r.take_tensor(0))
    var mx = own(r.take_tensor(1))
    return (_read_scalar_f64(mn), _read_scalar_f64(mx))


def _histc_counts_float(
    a: T, nbins: Int, lo: Float64, hi: Float64
) raises -> Owned:
    """CUDA's `getBin` and atomic counts as a composition: elements in
    [lo, hi] (NaN never) land in `int((x - lo) * nbins / (hi - lo))` in the
    input's float type, the top edge folded into the last bin."""
    var out = _zeros(_shape1(nbins), 1, a.stype, a.device)
    if a.numel == 0:
        return out^
    var flat = own_if_new(contiguous(a), a)
    var x = _view(flat.t, _shape1(a.numel), 1)
    var d = _call("aten::sub", "Scalar", [_t(x.t), _dbl(lo), _sint(1)])
    var p = _call("aten::mul", "Scalar", [_t(d.t), _sint(nbins)])
    var width: Float64
    if a.dtype == DType.float32:
        width = Float64(Float32(hi) - Float32(lo))
    else:
        width = hi - lo
    var q = _call("aten::div", "Scalar", [_t(p.t), _dbl(width)])
    var bins = _cast(q.t, ST_INT64)
    var clamped = _call(
        "aten::clamp", "", [_t(bins.t), _sint(0), _sint(nbins - 1)]
    )
    var ge = _call("aten::ge", "Scalar", [_t(x.t), _dbl(lo)])
    var le = _call("aten::le", "Scalar", [_t(x.t), _dbl(hi)])
    var inside = _call("aten::logical_and", "", [_t(ge.t), _t(le.t)])
    var weights = _cast(inside.t, a.stype)
    _ = call_op(
        "aten::index_add_",
        "",
        [_t(out.t), int_arg(0), _t(clamped.t), _t(weights.t), _sint(1)],
        1,
    )
    _ = flat^  # every temporary outlives the calls that read it
    _ = x^
    _ = d^
    _ = p^
    _ = q^
    _ = bins^
    _ = clamped^
    _ = ge^
    _ = le^
    _ = inside^
    _ = weights^
    return out^


def _histc_counts_int(a: T, nbins: Int, lo: Int, hi: Int) raises -> Owned:
    """The integer `getBin`: `(x - lo) * nbins / (hi - lo)` in int64 (CUDA's
    bounds_t), never through a float, so bounds past 2**53 stay exact. The
    counts accumulate in int64 and convert to the input dtype at the end
    (CUDA adds into the narrow dtype directly; the wrap is the same)."""
    var counts = _zeros(_shape1(nbins), 1, ST_INT64, a.device)
    if a.numel > 0:
        var flat = own_if_new(contiguous(a), a)
        var x = _view(flat.t, _shape1(a.numel), 1)
        var xi = _cast(x.t, ST_INT64)
        var d = _call("aten::sub", "Scalar", [_t(xi.t), _sint(lo), _sint(1)])
        var p = _call("aten::mul", "Scalar", [_t(d.t), _sint(nbins)])
        var bins = _call(
            "aten::floor_divide", "Scalar", [_t(p.t), _sint(hi - lo)]
        )
        var clamped = _call(
            "aten::clamp", "", [_t(bins.t), _sint(0), _sint(nbins - 1)]
        )
        # 0-d int64 bounds, not Scalars: the compare kernels embed a Scalar
        # through a double, which is inexact past 2**53.
        var lo_t = own(new_scalar(ST_INT64, a.device))
        fill_value(lo_t.t, _sint(lo))
        var hi_t = own(new_scalar(ST_INT64, a.device))
        fill_value(hi_t.t, _sint(hi))
        var ge = _call("aten::ge", "Tensor", [_t(xi.t), _t(lo_t.t)])
        var le = _call("aten::le", "Tensor", [_t(xi.t), _t(hi_t.t)])
        _ = lo_t^  # alive past the calls that read them
        _ = hi_t^
        var inside = _call("aten::logical_and", "", [_t(ge.t), _t(le.t)])
        var weights = _cast(inside.t, ST_INT64)
        _ = call_op(
            "aten::index_add_",
            "",
            [_t(counts.t), int_arg(0), _t(clamped.t), _t(weights.t), _sint(1)],
            1,
        )
        _ = flat^  # every temporary outlives the calls that read it
        _ = x^
        _ = xi^
        _ = d^
        _ = p^
        _ = bins^
        _ = clamped^
        _ = ge^
        _ = le^
        _ = inside^
        _ = weights^
    var out = _cast(counts.t, a.stype)
    _ = counts^
    return out^


def _histc(a: T, nbins: Int, min_v: Value, max_v: Value) raises -> Owned:
    if not a.on_mojo():
        raise Error("expected a tensor on the mojo device")
    if a.dtype == DType.float16:
        raise Error("HalfTensor is not supported")
    var dt = a.dtype
    if not (dt == DType.float32 or dt == DType.float64 or _is_index_int(dt)):
        raise Error(
            '"histc" not implemented for \'', _scalar_type_name(dt), "'"
        )
    _decline_metal_f64(a, "histc")
    if dt.is_floating_point():
        alert_not_deterministic("_histc_cuda with floating point input")
    if nbins <= 0:
        raise Error("bins must be > 0")
    if not dt.is_floating_point():
        # bounds_t is int64 for every integer input.
        var ilo = scalar_to_int(min_v, ST_INT64)
        var ihi = scalar_to_int(max_v, ST_INT64)
        if ilo == ihi and a.numel > 0:
            var r = call_op(
                "aten::aminmax", "", [_t(a), none_arg(), bool_arg(False)], 2
            )
            var mn = own(r.take_tensor(0))
            var mx = own(r.take_tensor(1))
            ilo = _read_scalar_int(mn)
            ihi = _read_scalar_int(mx)
        if ilo == ihi:
            ilo -= 1
            ihi += 1
        if not (ilo < ihi):
            raise Error("max must be larger than min")
        return _histc_counts_int(a, nbins, ilo, ihi)
    # bounds_t is float for float32 (Scalar::to<float> is checked).
    var lo = scalar_to_float(min_v, a.stype)
    var hi = scalar_to_float(max_v, a.stype)
    if dt == DType.float32:
        lo = Float64(Float32(lo))
        hi = Float64(Float32(hi))
    if lo == hi and a.numel > 0:
        var mm = _aminmax_host(a)
        lo = mm[0]
        hi = mm[1]
    if lo == hi:
        lo = lo - 1
        hi = hi + 1
        if dt == DType.float32:
            lo = Float64(Float32(lo))
            hi = Float64(Float32(hi))
    if isinf(lo) or isinf(hi) or isnan(lo) or isnan(hi):
        raise Error(
            "range of [",
            _fmt_bound(lo),
            ", ",
            _fmt_bound(hi),
            "] is not finite",
        )
    if not (lo < hi):
        raise Error("max must be larger than min")
    return _histc_counts_float(a, nbins, lo, hi)


# aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor
def op_histc(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var out = _histc(
        v_tensor(args[unsafe_offset=0]),
        v_int_or(args[unsafe_offset=1], 100),
        args[unsafe_offset=2],
        args[unsafe_offset=3],
    )
    ret_owned(rets, 0, out)


# aten::histc.out(Tensor self, int bins=100, Scalar min=0, Scalar max=0, *,
#   Tensor(a!) out) -> Tensor(a!)
def op_histc_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=4])
    var res = _histc(
        a,
        v_int_or(args[unsafe_offset=1], 100),
        args[unsafe_offset=2],
        args[unsafe_offset=3],
    )
    # `_histc_out_cuda`: resize_output + copy_ (which converts).
    if not out.same_shape(res.t):
        resize_out(out, res.t.shape, 1)
    assert_no_internal_overlap(out)
    var conv = _cast(res.t, out.stype)
    _ = res^  # alive past the cast
    copy_strided_into(out, conv.t)
    _ = conv^  # alive past the launch
    ret_ref(rets, 0, out)


def _is_index_int(dt: DType) -> Bool:
    return (
        dt == DType.uint8
        or dt == DType.int8
        or dt == DType.int16
        or dt == DType.int32
        or dt == DType.int64
    )


# aten::bincount(Tensor self, Tensor? weights=None, SymInt minlength=0)
#   -> Tensor
def op_bincount(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var has_w = not v_is_none(args[unsafe_offset=1])
    var minlength = v_int_or(args[unsafe_offset=2], 0)
    if not a.on_mojo():
        raise Error("expected a tensor on the mojo device")
    if has_w:
        alert_not_deterministic("_bincount_cuda")
    if not _is_index_int(a.dtype):
        raise Error(
            '"bincount_cuda" not implemented for \'',
            _scalar_type_name(a.dtype),
            "'",
        )
    if minlength < 0:
        raise Error("minlength should be >= 0")
    if a.rank == 1 and a.numel == 0:
        var z = _zeros(_shape1(minlength), 1, ST_INT64, a.device)
        ret_owned(rets, 0, z)
        return
    if a.rank != 1:
        raise Error("bincount only supports 1-d non-negative integral inputs.")
    var w_st = ST_INT64
    var w = Optional[T](None)
    if has_w:
        var wt = v_tensor(args[unsafe_offset=1])
        if wt.rank != 1 or wt.dim(0) != a.dim(0):
            raise Error(
                "weights should be 1-d and have the same length as input"
            )
        w_st = ST_FLOAT32 if wt.stype == ST_FLOAT32 else ST_FLOAT64
        w = wt.copy()
    var mm = _aminmax_host(a)
    if a.dtype != DType.uint8 and mm[0] < 0:
        raise Error("bincount only supports 1-d non-negative integral inputs.")
    var nbins = max(Int(mm[1]) + 1, minlength)
    var out = _zeros(_shape1(nbins), 1, w_st, a.device)
    var idx = _cast(a, ST_INT64)
    var src: Owned
    if has_w:
        if w_st == ST_FLOAT64:
            _decline_metal_f64(out.t, "bincount")
        src = _cast(w.value(), w_st)
    else:
        src = own(new_tensor(a.shape, 1, ST_INT64, a.device))
        fill_value(src.t, 1.0)
    _ = call_op(
        "aten::index_add_",
        "",
        [_t(out.t), int_arg(0), _t(idx.t), _t(src.t), _sint(1)],
        1,
    )
    _ = idx^
    _ = src^
    ret_owned(rets, 0, out)


# ---------------------------------------------------------------------------
# histogram / histogramdd (Histogram.cpp, cpu/HistogramKernel.cpp). CUDA has
# no kernel for these (CPU and MPS do); the semantics are the CPU kernel's:
# a coordinate lands in the bin `upper_bound(edges, x) - 1` (what its
# local-search linear route also returns), the rightmost edge is inclusive,
# anything outside [first, last] edge (and NaN) is skipped, and `density`
# divides by the total and by each dimension's bin widths.
# ---------------------------------------------------------------------------


def _hist_check_dtype(t: T) raises:
    if t.dtype != DType.float32 and t.dtype != DType.float64:
        unsupported("histogram of dtype " + String(t.dtype))
    _decline_metal_f64(t, "histogram")


def _rows_2d(t: T) raises -> Owned:
    """`t` reshaped to (M, N) with N its innermost size, contiguous."""
    var dense = own_if_new(contiguous(t), t)
    var n = t.dim(t.rank - 1)
    var m = t.numel // n if n > 0 else 0
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 2] = m
    shape[MAX_RANK - 1] = n
    var r = _view(dense.t, shape, 2)
    _ = dense^
    return r^


def _column(x2d: T, d: Int) raises -> Owned:
    """Column `d` of the contiguous (M, N) `x2d` as a strided 1-d view."""
    var strides = IndexList[MAX_RANK](0)
    strides[MAX_RANK - 1] = x2d.dim(1)
    return own(
        view_strided(x2d, _shape1(x2d.dim(0)), strides, 1, x2d.offset + d)
    )


def _element(t: T, k: Int) raises -> Owned:
    """Element `k` of the contiguous 1-d `t` as a 0-d view."""
    return own(
        view_strided(
            t, IndexList[MAX_RANK](1), IndexList[MAX_RANK](0), 0, t.offset + k
        )
    )


def _hist_outer_edges(x2d: T, range_v: Value) raises -> List[Float64]:
    """`select_outer_bin_edges`: the range if given, else the data's extremes
    per dimension ([0, 1] for empty input); a degenerate range widens by
    0.5 each way. Returned as (left_0, right_0, left_1, ...)."""
    var n = x2d.dim(1)
    var left = List[Float64]()
    var right = List[Float64]()
    for _ in range(n):
        left.append(0.0)
        right.append(1.0)
    if range_v.tag != TAG_NONE:
        var r = DoubleList(range_v)
        if len(r) != 2 * n:
            raise Error(
                "torch.histogramdd: for a ",
                n,
                "-dimensional histogram range should have ",
                2 * n,
                " elements, but got ",
                len(r),
            )
        for d in range(n):
            left[d] = r[2 * d]
            right[d] = r[2 * d + 1]
    elif x2d.numel > 0:
        var res = call_op(
            "aten::aminmax", "", [_t(x2d), int_arg(0), bool_arg(False)], 2
        )
        var mn = own(res.take_tensor(0))
        var mx = own(res.take_tensor(1))
        for d in range(n):
            left[d] = _read_scalar_f64(_element(mn.t, d))
            right[d] = _read_scalar_f64(_element(mx.t, d))
        _ = mn^  # alive past the reads
        _ = mx^
    for d in range(n):
        if (
            isinf(left[d])
            or isnan(left[d])
            or isinf(right[d])
            or isnan(right[d])
        ):
            raise Error(
                "torch.histogramdd: dimension ",
                d,
                "'s range [",
                _fmt_bound(left[d]),
                ", ",
                _fmt_bound(right[d]),
                "] is not finite",
            )
        if not (left[d] <= right[d]):
            raise Error(
                "torch.histogramdd: min should not exceed max, but got min ",
                _fmt_bound(left[d]),
                " max ",
                _fmt_bound(right[d]),
                " for dimension ",
                d,
            )
        if left[d] == right[d]:
            left[d] -= 0.5
            right[d] += 0.5
    var pairs = List[Float64]()
    for d in range(n):
        pairs.append(left[d])
        pairs.append(right[d])
    return pairs^


def _linspace_edges(
    lo: Float64, hi: Float64, bins: Int, like: T
) raises -> Owned:
    var e = own(new_tensor(_shape1(bins + 1), 1, like.stype, like.device))
    _ = call_op(
        "aten::linspace",
        "out",
        [_dbl(lo), _dbl(hi), int_arg(bins + 1), _t(e.t)],
        1,
    )
    return e^


def _hist_edges_from_counts(
    x2d: T, bins: List[Int], range_v: Value
) raises -> List[Owned]:
    """`histogramdd_bin_edges`: linspace edges per dimension."""
    var outer = _hist_outer_edges(x2d, range_v)
    if len(bins) != x2d.dim(1):
        raise Error(
            "histogramdd: The size of bins must be equal to the innermost"
            " dimension of the input."
        )
    var edges = List[Owned]()
    for d in range(len(bins)):
        edges.append(
            _linspace_edges(outer[2 * d], outer[2 * d + 1], bins[d], x2d)
        )
    return edges^


def _hist_check_inputs(x: T, edges: List[T], weight: Optional[T]) raises:
    """`histogramdd_check_inputs`."""
    if x.rank < 2:
        raise Error(
            (
                "torch.histogramdd: input tensor should have at least 2"
                " dimensions, but got "
            ),
            x.rank,
        )
    var n = x.dim(x.rank - 1)
    if len(edges) != n:
        raise Error(
            "torch.histogramdd: expected ",
            n,
            " sequences of bin edges for a ",
            n,
            "-dimensional histogram but got ",
            len(edges),
        )
    for d in range(n):
        if edges[d].stype != x.stype:
            raise Error(
                (
                    "torch.histogramdd: input tensor and bins tensors should"
                    " have the same dtype, but got input with dtype "
                ),
                _scalar_type_name(x.dtype),
                " and bins for dimension ",
                d,
                " with dtype ",
                _scalar_type_name(edges[d].dtype),
            )
        if edges[d].rank != 1:
            raise Error(
                (
                    "torch.histogramdd: bins tensor should have one dimension,"
                    " but got "
                ),
                edges[d].rank,
                " dimensions in the bins tensor for dimension ",
                d,
            )
        if edges[d].numel <= 0:
            raise Error(
                (
                    "torch.histogramdd: bins tensor should have at least 1"
                    " element, but got "
                ),
                edges[d].numel,
                " elements in the bins tensor for dimension ",
                d,
            )
    if weight:
        var w = weight.value().copy()
        if w.stype != x.stype:
            raise Error(
                "torch.histogramdd: if weight tensor is provided, input tensor"
                " and weight tensor should have the same dtype"
            )
        var ok = (w.rank == x.rank - 1) or (
            w.rank == 0 and x.rank == 2 and x.dim(0) == 1
        )
        if ok and w.rank == x.rank - 1:
            for i in range(w.rank):
                if w.dim(i) != x.dim(i):
                    ok = False
        if not ok:
            raise Error(
                "torch.histogramdd: if weight tensor is provided it should have"
                " the same shape as the input tensor excluding its innermost"
                " dimension"
            )


def _hist_counts(
    x2d: T, edges: List[T], weight: Optional[T], density: Bool
) raises -> Owned:
    """The (bins_0, ..., bins_{N-1}) histogram of the (M, N) coordinates."""
    var m = x2d.dim(0)
    var n = x2d.dim(1)
    var shape = IndexList[MAX_RANK](1)
    var total = 1
    for d in range(n):
        var nb = edges[d].numel - 1
        if nb <= 0:
            raise Error(
                "torch.histogram(): bins must be > 0, but got ",
                nb,
                " for dimension ",
                d,
            )
        shape[MAX_RANK - n + d] = nb
        total *= nb
    var hist = _zeros(_shape1(total), 1, x2d.stype, x2d.device)
    if m > 0 and n > 0:
        var flat = _zeros(_shape1(m), 1, ST_INT64, x2d.device)
        var keep = own(new_tensor(_shape1(m), 1, ST_BOOL, x2d.device))
        fill_value(keep.t, 1.0)
        var stride = total
        for d in range(n):
            var nb = edges[d].numel - 1
            stride //= nb
            var e = own_if_new(contiguous(edges[d]), edges[d])
            var col = _column(x2d, d)
            var ub = _call(
                "aten::searchsorted",
                "Tensor",
                [
                    _t(e.t),
                    _t(col.t),
                    bool_arg(False),
                    bool_arg(True),
                    none_arg(),
                    none_arg(),
                ],
            )
            var pos = _call("aten::clamp", "", [_t(ub.t), _sint(1), _sint(nb)])
            var term = _call("aten::mul", "Scalar", [_t(pos.t), _sint(stride)])
            # bin = clamp(upper_bound, 1, nb) - 1, scaled by the dim's stride
            var shifted = _call(
                "aten::sub", "Scalar", [_t(term.t), _sint(stride), _sint(1)]
            )
            var nflat = _call(
                "aten::add", "Tensor", [_t(flat.t), _t(shifted.t), _sint(1)]
            )
            # The old value must outlive the call that reads it: the
            # reassignment below is not a use, so without this it would be
            # released before the call ran.
            _ = flat^
            var first = _element(e.t, 0)
            var last = _element(e.t, nb)
            var ge = _call("aten::ge", "Tensor", [_t(col.t), _t(first.t)])
            var le = _call("aten::le", "Tensor", [_t(col.t), _t(last.t)])
            var ok = _call("aten::logical_and", "", [_t(ge.t), _t(le.t)])
            var nkeep = _call("aten::logical_and", "", [_t(keep.t), _t(ok.t)])
            _ = keep^
            flat = nflat^
            keep = nkeep^
            _ = e^  # every temporary outlives the calls that read it
            _ = col^
            _ = ub^
            _ = pos^
            _ = term^
            _ = shifted^
            _ = first^
            _ = last^
            _ = ge^
            _ = le^
            _ = ok^
        var w: Owned
        if weight:
            var wd = own_if_new(contiguous(weight.value()), weight.value())
            w = _view(wd.t, _shape1(m), 1)
            _ = wd^
        else:
            w = own(new_tensor(_shape1(m), 1, x2d.stype, x2d.device))
            fill_value(w.t, 1.0)
        var wk = _call(
            "aten::where", "ScalarOther", [_t(keep.t), _t(w.t), _dbl(0.0)]
        )
        _ = call_op(
            "aten::index_add_",
            "",
            [_t(hist.t), int_arg(0), _t(flat.t), _t(wk.t), _sint(1)],
            1,
        )
        _ = flat^
        _ = keep^
        _ = w^
        _ = wk^
    if not density:
        var out = _view(hist.t, shape, n)
        _ = hist^
        return out^
    # density, on the flat histogram (elementwise broadcasting stops at rank
    # 4, a histogramdd can have more dims): / total, then / each dim's bin
    # widths expanded to the full shape.
    var total_w = _call("aten::sum", "", [_t(hist.t), none_arg()])
    var normed = _call("aten::div", "Tensor", [_t(hist.t), _t(total_w.t)])
    _ = total_w^  # alive past the call that reads it
    _ = hist^
    for d in range(n):
        var nb = edges[d].numel - 1
        var e = own_if_new(contiguous(edges[d]), edges[d])
        var hi = own(
            view_strided(e.t, _shape1(nb), e.t.strides, 1, e.t.offset + 1)
        )
        var lo = own(view_strided(e.t, _shape1(nb), e.t.strides, 1, e.t.offset))
        var widths = _sub(hi.t, lo.t)
        var wstrides = IndexList[MAX_RANK](0)
        wstrides[MAX_RANK - n + d] = 1
        var wexp = own(
            view_strided(widths.t, shape, wstrides, n, widths.t.offset)
        )
        var wfull = own(new_tensor(shape, n, widths.t.stype, widths.t.device))
        copy_strided_into(wfull.t, wexp.t)
        var wflat = _view(wfull.t, _shape1(total), 1)
        var nn = _call("aten::div", "Tensor", [_t(normed.t), _t(wflat.t)])
        _ = normed^  # alive past the call that reads it
        normed = nn^
        _ = e^  # every temporary outlives the calls that read it
        _ = hi^
        _ = lo^
        _ = widths^
        _ = wexp^
        _ = wfull^
        _ = wflat^
    var out = _view(normed.t, shape, n)
    _ = normed^
    return out^


def _owned_ts(edges: List[Owned]) -> List[T]:
    var ts = List[T]()
    for i in range(len(edges)):
        ts.append(edges[i].t.copy())
    return ts^


def _flat_1d(t: T) raises -> Owned:
    """`t.reshape({numel, 1})` (histogram's view of its input)."""
    var dense = own_if_new(contiguous(t), t)
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 2] = t.numel
    var r = _view(dense.t, shape, 2)
    _ = dense^
    return r^


def _opt_weight(v: Value) raises -> Optional[T]:
    if v_is_none(v):
        return None
    return v_tensor(v)


def _hist_flat_weight(w_v: Value) raises -> Tuple[Optional[Owned], Optional[T]]:
    """histogram's `weight.reshape({numel})`: the owner and its view."""
    var weight = _opt_weight(w_v)
    if not weight:
        return (Optional[Owned](None), Optional[T](None))
    var w = weight.value().copy()
    var wd = own_if_new(contiguous(w), w)
    var wv = _view(wd.t, _shape1(w.numel), 1)
    _ = wd^
    var t = wv.t.copy()
    return (Optional[Owned](wv^), Optional[T](t^))


def _histogram_bins_tensor(
    a: T, bins: T, w_v: Value, density: Bool
) raises -> Tuple[Owned, Owned]:
    _hist_check_dtype(a)
    var x2d = _flat_1d(a)
    var w = _hist_flat_weight(w_v)
    var edges = List[T]()
    edges.append(bins.copy())
    _hist_check_inputs(x2d.t, edges, w[1])
    var hist = _hist_counts(x2d.t, edges, w[1], density)
    var be = own(new_tensor(bins.shape, bins.rank, bins.stype, bins.device))
    copy_strided_into(be.t, bins)
    _ = x2d^  # alive past the launches
    _ = w^
    return (hist^, be^)


def _histogram_bin_ct(
    a: T, bins: Int, range_v: Value, w_v: Value, density: Bool
) raises -> Tuple[Owned, Owned]:
    _hist_check_dtype(a)
    if bins <= 0:
        raise Error(
            "torch.histogram(): bins must be > 0, but got ",
            bins,
            " for dimension 0",
        )
    var x2d = _flat_1d(a)
    var outer = _hist_outer_edges(x2d.t, range_v)
    var e = _linspace_edges(outer[0], outer[1], bins, a)
    var edges = List[T]()
    edges.append(e.t.copy())
    var w = _hist_flat_weight(w_v)
    _hist_check_inputs(x2d.t, edges, w[1])
    var hist = _hist_counts(x2d.t, edges, w[1], density)
    _ = x2d^  # alive past the launches
    _ = w^
    return (hist^, e^)


# aten::histogram.bins_tensor(Tensor self, Tensor bins, *, Tensor? weight=None,
#   bool density=False) -> (Tensor hist, Tensor bin_edges)
def op_histogram_bins_tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var r = _histogram_bins_tensor(
        v_tensor(args[unsafe_offset=0]),
        v_tensor(args[unsafe_offset=1]),
        args[unsafe_offset=2],
        v_bool_or(args[unsafe_offset=3], False),
    )
    ret_owned(rets, 0, r[0])
    ret_owned(rets, 1, r[1])


# aten::histogram.bin_ct(Tensor self, int bins=100, *, float[]? range=None,
#   Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor bin_edges)
def op_histogram_bin_ct(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var r = _histogram_bin_ct(
        v_tensor(args[unsafe_offset=0]),
        v_int_or(args[unsafe_offset=1], 100),
        args[unsafe_offset=2],
        args[unsafe_offset=3],
        v_bool_or(args[unsafe_offset=4], False),
    )
    ret_owned(rets, 0, r[0])
    ret_owned(rets, 1, r[1])


def _hist_into_outs(
    a: T, mut hist_out: T, mut edges_out: T, res: Tuple[Owned, Owned]
) raises:
    """`histogramdd_prepare_out`'s dtype checks, then resize and copy."""
    if hist_out.stype != a.stype:
        raise Error(
            (
                "torch.histogram: input tensor and hist tensor should have the"
                " same dtype, but got input "
            ),
            dtype_name_of(a),
            " and hist ",
            dtype_name_of(hist_out),
        )
    if edges_out.stype != a.stype:
        raise Error(
            (
                "torch.histogram: input tensor and bin_edges tensor should have"
                " the same dtype, but got input "
            ),
            dtype_name_of(a),
            " and bin_edges ",
            dtype_name_of(edges_out),
            " for dimension 0",
        )
    if not hist_out.same_shape(res[0].t):
        resize_out(hist_out, res[0].t.shape, res[0].t.rank)
    if not edges_out.same_shape(res[1].t):
        resize_out(edges_out, res[1].t.shape, res[1].t.rank)
    assert_no_internal_overlap(hist_out)
    assert_no_internal_overlap(edges_out)
    copy_strided_into(hist_out, res[0].t)
    copy_strided_into(edges_out, res[1].t)


def dtype_name_of(t: T) -> String:
    return _scalar_type_name(t.dtype)


# aten::histogram.bins_tensor_out(Tensor self, Tensor bins, *,
#   Tensor? weight=None, bool density=False, Tensor(a!) hist,
#   Tensor(b!) bin_edges) -> (Tensor(a!) hist, Tensor(b!) bin_edges)
def op_histogram_bins_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var hist_out = v_tensor(args[unsafe_offset=4])
    var edges_out = v_tensor(args[unsafe_offset=5])
    var r = _histogram_bins_tensor(
        a,
        v_tensor(args[unsafe_offset=1]),
        args[unsafe_offset=2],
        v_bool_or(args[unsafe_offset=3], False),
    )
    _hist_into_outs(a, hist_out, edges_out, r)
    _ = r^  # alive past the copies
    ret_ref(rets, 0, hist_out)
    ret_ref(rets, 1, edges_out)


# aten::histogram.bin_ct_out(Tensor self, int bins=100, *, float[]? range=None,
#   Tensor? weight=None, bool density=False, Tensor(a!) hist,
#   Tensor(b!) bin_edges) -> (Tensor(a!) hist, Tensor(b!) bin_edges)
def op_histogram_bin_ct_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var hist_out = v_tensor(args[unsafe_offset=5])
    var edges_out = v_tensor(args[unsafe_offset=6])
    var r = _histogram_bin_ct(
        a,
        v_int_or(args[unsafe_offset=1], 100),
        args[unsafe_offset=2],
        args[unsafe_offset=3],
        v_bool_or(args[unsafe_offset=4], False),
    )
    _hist_into_outs(a, hist_out, edges_out, r)
    _ = r^  # alive past the copies
    ret_ref(rets, 0, hist_out)
    ret_ref(rets, 1, edges_out)


def _dd_prep(a: T) raises -> Owned:
    _hist_check_dtype(a)
    if a.rank < 2:
        raise Error(
            "torch.histogramdd: input tensor should have at least 2 dimensions"
        )
    return _rows_2d(a)


def _dd_weight(
    w_v: Value, x2d: T
) raises -> Tuple[Optional[Owned], Optional[T]]:
    var weight = _opt_weight(w_v)
    if not weight:
        return (Optional[Owned](None), Optional[T](None))
    var wd = own_if_new(contiguous(weight.value()), weight.value())
    var wv = _view(wd.t, _shape1(x2d.dim(0)), 1)
    _ = wd^
    var t = wv.t.copy()
    return (Optional[Owned](wv^), Optional[T](t^))


# aten::_histogramdd_bin_edges(Tensor self, int[] bins, *, float[]? range=None,
#   Tensor? weight=None, bool density=False) -> Tensor[]
def op__histogramdd_bin_edges(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var x2d = _dd_prep(a)
    var bins = IntList(args[unsafe_offset=1]).to_list()
    var edges = _hist_edges_from_counts(x2d.t, bins, args[unsafe_offset=2])
    var ts = List[T]()
    for i in range(len(edges)):
        ts.append(edges[i].take())
    _ = x2d^
    ret_tensor_list(rets, 0, ts)


# aten::_histogramdd_from_bin_cts(Tensor self, int[] bins, *,
#   float[]? range=None, Tensor? weight=None, bool density=False) -> Tensor
def op__histogramdd_from_bin_cts(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var x2d = _dd_prep(a)
    var bins = IntList(args[unsafe_offset=1]).to_list()
    var edges = _hist_edges_from_counts(x2d.t, bins, args[unsafe_offset=2])
    var ts = _owned_ts(edges)
    var w = _dd_weight(args[unsafe_offset=3], x2d.t)
    _hist_check_inputs(a, ts, _opt_weight(args[unsafe_offset=3]))
    var hist = _hist_counts(
        x2d.t, ts, w[1], v_bool_or(args[unsafe_offset=4], False)
    )
    _ = x2d^
    _ = edges^
    _ = w^
    ret_owned(rets, 0, hist)


# aten::_histogramdd_from_bin_tensors(Tensor self, Tensor[] bins, *,
#   Tensor? weight=None, bool density=False) -> Tensor
def op__histogramdd_from_bin_tensors(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var ts = v_tensor_list(args[unsafe_offset=1])
    _hist_check_inputs(a, ts, _opt_weight(args[unsafe_offset=2]))
    var x2d = _dd_prep(a)
    var w = _dd_weight(args[unsafe_offset=2], x2d.t)
    var hist = _hist_counts(
        x2d.t, ts, w[1], v_bool_or(args[unsafe_offset=3], False)
    )
    _ = x2d^
    _ = w^
    ret_owned(rets, 0, hist)


# ---------------------------------------------------------------------------
# renorm (Normalization.cpp `renorm_out`, RenormKernel.cu)
# ---------------------------------------------------------------------------


def _renorm(a: T, p_v: Value, dim_in: Int, maxnorm_v: Value) raises -> Owned:
    """`self * factor`, factor = maxnorm / (norm + 1e-7) where the slice norm
    (float32 for the halves) exceeds maxnorm, else 1; the factor is computed
    in the norm's dtype and rounded to self's."""
    if not a.on_mojo():
        raise Error("expected a tensor on the mojo device")
    var p = v_f64(p_v)
    if not (p > 0.0):
        raise Error("renorm: non-positive-norm not supported")
    var maxnorm = v_f64(maxnorm_v)
    if not (maxnorm >= 0.0):
        raise Error(
            "renorm: expected maxnorm to be >= 0 but got ", _fmt_bound(maxnorm)
        )
    if a.rank <= 1:
        raise Error(
            "renorm: input needs at least 2 dimensions, got ",
            a.rank,
            " dimensions",
        )
    var dim = _norm_dim(dim_in, a.rank)
    var dims = List[Int64]()
    for d in range(a.rank):
        if d != dim:
            dims.append(Int64(d))
    var acc = _acc_stype(a.stype)
    var dtype_arg = none_arg()
    if acc != a.stype:
        dtype_arg = _dtype_v(acc)
    var norm = _call(
        "aten::linalg_vector_norm",
        "",
        [_t(a), p_v.copy(), _ilist(dims), bool_arg(True), dtype_arg^],
    )
    _ = dims
    # The factor's scalars are scalar_t of the norm's dtype.
    var max_s = maxnorm
    var eps = 1e-7
    if norm.t.stype == ST_FLOAT32:
        max_s = Float64(Float32(maxnorm))
        eps = Float64(Float32(1e-7))
    var shifted = _call(
        "aten::add", "Scalar", [_t(norm.t), _dbl(eps), _sint(1)]
    )
    var num = own(new_scalar(norm.t.stype, a.device))
    fill_value(num.t, max_s)
    var q = _call("aten::div", "Tensor", [_t(num.t), _t(shifted.t)])
    var big = _call("aten::gt", "Scalar", [_t(norm.t), _dbl(max_s)])
    var factor = _call(
        "aten::where", "ScalarOther", [_t(big.t), _t(q.t), _dbl(1.0)]
    )
    var f = _cast(factor.t, a.stype)
    var r = _mul(a, f.t)
    _ = norm^  # every temporary outlives the calls that read it
    _ = shifted^
    _ = num^
    _ = q^
    _ = big^
    _ = factor^
    _ = f^
    return r^


# aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor
def op_renorm(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var out = _renorm(
        v_tensor(args[unsafe_offset=0]),
        args[unsafe_offset=1],
        v_int(args[unsafe_offset=2]),
        args[unsafe_offset=3],
    )
    ret_owned(rets, 0, out)


# aten::renorm.out(Tensor self, Scalar p, int dim, Scalar maxnorm, *,
#   Tensor(a!) out) -> Tensor(a!)
def op_renorm_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=4])
    check_out(out, a)
    var res = _renorm(
        a,
        args[unsafe_offset=1],
        v_int(args[unsafe_offset=2]),
        args[unsafe_offset=3],
    )
    if not out.same_shape(res.t):
        resize_out(out, a.shape, a.rank)
    assert_no_internal_overlap(out)
    copy_strided_into(out, res.t)
    _ = res^  # alive past the launch
    ret_ref(rets, 0, out)


# aten::renorm_(Tensor(a!) self, Scalar p, int dim, Scalar maxnorm)
#   -> Tensor(a!)
def op_renorm_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    assert_no_internal_overlap(a)
    var res = _renorm(
        a,
        args[unsafe_offset=1],
        v_int(args[unsafe_offset=2]),
        args[unsafe_offset=3],
    )
    copy_strided_into(a, res.t)
    _ = res^  # alive past the launch
    ret_ref(rets, 0, a)


# ---------------------------------------------------------------------------
# _weight_norm_interface and its backward (WeightNorm.cu): the norm over
# every dim but `dim`, accumulated (and stored) in float32 for the halves.
# ---------------------------------------------------------------------------


def _other_dims(rank: Int, dim: Int) -> List[Int64]:
    var dims = List[Int64]()
    for d in range(rank):
        if d != dim:
            dims.append(Int64(d))
    return dims^


def _keep_shape(v: T, dim: Int) -> IndexList[MAX_RANK]:
    """`v`'s shape with every dim but `dim` set to 1."""
    var s = IndexList[MAX_RANK](1)
    s[MAX_RANK - v.rank + dim] = v.dim(dim)
    return s


def _as_keep(t: T, v: T, dim: Int) raises -> Owned:
    """`t` (one element per slice of `v` along `dim`) viewed in `v`'s
    keepdim shape."""
    if t.numel != v.dim(dim):
        raise Error("weight_norm: expected one value per slice along dim ", dim)
    var dense = own_if_new(contiguous(t), t)
    var r = _view(dense.t, _keep_shape(v, dim), v.rank)
    _ = dense^
    return r^


def _as_column(t: T) raises -> Owned:
    """A 0-/1-d `t` viewed as (numel, 1) (contiguous first); any other `t`
    as itself."""
    if t.rank > 1:
        return own(T(retain(t)))
    var dense = own_if_new(contiguous(t), t)
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 2] = t.numel
    var r = _view(dense.t, shape, 2)
    _ = dense^
    return r^


# aten::_weight_norm_interface(Tensor v, Tensor g, int dim=0)
#   -> (Tensor, Tensor)
def op__weight_norm_interface(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var v = v_tensor(args[unsafe_offset=0])
    var g = v_tensor(args[unsafe_offset=1])
    var dim = _norm_dim(v_int_or(args[unsafe_offset=2], 0), v.rank)
    if not v.dtype.is_floating_point():
        raise Error(
            '"weight_norm_fwd_first_dim_kernel" not implemented for \'',
            _scalar_type_name(v.dtype),
            "'",
        )
    _decline_metal_f64(v, "_weight_norm_interface")
    # A 0-/1-d weight has no other dim: every element is its own slice
    # (CUDA's first-dim kernel with rowSize 1), so read it as (n, 1).
    var v_shape = v.shape
    var v_rank = v.rank
    var vcol = _as_column(v)
    v = vcol.t.copy()
    if v_rank <= 1:
        dim = 0
    var acc = _acc_stype(g.stype)
    var vf = _cast(v, acc)
    var dims = _other_dims(v.rank, dim)
    # norms = sqrt(sum v^2) per slice in the accumulate type, as
    # WeightNorm.cu forms it. linalg_vector_norm's L2 reduce is that plain
    # sum of squares (no rescaling), except for torch's own shortcut when
    # every reduced dim has extent 1 (abs(v), which cannot overflow): there
    # square v and take the root directly, so a float32 1e20 gives an inf
    # norm and a zero weight, as on CUDA.
    var slice_len = vf.t.numel // vf.t.dim(dim) if vf.t.dim(dim) > 0 else 0
    var norm: Owned
    if slice_len == 1:
        var sq = _mul(vf.t, vf.t)
        # aten::sqrt has no float64 kernel; pow(x, 0.5) is PowKernel.cu's
        # sqrt special case (`_pw_pow_f64_scalar`), the same rounded root.
        norm = _call(
            "aten::pow", "Tensor_Scalar", [_t(sq.t), _dbl(0.5)]
        ) if sq.t.stype == ST_FLOAT64 else _call("aten::sqrt", "", [_t(sq.t)])
        _ = sq^  # alive past the call that reads it
    else:
        norm = _call(
            "aten::linalg_vector_norm",
            "",
            [_t(vf.t), _dbl(2.0), _ilist(dims), bool_arg(True), none_arg()],
        )
    _ = dims
    var rnorm = _reciprocal(norm.t)
    var gf = _cast(g, acc)
    var gk = _as_keep(gf.t, v, dim)
    var gv = _mul(gk.t, vf.t)
    var wf = _mul(gv.t, rnorm.t)
    var w2 = _cast(wf.t, v.stype)
    var w = _view(w2.t, v_shape, v_rank)
    _ = w2^
    _ = vcol^
    var norms = own(new_tensor(g.shape, g.rank, acc, g.device))
    var nview = _view(norm.t, g.shape, g.rank)
    copy_strided_into(norms.t, nview.t)
    _ = nview^  # every temporary outlives the calls that read it
    _ = vf^
    _ = norm^
    _ = rnorm^
    _ = gf^
    _ = gk^
    _ = gv^
    _ = wf^
    ret_owned(rets, 0, w)
    ret_owned(rets, 1, norms)


# aten::_weight_norm_interface_backward(Tensor grad_w, Tensor saved_v,
#   Tensor saved_g, Tensor saved_norms, int dim) -> (Tensor, Tensor)
def op__weight_norm_interface_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var gw = v_tensor(args[unsafe_offset=0])
    var v = v_tensor(args[unsafe_offset=1])
    var g = v_tensor(args[unsafe_offset=2])
    var norms = v_tensor(args[unsafe_offset=3])
    var dim = _norm_dim(v_int(args[unsafe_offset=4]), v.rank)
    if not v.contig:
        raise Error("saved_v must be contiguous")
    if not g.contig:
        raise Error("saved_g must be contiguous")
    if not norms.contig:
        raise Error("saved_norms must be contiguous")
    if dim != 0 and dim != v.rank - 1:
        raise Error("fused kernels can only be applied for first or last dim")
    _decline_metal_f64(v, "_weight_norm_interface_backward")
    var v_shape = v.shape
    var v_rank = v.rank
    var vcol = _as_column(v)
    var gwcol = _as_column(gw)
    v = vcol.t.copy()
    gw = gwcol.t.copy()
    if v_rank <= 1:
        dim = 0
    var acc = _acc_stype(v.stype)
    var gwf = _cast(gw, acc)
    var vf = _cast(v, acc)
    var dims = _other_dims(v.rank, dim)
    var prod = _mul(gwf.t, vf.t)
    var result = _sum_dims(prod.t, dims, True)
    var nk = _as_keep(norms, v, dim)
    var nf = _cast(nk.t, acc)
    var rnorm = _reciprocal(nf.t)
    var r2 = _mul(rnorm.t, rnorm.t)
    var rnorm3 = _mul(r2.t, rnorm.t)
    # grad_g = result * rnorm
    var gg_f = _mul(result.t, rnorm.t)
    var gg = _cast(gg_f.t, g.stype)
    var grad_g = own(new_tensor(g.shape, g.rank, g.stype, g.device))
    var ggv = _view(gg.t, g.shape, g.rank)
    copy_strided_into(grad_g.t, ggv.t)
    _ = ggv^
    _ = gg^
    _ = gg_f^
    # grad_v = g * (rnorm * grad_w - rnorm3 * v * result)
    var t1 = _mul(rnorm.t, gwf.t)
    var t2a = _mul(rnorm3.t, vf.t)
    var t2 = _mul(t2a.t, result.t)
    var diff = _sub(t1.t, t2.t)
    var gf = _cast(g, acc)
    var gk = _as_keep(gf.t, v, dim)
    var gvf = _mul(gk.t, diff.t)
    var gv2 = _cast(gvf.t, v.stype)
    var grad_v = _view(gv2.t, v_shape, v_rank)
    _ = gv2^
    _ = vcol^
    _ = gwcol^
    _ = gwf^  # every temporary outlives the calls that read it
    _ = vf^
    _ = prod^
    _ = result^
    _ = nk^
    _ = nf^
    _ = rnorm^
    _ = r2^
    _ = rnorm3^
    _ = t1^
    _ = t2a^
    _ = t2^
    _ = diff^
    _ = gf^
    _ = gk^
    _ = gvf^
    ret_owned(rets, 0, grad_v)
    ret_owned(rets, 1, grad_g)


# ---------------------------------------------------------------------------
# _fused_rms_norm_backward (layer_norm_kernel.cu, `compute_gI` with
# rms_norm=true and the gamma-gradient kernel)
# ---------------------------------------------------------------------------


def _bool_list(v: Value) raises -> List[Bool]:
    var out = List[Bool]()
    if v.tag == TAG_NONE:
        return out^
    var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(v.a))
    for i in range(Int(v.len)):
        out.append(p[unsafe_offset=i] != 0)
    return out^


def _empty_like_result(st: Int32, device: Int) raises -> Owned:
    """The 0-element stand-in for a gradient autograd did not ask for (the
    convention `native_layer_norm_backward` uses)."""
    return own(new_tensor(IndexList[MAX_RANK](0), 1, st, device))


# aten::_fused_rms_norm_backward(Tensor grad_out, Tensor input,
#   int[] normalized_shape, Tensor rstd, Tensor? weight, bool[2] output_mask)
#   -> (Tensor, Tensor)
def op__fused_rms_norm_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var dy = v_tensor(args[unsafe_offset=0])
    var x = v_tensor(args[unsafe_offset=1])
    var ns = IntList(args[unsafe_offset=2])
    var rstd = v_tensor(args[unsafe_offset=3])
    var has_w = not v_is_none(args[unsafe_offset=4])
    var mask = _bool_list(args[unsafe_offset=5])
    var k = len(ns)
    if k == 0 or k > x.rank:
        raise Error(
            "Expected normalized_shape to be at least 1-dimensional and to"
            " fit the input"
        )
    for i in range(k):
        if x.dim(x.rank - k + i) != ns[i]:
            raise Error("Given normalized_shape does not match the input shape")
    if not x.dtype.is_floating_point():
        raise Error("_fused_rms_norm_backward expects a floating input")
    _decline_metal_f64(x, "_fused_rms_norm_backward")
    var n = 1
    for i in range(k):
        n *= ns[i]
    var acc = _acc_stype(x.stype)
    var axis = x.rank - k
    var red = List[Int64]()
    for d in range(axis, x.rank):
        red.append(Int64(d))
    var lead = List[Int64]()
    for d in range(axis):
        lead.append(Int64(d))
    var xf = _cast(x, acc)
    var dyf = _cast(dy, acc)
    var rf = _cast(rstd, acc)
    var want_dx = len(mask) > 0 and mask[0]
    var want_dw = len(mask) > 1 and mask[1]
    var dx: Owned
    if want_dx:
        # stats = sum(dy * gamma * x * rstd) over the normalized dims
        var dyg: Owned
        if has_w:
            var wf = _cast(v_tensor(args[unsafe_offset=4]), acc)
            dyg = _mul(dyf.t, wf.t)
            _ = wf^  # alive past the call that reads it
        else:
            dyg = own(T(retain(dyf.t)))
        var a1 = _mul(dyg.t, xf.t)
        var a2 = _mul(a1.t, rf.t)
        var stats = _sum_dims(a2.t, red, True)
        # dx = (1 / N) * rstd * (N * gamma * dy - x * rstd * stats), in
        # compute_gI's order: (N * gamma) * dy, (x * rstd) * stats.
        var nd: Owned
        if has_w:
            var wf2 = _cast(v_tensor(args[unsafe_offset=4]), acc)
            var wn = _call("aten::mul", "Scalar", [_t(wf2.t), _dbl(Float64(n))])
            nd = _mul(wn.t, dyf.t)
            _ = wf2^  # alive past the calls that read them
            _ = wn^
        else:
            nd = _call("aten::mul", "Scalar", [_t(dyf.t), _dbl(Float64(n))])
        var xr = _mul(xf.t, rf.t)
        var xrs = _mul(xr.t, stats.t)
        var f = _sub(nd.t, xrs.t)
        var inv_n = 1.0 / Float64(n)
        if acc == ST_FLOAT32:
            inv_n = Float64(Float32(1.0) / Float32(n))
        var term1 = _call("aten::mul", "Scalar", [_t(rf.t), _dbl(inv_n)])
        var dxf = _mul(f.t, term1.t)
        dx = _cast(dxf.t, x.stype)
        _ = dyg^  # every temporary outlives the calls that read it
        _ = a1^
        _ = a2^
        _ = stats^
        _ = nd^
        _ = xr^
        _ = xrs^
        _ = f^
        _ = term1^
        _ = dxf^
    else:
        dx = _empty_like_result(x.stype, x.device)  # released, never returned
    var dw: Owned
    if want_dw and has_w:
        var w = v_tensor(args[unsafe_offset=4])
        var b1 = _mul(dyf.t, xf.t)
        var b2 = _mul(b1.t, rf.t)
        var s: Owned
        if len(lead) > 0:
            s = _sum_dims(b2.t, lead, False)
        else:
            s = own(T(retain(b2.t)))
        var sw = _cast(s.t, w.stype)
        dw = own(new_tensor(w.shape, w.rank, w.stype, w.device))
        var swv = _view(sw.t, w.shape, w.rank)
        copy_strided_into(dw.t, swv.t)
        _ = swv^  # every temporary outlives the calls that read it
        _ = sw^
        _ = s^
        _ = b1^
        _ = b2^
    else:
        dw = _empty_like_result(x.stype, x.device)
    _ = xf^
    _ = dyf^
    _ = rf^
    # A masked-off (or weightless) gradient is an undefined Tensor: a None
    # record for a `Tensor` result (shim_dispatch.cpp `from_record`).
    if want_dx:
        ret_owned(rets, 0, dx)
    else:
        rets[unsafe_offset=0] = none_arg()
    if want_dw and has_w:
        ret_owned(rets, 1, dw)
    else:
        rets[unsafe_offset=1] = none_arg()


# ---------------------------------------------------------------------------
# _compute_linear_combination (FunctionOfAMatrixUtils.cpp)
# ---------------------------------------------------------------------------


def _lincomb_into(inp: T, coeff: T, dst: T) raises:
    """`out[i, ...] += sum_j coeff[i, j] * inp[j, ...]` into the contiguous
    `dst` (accumulating onto its current contents, as ATen's kernel does)."""
    if inp.rank == 0 or inp.numel == 0:
        raise Error("Empty tensor not supported")
    if coeff.rank != 2 or coeff.dim(1) != inp.dim(0):
        raise Error(
            "_compute_linear_combination: coefficients must be [m, n] with n"
            " = input.size(0)"
        )
    if coeff.stype != inp.stype or dst.stype != inp.stype:
        unsupported("_compute_linear_combination with mixed dtypes")
    var dt = inp.dtype
    if dt == DType.bool or not (dt.is_floating_point() or dt.is_integral()):
        unsupported("_compute_linear_combination of dtype " + String(dt))
    if dt == DType.uint16 or dt == DType.uint32 or dt == DType.uint64:
        unsupported("_compute_linear_combination of dtype " + String(dt))
    _decline_metal_f64(inp, "_compute_linear_combination")
    var m = coeff.dim(0)
    var n = coeff.dim(1)
    var rest = inp.numel // n
    var ic = own_if_new(contiguous(inp), inp)
    var cc = own_if_new(contiguous(coeff), coeff)
    var ctx = ctx_for(dst.device)
    var tup = List[Int]()
    tup.append(dtype_code(dt))
    tup.append(ctx_ptr(ctx))
    var call = KernelCall("stats", "LinearCombination")
    call.arg_dtype(0, dt)
    call.int(dst.ptr)
    call.int(ic.t.ptr)
    call.int(cc.t.ptr)
    call.int(m)
    call.int(n)
    call.int(rest)
    call.tuple(tup)
    call.run()
    _ = ctx
    _ = ic^  # alive past the launch
    _ = cc^


def _lincomb_shape(inp: T, coeff: T) -> IndexList[MAX_RANK]:
    var s = inp.shape
    s[MAX_RANK - inp.rank] = coeff.dim(0) if coeff.rank > 0 else 0
    return s


# aten::_compute_linear_combination(Tensor input, Tensor coefficients)
#   -> Tensor
def op__compute_linear_combination(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var inp = v_tensor(args[unsafe_offset=0])
    var coeff = v_tensor(args[unsafe_offset=1])
    if inp.rank == 0 or inp.numel == 0:
        raise Error("Empty tensor not supported")
    var out = _zeros(
        _lincomb_shape(inp, coeff), inp.rank, inp.stype, inp.device
    )
    _lincomb_into(inp, coeff, out.t)
    ret_owned(rets, 0, out)


# aten::_compute_linear_combination.out(Tensor input, Tensor coefficients, *,
#   Tensor(a!) out) -> Tensor(a!)
def op__compute_linear_combination_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var inp = v_tensor(args[unsafe_offset=0])
    var coeff = v_tensor(args[unsafe_offset=1])
    var out = v_tensor(args[unsafe_offset=2])
    check_out(out, inp)
    var shape = _lincomb_shape(inp, coeff)
    var ok = out.rank == inp.rank
    if ok:
        for i in range(inp.rank):
            if out.dim(i) != shape[MAX_RANK - inp.rank + i]:
                ok = False
    if not ok:
        raise Error(
            "_compute_linear_combination: out must have shape [m, ...] of"
            " the result"
        )
    if out.contig:
        _lincomb_into(inp, coeff, out)
    else:
        var tmp = own(new_tensor(shape, inp.rank, inp.stype, inp.device))
        copy_strided_into(tmp.t, out)
        _lincomb_into(inp, coeff, tmp.t)
        copy_strided_into(out, tmp.t)
        _ = tmp^  # alive past the launches
    ret_ref(rets, 0, out)


# ---------------------------------------------------------------------------
# segment_reduce and its backward (SegmentReduce.cpp / SegmentReduce.cu)
# ---------------------------------------------------------------------------


comptime _RED_MAX = 0
comptime _RED_MEAN = 1
comptime _RED_MIN = 2
comptime _RED_SUM = 3
comptime _RED_PROD = 4


def _reduction_enum(reduce: String) raises -> Int:
    """ATen's `get_reduction_enum`."""
    if reduce == "max" or reduce == "amax":
        return _RED_MAX
    if reduce == "mean":
        return _RED_MEAN
    if reduce == "min" or reduce == "amin":
        return _RED_MIN
    if reduce == "sum":
        return _RED_SUM
    if reduce == "prod":
        return _RED_PROD
    raise Error(
        "reduce argument must be either sum, prod, mean, amax or amin, got ",
        reduce,
    )


struct _Segments(Movable):
    """lengths and offsets as contiguous int64 (outer, segments) /
    (outer, segments + 1) tensors, whichever of the two was given."""

    var lengths: Owned
    var offsets: Owned
    var outer: Int
    var segments: Int

    def __init__(
        out self,
        var lengths: Owned,
        var offsets: Owned,
        outer: Int,
        segments: Int,
    ):
        self.lengths = lengths^
        self.offsets = offsets^
        self.outer = outer
        self.segments = segments


def _index_tensor(t: T, what: StaticString) raises -> Owned:
    if t.dtype != DType.int64 and t.dtype != DType.int32:
        raise Error(
            '"',
            what,
            "\" not implemented for '",
            _scalar_type_name(t.dtype),
            "'",
        )
    var c = _cast(t, ST_INT64)
    var d = own_if_new(contiguous(c.t), c.t)
    if d.t.h == c.t.h:
        return c^
    return d^


def _segments(
    lengths_v: Value, offsets_v: Value, what: StaticString
) raises -> _Segments:
    """Both forms from whichever was given (offsets win, as ATen's do):
    lengths = diff(offsets) or offsets = cat(0, cumsum(lengths)) along the
    last dim."""
    if not v_is_none(offsets_v):
        var off = _index_tensor(v_tensor(offsets_v), what)
        var r = off.t.rank
        var segs = off.t.dim(r - 1) - 1
        var outer = off.t.numel // (segs + 1) if segs + 1 > 0 else 0
        var shape = off.t.shape
        shape[MAX_RANK - 1] = segs
        var strides = off.t.strides
        var hi = own(view_strided(off.t, shape, strides, r, off.t.offset + 1))
        var lo = own(view_strided(off.t, shape, strides, r, off.t.offset))
        var lens = _sub(hi.t, lo.t)
        _ = hi^  # alive past the subtraction
        _ = lo^
        var lens_c = own_if_new(contiguous(lens.t), lens.t)
        if lens_c.t.h != lens.t.h:
            return _Segments(lens_c^, off^, outer, segs)
        return _Segments(lens^, off^, outer, segs)
    var lens = _index_tensor(v_tensor(lengths_v), what)
    var r = lens.t.rank
    var segs = lens.t.dim(r - 1) if r > 0 else 1
    var outer = lens.t.numel // segs if segs > 0 else 0
    var shape = lens.t.shape
    shape[MAX_RANK - 1] = segs + 1
    var off = _zeros(shape, max(r, 1), ST_INT64, lens.t.device)
    if lens.t.numel > 0:
        var cs = _call(
            "aten::cumsum", "", [_t(lens.t), int_arg(-1), none_arg()]
        )
        var vshape = shape
        vshape[MAX_RANK - 1] = segs
        var tail = own(
            view_strided(
                off.t, vshape, off.t.strides, max(r, 1), off.t.offset + 1
            )
        )
        copy_strided_into(tail.t, cs.t)
        _ = tail^  # alive past the copy
        _ = cs^
    return _Segments(lens^, off^, outer, segs)


def _segment_dtype_check(t: T) raises:
    var dt = t.dtype
    if not (
        dt == DType.float32
        or dt == DType.float16
        or dt == DType.bfloat16
        or dt == DType.float64
    ):
        raise Error(
            '"segment_reduce_cuda" not implemented for \'',
            _scalar_type_name(dt),
            "'",
        )
    _decline_metal_f64(t, "segment_reduce")


def _initial_value(
    v: Value, st: Int32, reduction: Int
) raises -> Tuple[Bool, Float64]:
    """(initial given?, the start value): `initial.to<scalar_t>()` or the
    reduction's identity."""
    if not v_is_none(v):
        return (True, scalar_to_float(v, st))
    if reduction == _RED_MAX:
        return (False, min_or_neg_inf[DType.float64]())
    if reduction == _RED_MIN:
        return (False, max_or_inf[DType.float64]())
    if reduction == _RED_PROD:
        return (False, 1.0)
    return (False, 0.0)


def _outer_inner(t: T, axis: Int) -> Tuple[Int, Int]:
    var outer = 1
    for d in range(axis):
        outer *= t.dim(d)
    var inner = 1
    for d in range(axis + 1, t.rank):
        inner *= t.dim(d)
    return (outer, inner)


# aten::segment_reduce(Tensor data, str reduce, *, Tensor? lengths=None,
#   Tensor? indices=None, Tensor? offsets=None, int axis=0, bool unsafe=False,
#   Scalar? initial=None) -> Tensor
def op_segment_reduce(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var data = v_tensor(args[unsafe_offset=0])
    var reduce = v_string(args[unsafe_offset=1])
    var lengths_v = args[unsafe_offset=2].copy()
    var indices_v = args[unsafe_offset=3].copy()
    var offsets_v = args[unsafe_offset=4].copy()
    var axis = _norm_dim(v_int_or(args[unsafe_offset=5], 0), data.rank)
    var unsafe_ = v_bool_or(args[unsafe_offset=6], False)
    if not v_is_none(indices_v):
        raise Error(
            "segment_reduce(): indices based reduction is not supported yet."
        )
    if v_is_none(lengths_v) and v_is_none(offsets_v):
        raise Error(
            "segment_reduce(): Either lengths or offsets must be defined."
        )
    var reduction = _reduction_enum(reduce)
    var given = v_tensor(offsets_v) if not v_is_none(offsets_v) else v_tensor(
        lengths_v
    )
    var word = String("lengths")
    if not v_is_none(offsets_v):
        word = "offsets"
    if given.device != data.device or given.device_type != data.device_type:
        raise Error(
            "segment_reduce(): data and ", word, " on different devices"
        )
    if data.rank < given.rank:
        raise Error("segment_reduce(): data.dim() must be >= ", word, ".dim()")
    if axis != given.rank - 1:
        raise Error(
            "segment_reduce(): Expected axis to be the last dimension of ",
            word,
            " but got ",
            axis,
            ".",
        )
    _segment_dtype_check(data)
    if v_is_none(offsets_v) and not unsafe_:
        var lens0 = _index_tensor(given, "segment_reduce")
        if lens0.t.numel > 0:
            var mn = _call("aten::min", "", [_t(lens0.t)])
            if _read_scalar_int(mn) < 0:
                raise Error("lengths contains negative value!")
        var red = List[Int64]()
        red.append(-1)
        var sums = _sum_dims(lens0.t, red, False)
        var eq = _call(
            "aten::eq", "Scalar", [_t(sums.t), _sint(data.dim(axis))]
        )
        var all_ = _call("aten::all", "", [_t(eq.t)])
        _ = lens0^  # alive past the reductions that read it
        _ = sums^  # alive past the comparison
        _ = eq^
        if _read_scalar_int(all_) == 0:
            raise Error(
                "segment_reduce(): Expected all rows of lengths along axis to"
                " sum to data.size(lengths.dim()-1) when !unsafe."
            )
    var seg = _segments(lengths_v, offsets_v, "_segment_reduce_cuda_kernel1")
    var dc = own_if_new(contiguous(data), data)
    var oi = _outer_inner(dc.t, axis)
    var shape = dc.t.shape
    shape[MAX_RANK - dc.t.rank + axis] = seg.segments
    var out = own(new_tensor(shape, dc.t.rank, dc.t.stype, dc.t.device))
    var init = _initial_value(args[unsafe_offset=7], data.stype, reduction)
    if out.t.numel > 0:
        var ctx = ctx_for(data.device)
        var call = KernelCall("stats", "SegmentReduce")
        call.arg_dtype(0, data.dtype)
        call.int(out.t.ptr)
        call.int(dc.t.ptr)
        call.int(seg.lengths.t.ptr)
        call.int(seg.offsets.t.ptr)
        call.int(reduction)
        call.int(oi[0])
        call.int(seg.segments)
        call.int(oi[1])
        call.int(dc.t.dim(axis))
        call.int(1 if init[0] else 0)
        call.f64(init[1])
        call.int(dtype_code(data.dtype))
        call.int(ctx_ptr(ctx))
        call.run()
        _ = ctx
    _ = dc^  # alive past the launch
    _ = seg^
    ret_owned(rets, 0, out)


# aten::_segment_reduce_backward(Tensor grad, Tensor output, Tensor data,
#   str reduce, *, Tensor? lengths=None, Tensor? offsets=None, int axis=0,
#   Scalar? initial=None) -> Tensor
def op__segment_reduce_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var output = v_tensor(args[unsafe_offset=1])
    var data = v_tensor(args[unsafe_offset=2])
    var reduction = _reduction_enum(v_string(args[unsafe_offset=3]))
    var lengths_v = args[unsafe_offset=4].copy()
    var offsets_v = args[unsafe_offset=5].copy()
    var axis = _norm_dim(v_int_or(args[unsafe_offset=6], 0), data.rank)
    if v_is_none(lengths_v) and v_is_none(offsets_v):
        raise Error(
            "segment_reduce(): Either lengths or offsets must be defined."
        )
    _segment_dtype_check(data)
    if grad.stype != data.stype or output.stype != data.stype:
        unsupported("_segment_reduce_backward with mixed dtypes")
    var seg = _segments(
        lengths_v,
        offsets_v,
        "_segment_reduce_cuda_lengths_offsets_backward_kernel1",
    )
    var gc = own_if_new(contiguous(grad), grad)
    var yc = own_if_new(contiguous(output), output)
    var dc = own_if_new(contiguous(data), data)
    var oi = _outer_inner(yc.t, axis)
    var gi = _zeros(dc.t.shape, dc.t.rank, grad.stype, grad.device)
    var init = 1.0
    if not v_is_none(args[unsafe_offset=7]):
        init = scalar_to_float(args[unsafe_offset=7], data.stype)
    if gc.t.numel > 0:
        var ctx = ctx_for(data.device)
        var tup = List[Int]()
        tup.append(dtype_code(data.dtype))
        tup.append(ctx_ptr(ctx))
        var call = KernelCall("stats", "SegmentReduceBwd")
        call.arg_dtype(0, data.dtype)
        call.int(gi.t.ptr)
        call.int(gc.t.ptr)
        call.int(yc.t.ptr)
        call.int(dc.t.ptr)
        call.int(seg.lengths.t.ptr)
        call.int(seg.offsets.t.ptr)
        call.int(reduction)
        call.int(oi[0])
        call.int(seg.segments)
        call.int(oi[1])
        call.int(dc.t.dim(axis))
        call.f64(init)
        call.tuple(tup)
        call.run()
        _ = ctx
    _ = gc^  # alive past the launch
    _ = yc^
    _ = dc^
    _ = seg^
    ret_owned(rets, 0, gi)


def register_stats(site: Site) raises:
    impl[op__compute_linear_combination, "_compute_linear_combination"](site)
    impl[op__compute_linear_combination_out, "_compute_linear_combination.out"](
        site
    )
    impl[op__fused_rms_norm_backward, "_fused_rms_norm_backward"](site)
    impl[op__histogramdd_bin_edges, "_histogramdd_bin_edges"](site)
    impl[op__histogramdd_from_bin_cts, "_histogramdd_from_bin_cts"](site)
    impl[op__histogramdd_from_bin_tensors, "_histogramdd_from_bin_tensors"](
        site
    )
    impl[op__segment_reduce_backward, "_segment_reduce_backward"](site)
    impl[op__weight_norm_interface, "_weight_norm_interface"](site)
    impl[op__weight_norm_interface_backward, "_weight_norm_interface_backward"](
        site
    )
    impl[op_bincount, "bincount"](site)
    impl[op_histc, "histc"](site)
    impl[op_histc_out, "histc.out"](site)
    impl[op_histogram_bin_ct, "histogram.bin_ct"](site)
    impl[op_histogram_bin_ct_out, "histogram.bin_ct_out"](site)
    impl[op_histogram_bins_tensor, "histogram.bins_tensor"](site)
    impl[op_histogram_bins_tensor_out, "histogram.bins_tensor_out"](site)
    impl[op_mode, "mode"](site)
    impl[op_mode_values, "mode.values"](site)
    impl[op_renorm, "renorm"](site)
    impl[op_renorm_, "renorm_"](site)
    impl[op_renorm_out, "renorm.out"](site)
    impl[op_segment_reduce, "segment_reduce"](site)
