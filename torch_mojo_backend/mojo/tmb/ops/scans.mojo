"""ATen ops: the cumulative scans along one dimension -- cumsum, cumprod
(functional, `out=` and in-place), logcumsumexp (`_logcumsumexp`) and the
cummax / cummin helpers ATen's composite `cummax` / `cummin` call.

Every scan runs on the scan family (tmb/kernels/scan/entry.mojo) over a
contiguous operand viewed as (outer, n, inner); cumsum keeps the nn family's
block-prefix-sum kernels for the layouts they cover (the trailing dim of a
contiguous operand, dim 0 of a rank-2 one). Results are computed into a fresh
contiguous tensor and copied into a strided or aliasing `out=`, so an `out`
that shares the operand's storage is only written after the operand was read.

Semantics are ATen's `meta_func_cum_ops` / `impl_func_cum_ops` (ReduceOps.cpp)
and `cummax_helper_cuda` / `_logcumsumexp_out_cuda` (ScanKernels.cpp): the
dim is wrapped (a 0-d operand has one), bool and integer operands promote to
int64 without `dtype=` (and with no `out=` to take a dtype from), and a 0-d
operand is copied while an empty one has nothing to scan.
"""
from std.utils import IndexList

from tmb.backend.abi import (
    ST_INT64,
    Owned,
    T,
    Values,
    dtype_code,
    dtype_name,
    max_dtype,
    new_like_dtype,
    new_tensor,
    own,
    own_if_new,
    ret_owned,
    ret_ref,
    unsupported,
    v_dtype_or,
    v_int,
    v_tensor,
)
from tmb.backend.device import ctx_for, ctx_ptr, dev
from tmb.backend.kernel_call import KernelCall
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.common import (
    assert_no_internal_overlap,
    check_out,
    check_out_as,
    contiguous,
    copy_strided_into,
    resize_out,
)
from tmb.ops.reductions import _cast_any, _norm_dim
from tmb.backend.registry import Site, impl


# ---------------------------------------------------------------------------
# geometry and launch
# ---------------------------------------------------------------------------


def _geometry(t: T, dim: Int) -> Tuple[Int, Int, Int]:
    """(outer, n, inner) of `t` around `dim` (a 0-d tensor is (1, 1, 1))."""
    if t.rank == 0:
        return (1, 1, 1)
    var outer = 1
    for d in range(dim):
        outer *= t.dim(d)
    var inner = 1
    for d in range(dim + 1, t.rank):
        inner *= t.dim(d)
    return (outer, t.dim(dim), inner)


def _kernel_dtype(dt: DType) -> DType:
    """bool rides uint8 storage through the selecting scans."""
    return DType.uint8 if dt == DType.bool else dt


def _scan_into(op: StaticString, src: T, dim: Int, dst: T, idx_ptr: Int) raises:
    """One scan of the contiguous `src` into the fresh contiguous `dst` (same
    shape and dtype), plus int64 indices at `idx_ptr` for cummax / cummin."""
    if src.numel == 0:
        return
    if dev(dst.device)[].api == "cpu":
        unsupported("the scan kernels need a GPU (MAX's CPU device has none)")
    var g = _geometry(src, dim)
    var kdt = _kernel_dtype(src.dtype)
    var ctx = ctx_for(dst.device)
    var call = KernelCall("scan", String(op))
    call.arg_dtype(0, kdt)
    call.int(dst.ptr)
    call.int(idx_ptr)
    call.int(src.ptr)
    call.int(g[0])
    call.int(g[1])
    call.int(g[2])
    call.int(dtype_code(kdt))
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx


def _decline_metal_f64(t: T, what: StaticString) raises:
    if t.dtype == DType.float64 and dev(t.device)[].api == "metal":
        unsupported(String(what) + ": float64 is unavailable on Apple GPUs")


def _is_sum_scan_dtype(dt: DType) -> Bool:
    """The scan family's SUM_DTYPES (cumsum / cumprod results)."""
    return (
        dt == DType.float32
        or dt == DType.float16
        or dt == DType.bfloat16
        or dt == DType.float64
        or dt == DType.int64
        or dt == DType.int32
    )


def _nn_cumsum_ok(src: T, dim: Int) raises -> Bool:
    """The nn family's block-prefix-sum regimes (`_cumsum_spec_into_go`) of a
    contiguous float32/int32/int64 operand: the trailing dim everywhere
    (MAX's CPU device included), and on a GPU also dim 0 of a rank-2
    operand. bf16/f16 never take it: its float32 running sum rounds once,
    where CUDA's `scan_dim<scalar_t>` rounds the half sum after every
    addition (`ones(4096)` saturates at 2048 in half) -- the scan family
    reproduces that."""
    if not src.contig or src.rank < 1:
        return False
    var dt = src.dtype
    if not (dt == DType.float32 or dt == DType.int32 or dt == DType.int64):
        return False
    var gpu = dev(src.device)[].api != "cpu"
    return dim == src.rank - 1 or (gpu and src.rank == 2 and dim == 0)


def _cumsum_nn_into(src: T, dim: Int, dst: T) raises:
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("nn", "CumsumSpec")
    call.arg_dtype(0, src.dtype)
    call.out_dtype(dst.dtype)
    call.spec(src.spec(cp))
    call.int(dim)
    call.spec(dst.spec(cp))
    call.run()
    _ = ctx


# ---------------------------------------------------------------------------
# cumsum / cumprod
# ---------------------------------------------------------------------------


def _cum_result(
    is_sum: Bool, a: T, dim_in: Int, res_st: Int32, what: StaticString
) raises -> Owned:
    """`impl_func_cum_ops`: the scan of `a.to(res_st)` along `dim_in` as a
    fresh contiguous tensor of `a`'s shape and dtype `res_st`."""
    var dim = _norm_dim(dim_in, a.rank)
    var rdt = max_dtype(res_st)
    if a.numel == 0:
        return own(new_tensor(a.shape, a.rank, res_st, a.device))
    var metal = dev(a.device)[].api == "metal"
    if metal and rdt == DType.float64 and a.stype != res_st:
        unsupported(String(what) + ": float64 is unavailable on Apple GPUs")
    if a.rank == 0:
        if a.stype != res_st:
            _decline_metal_f64(a, what)
        var out = own(new_tensor(a.shape, a.rank, res_st, a.device))
        var c = own(_cast_any(a, res_st))
        copy_strided_into(out.t, c.t)
        _ = c^  # alive past the launch
        return out^
    # The 8/16-bit integer results scan in int64 and narrow at the end: + and
    # * modulo 2**8 / 2**16 agree with CUDA's wrapping scan in the narrow
    # dtype, once the operand itself was converted to it first.
    var narrow = rdt == DType.int8 or rdt == DType.int16 or rdt == DType.uint8
    if not narrow and not _is_sum_scan_dtype(rdt):
        unsupported(String(what) + " into dtype " + String(rdt))
    if metal and rdt == DType.float64:
        unsupported(String(what) + ": float64 is unavailable on Apple GPUs")
    _decline_metal_f64(a, what)
    var src = own(_cast_any(a, res_st))
    if narrow:
        var wide = own(_cast_any(src.t, ST_INT64))
        src = wide^
    var dense = own_if_new(contiguous(src.t), src.t)
    var work = own(new_tensor(a.shape, a.rank, dense.t.stype, a.device))
    if is_sum and _nn_cumsum_ok(dense.t, dim):
        _cumsum_nn_into(dense.t, dim, work.t)
    else:
        var op: StaticString = "ScanProd"
        if is_sum:
            op = "ScanSum"
        _scan_into(op, dense.t, dim, work.t, 0)
    _ = dense^  # alive past the launch
    _ = src^
    if not narrow:
        return work^
    var r = own(_cast_any(work.t, res_st))
    _ = work^  # alive past the conversion
    return r^


def _cum_default_stype(a: T, dtype_st: Int32) -> Int32:
    """The functional result dtype: `dtype=` if given, else int64 for bool
    and integer operands, else the operand's own."""
    if dtype_st >= 0:
        return dtype_st
    if a.dtype.is_floating_point():
        return a.stype
    return ST_INT64


def _cum(is_sum: Bool, args: Values, rets: Values, what: StaticString) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var st = _cum_default_stype(a, v_dtype_or(args[unsafe_offset=2], -1))
    var out = _cum_result(is_sum, a, v_int(args[unsafe_offset=1]), st, what)
    ret_owned(rets, 0, out)


def _cum_out(
    is_sum: Bool, args: Values, rets: Values, what: StaticString
) raises:
    """`.out`: the result dtype is `dtype=` if given -- which must then equal
    the out's (the structured `set_output` check) -- else the out's own."""
    var a = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=3])
    var st = v_dtype_or(args[unsafe_offset=2], -1)
    if st < 0:
        st = out.stype
    check_out_as(out, st, a)
    var res = _cum_result(is_sum, a, v_int(args[unsafe_offset=1]), st, what)
    if not out.same_shape(res.t):
        resize_out(out, a.shape, a.rank)
    assert_no_internal_overlap(out)
    copy_strided_into(out, res.t)
    _ = res^  # alive past the launch
    ret_ref(rets, 0, out)


def _cum_inplace(
    is_sum: Bool, args: Values, rets: Values, what: StaticString
) raises:
    """`cumsum_` / `cumprod_`: the result dtype is the operand's, and a
    different `dtype=` is the structured in-place dtype error."""
    var a = v_tensor(args[unsafe_offset=0])
    assert_no_internal_overlap(a)
    var st = v_dtype_or(args[unsafe_offset=2], -1)
    if st >= 0 and st != a.stype:
        raise Error(
            "Bad in-place call: input tensor dtype ",
            dtype_name(a.stype),
            " and output tensor dtype ",
            dtype_name(st),
            " should match",
        )
    var res = _cum_result(
        is_sum, a, v_int(args[unsafe_offset=1]), a.stype, what
    )
    copy_strided_into(a, res.t)
    _ = res^  # alive past the launch
    ret_ref(rets, 0, a)


# aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
def op_cumsum(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _cum(True, args, rets, "cumsum")


# aten::cumsum.out(Tensor self, int dim, *, ScalarType? dtype=None,
#   Tensor(a!) out) -> Tensor(a!)
def op_cumsum_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _cum_out(True, args, rets, "cumsum")


# aten::cumsum_(Tensor(a!) self, int dim, *, ScalarType? dtype=None)
#   -> Tensor(a!)
def op_cumsum_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _cum_inplace(True, args, rets, "cumsum")


# aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
def op_cumprod(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _cum(False, args, rets, "cumprod")


# aten::cumprod.out(Tensor self, int dim, *, ScalarType? dtype=None,
#   Tensor(a!) out) -> Tensor(a!)
def op_cumprod_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _cum_out(False, args, rets, "cumprod")


# aten::cumprod_(Tensor(a!) self, int dim, *, ScalarType? dtype=None)
#   -> Tensor(a!)
def op_cumprod_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _cum_inplace(False, args, rets, "cumprod")


# ---------------------------------------------------------------------------
# logcumsumexp
# ---------------------------------------------------------------------------


def _logcumsumexp_result(a: T, dim_in: Int) raises -> Owned:
    """`_logcumsumexp_out_cuda`: a 0-d operand is copied, an empty one has
    nothing to scan; CUDA dispatches floating dtypes only."""
    var dim = _norm_dim(dim_in, a.rank)
    var out = own(new_tensor(a.shape, a.rank, a.stype, a.device))
    # CUDA's shortcuts run before its floating-dtype dispatch: a 0-d operand
    # is copied (`fill_(self)`) and an empty one zeroed, whatever its dtype.
    if a.rank == 0:
        copy_strided_into(out.t, a)
        return out^
    if a.numel == 0:
        return out^
    if not a.dtype.is_floating_point():
        raise Error(
            '"logcumsumexp_cuda" not implemented for \'',
            _torch_type_name(a.dtype),
            "'",
        )
    _decline_metal_f64(a, "logcumsumexp")
    var dense = own_if_new(contiguous(a), a)
    _scan_into("ScanLogSumExp", dense.t, dim, out.t, 0)
    _ = dense^  # alive past the launch
    return out^


def _torch_type_name(dt: DType) -> String:
    """ScalarType names as AT_DISPATCH errors print them."""
    if dt == DType.int64:
        return "Long"
    if dt == DType.int32:
        return "Int"
    if dt == DType.int16:
        return "Short"
    if dt == DType.int8:
        return "Char"
    if dt == DType.uint8:
        return "Byte"
    if dt == DType.bool:
        return "Bool"
    return String(dt)


# aten::_logcumsumexp(Tensor self, int dim) -> Tensor
def op__logcumsumexp(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var out = _logcumsumexp_result(
        v_tensor(args[unsafe_offset=0]), v_int(args[unsafe_offset=1])
    )
    ret_owned(rets, 0, out)


# aten::_logcumsumexp.out(Tensor self, int dim, *, Tensor(a!) out)
#   -> Tensor(a!)
def op__logcumsumexp_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=2])
    check_out(out, a)
    var res = _logcumsumexp_result(a, v_int(args[unsafe_offset=1]))
    if not out.same_shape(res.t):
        resize_out(out, a.shape, a.rank)
    assert_no_internal_overlap(out)
    copy_strided_into(out, res.t)
    _ = res^  # alive past the launch
    ret_ref(rets, 0, out)


# ---------------------------------------------------------------------------
# cummax / cummin helpers
# ---------------------------------------------------------------------------


def _cum_extremum_helper(args: Values, op: StaticString) raises:
    """`cummax_helper_cuda`: `values` / `indices` arrive already resized to
    the operand's shape by ATen's composite `cummax_out`; a non-contiguous
    one is filled through a contiguous temporary."""
    var a = v_tensor(args[unsafe_offset=0])
    var values = v_tensor(args[unsafe_offset=1])
    var indices = v_tensor(args[unsafe_offset=2])
    var dim = _norm_dim(v_int(args[unsafe_offset=3]), a.rank)
    check_out(values, a)
    check_out_as(indices, ST_INT64, a)
    if not values.same_shape(a) or not indices.same_shape(a):
        raise Error(
            String(op), ": values and indices must have the operand's shape"
        )
    var dt = a.dtype
    if not (
        dt.is_floating_point()
        or dt == DType.int64
        or dt == DType.int32
        or dt == DType.int16
        or dt == DType.int8
        or dt == DType.uint8
        or dt == DType.bool
    ):
        unsupported(String(op) + " of dtype " + String(dt))
    _decline_metal_f64(a, op)
    if a.numel == 0:
        return
    assert_no_internal_overlap(values)
    assert_no_internal_overlap(indices)
    var dense = own_if_new(contiguous(a), a)
    var tv = own(new_tensor(a.shape, a.rank, a.stype, a.device))
    var ti = own(new_tensor(a.shape, a.rank, ST_INT64, a.device))
    _scan_into(op, dense.t, dim, tv.t, ti.t.ptr)
    _ = dense^  # alive past the launch
    copy_strided_into(values, tv.t)
    copy_strided_into(indices, ti.t)
    _ = tv^  # alive past the copies
    _ = ti^


# aten::_cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices,
#   int dim) -> ()
def op__cummax_helper(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _cum_extremum_helper(args, "ScanMax")


# aten::_cummin_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices,
#   int dim) -> ()
def op__cummin_helper(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _cum_extremum_helper(args, "ScanMin")


def register_scans(site: Site) raises:
    impl[op__cummax_helper, "_cummax_helper"](site)
    impl[op__cummin_helper, "_cummin_helper"](site)
    impl[op__logcumsumexp, "_logcumsumexp"](site)
    impl[op__logcumsumexp_out, "_logcumsumexp.out"](site)
    impl[op_cumprod, "cumprod"](site)
    impl[op_cumprod_, "cumprod_"](site)
    impl[op_cumprod_out, "cumprod.out"](site)
    impl[op_cumsum, "cumsum"](site)
    impl[op_cumsum_, "cumsum_"](site)
    impl[op_cumsum_out, "cumsum.out"](site)
