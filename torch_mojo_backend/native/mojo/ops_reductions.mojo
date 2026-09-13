"""aten ops: reductions (sum, mean, amax/amin, max/min, the arg-reductions,
any/all, var, the L2 vector norm and cumsum).

Ported from the old Python fast path (`eager_kernels/aten_fast.py`), keeping
its three decisions:

* **dtype gating and promotion.** Each accumulator serves a fixed dtype set
  (`reduce_skeleton.mojo`'s SCALAR_DTYPES / FLOAT_ONLY_DTYPES / TRUTHY_DTYPES);
  an input outside it is declined here rather than compiled into a variant
  that raises. `dtype=` and torch's bool/int -> int64 promotion are a cast
  BEFORE the reduction, never after.
* **the two routes.** One adjacent, ascending reduce-dim interval of a
  contiguous operand on an accelerator is read WHERE IT LIES (the kernels'
  strided-axis path, `_adjacent_reduce_geom`); everything else is permuted so
  the reduce dims become trailing and materialized once, which is the only
  layout `_reduce_spec_geom` accepts.
* **the declines.** Every gate the old path answered with NOT_HANDLED is an
  `unsupported()` here, so the kernel never has to decline its inputs.

The output tensor is allocated with its FINAL torch shape (keepdim applied at
the original dim positions). The kernels validate the output by element count,
contiguity, device and dtype only, so no post-reduction reshape is needed even
on the permuted route.
"""
from std.utils import IndexList

from abi import (
    ST_BOOL,
    ST_INT64,
    TAG_INT,
    TAG_INT_LIST,
    TAG_NONE,
    TAG_SCALAR_INT,
    IntList,
    Owned,
    T,
    Value,
    Values,
    max_dtype,
    new_tensor,
    own,
    release,
    ret_owned,
    ret_ref,
    unsupported,
    v_bool_or,
    v_dtype_or,
    v_f64,
    v_f64_or,
    v_int,
    v_scalar_is_bool,
    v_tensor,
)
from device import ctx_for, ctx_ptr, dev
from kernels import KernelCall
from op_utils import MAX_RANK
from ops_common import cast_into, cast_to, copy_strided_into, resize_out
from registry import Site, impl, op_address_of

# Smallest contiguous inner extent that makes the strided arg-reduction kernel
# (one thread per output column) worth taking over materializing a transposed
# copy. Swept on an H100 PCIe, f32, 4.2M elements reducing dim 0; the crossover
# sits between 8 and 16 lanes. Fitted on that card (see aten_fast.py's
# `_ARG_DIRECT_MIN_INNER`, which this mirrors).
comptime ARG_DIRECT_MIN_INNER = 16

# aten caps the single-block bool full-reduction (AllBool / AnyBool) here; past
# it the generic AllSpec / AnySpec skeleton takes over.
comptime BOOL_FULL_REDUCE_MAX = 1 << 22


# ---------------------------------------------------------------------------
# dtype sets (mirrors of the Mojo-side gates, so a dtype the kernel refuses is
# declined here instead of building a variant that raises)
# ---------------------------------------------------------------------------


def _is_float3(dt: DType) -> Bool:
    """reduce_skeleton FLOAT_ONLY_DTYPES: mean, var, the L2 norm."""
    return dt == DType.float32 or dt == DType.float16 or dt == DType.bfloat16


def _is_row_reduce(dt: DType) -> Bool:
    """reduce_skeleton SCALAR_DTYPES: sum, amax/amin, max/min, the
    arg-reductions."""
    return _is_float3(dt) or dt == DType.int64 or dt == DType.int32


def _is_truthy(dt: DType) -> Bool:
    """reduce_skeleton TRUTHY_DTYPES: the operand dtypes any()/all() accept."""
    return (
        _is_row_reduce(dt)
        or dt == DType.int16
        or dt == DType.int8
        or dt == DType.uint8
        or dt == DType.bool
    )


def _is_castable(dt: DType) -> Bool:
    """aten_fast._CAST_DTYPES: what the cast kernel dispatches on, and so the
    dtype pairs a promotion can go through."""
    return (
        _is_float3(dt)
        or dt == DType.int64
        or dt == DType.int32
        or dt == DType.uint8
        or dt == DType.bool
    )


def _is_sum_dtype(dt: DType) -> Bool:
    """What `fast_aten_sum` accumulates in (int32 is promoted to int64 by
    torch before it gets here, and an explicit `dtype=int32` was declined)."""
    return _is_float3(dt) or dt == DType.int64


def _is_cumsum_dtype(dt: DType, is_cuda: Bool) -> Bool:
    """cumsum_kernels CUMSUM_DTYPES, minus the bf16/f16 entries on a device
    where the fast kernels were never measured (`fast_aten_cumsum`'s is_cuda
    gate; the Mojo dispatch mirrors it)."""
    if dt == DType.float32 or dt == DType.int32 or dt == DType.int64:
        return True
    return is_cuda and (dt == DType.bfloat16 or dt == DType.float16)


# ---------------------------------------------------------------------------
# operands
# ---------------------------------------------------------------------------


struct Operand(Movable):
    """The tensor a reduction actually reads: the caller's own tensor, or a
    promoted / materialized copy this struct owns.

    `owned` is what separates the two: an input handle belongs to torch and
    must never be released, a copy we made must be released on every path.
    """

    var t: T
    var owned: Bool

    def __init__(out self, var t: T, owned: Bool):
        self.t = t^
        self.owned = owned

    def replace(mut self, var t: T, owned: Bool):
        if self.owned:
            release(self.t.h)
        self.t = t^
        self.owned = owned

    def __deinit__(deinit self):
        if self.owned:
            release(self.t.h)


def _borrow(t: T) -> Operand:
    return Operand(t.copy(), False)


def _require_mojo(t: T) raises:
    if not t.on_mojo():
        raise Error("expected a tensor on the mojo device")


def _opt_dtype(v: Value) raises -> Int32:
    """A `ScalarType?` argument: -1 when None."""
    return v_dtype_or(v, Int32(-1))


def _promote(mut op: Operand, stype: Int32) raises:
    """Cast the operand to `stype` first, the way torch's `dtype=` kwarg and
    its bool/int -> int64 rule do (cast-then-reduce, never reduce-then-cast).
    """
    if op.t.stype == stype:
        return
    var target = max_dtype(stype)
    if not _is_castable(op.t.dtype) or not _is_castable(target):
        unsupported(
            "reduction dtype promotion from "
            + String(op.t.dtype)
            + " to "
            + String(target)
        )
    op.replace(cast_to(op.t, stype), True)


# ---------------------------------------------------------------------------
# dim specs and output shapes
# ---------------------------------------------------------------------------


def _norm_dim(d: Int, rank: Int) raises -> Int:
    if rank == 0 or d < -rank or d >= rank:
        unsupported(
            "reduce dim " + String(d) + " out of range for rank " + String(rank)
        )
    return d + rank if d < 0 else d


def _reduce_dims(v: Value, rank: Int, empty_is_all: Bool) raises -> List[Int]:
    """Sorted, unique, normalized reduce dims (aten_fast._norm_reduce_dims).

    `None` always reduces every dim. An EMPTY dim list reduces every dim for
    sum/mean/amax/amin/var, and nothing for any.dims/all.dims. A duplicate or
    out-of-range dim is declined. An empty result means the caller declines:
    the reduce bridges reject a zero-length dim spec.
    """
    var dims = List[Int]()
    if v.tag == TAG_NONE:
        for d in range(rank):
            dims.append(d)
        return dims^
    if v.tag == TAG_INT or v.tag == TAG_SCALAR_INT:
        dims.append(_norm_dim(v_int(v), rank))
        return dims^
    if v.tag != TAG_INT_LIST:
        raise Error("expected an int or int[] dim argument, got tag ", v.tag)
    var given = IntList(v)
    if len(given) == 0:
        if empty_is_all:
            for d in range(rank):
                dims.append(d)
        return dims^
    var seen = InlineArray[Bool, MAX_RANK](fill=False)
    for i in range(len(given)):
        var d = _norm_dim(given[i], rank)
        if seen[d]:
            unsupported("duplicate reduce dim " + String(d))
        seen[d] = True
    for d in range(rank):
        if seen[d]:
            dims.append(d)
    return dims^


def _reduced_shape(
    t: T,
    dims: List[Int],
    keepdim: Bool,
    mut shape: IndexList[MAX_RANK],
    mut rank: Int,
):
    """The reduction's torch output shape, leading-padded: keepdim leaves a 1
    at every reduced position, otherwise the kept dims pack together."""
    var is_red = InlineArray[Bool, MAX_RANK](fill=False)
    for d in dims:
        is_red[d] = True
    shape = IndexList[MAX_RANK](1)
    rank = t.rank if keepdim else t.rank - len(dims)
    var pad = MAX_RANK - rank
    var w = 0
    for d in range(t.rank):
        if is_red[d]:
            if keepdim:
                shape[pad + w] = 1
                w += 1
        else:
            shape[pad + w] = t.dim(d)
            w += 1


def _is_adjacent(dims: List[Int]) -> Bool:
    for k in range(len(dims)):
        if dims[k] != dims[0] + k:
            return False
    return True


def _is_trailing(dims: List[Int], rank: Int) -> Bool:
    return len(dims) > 0 and dims[0] == rank - len(dims) and _is_adjacent(dims)


def _trailing_dims(rank: Int, n: Int) -> List[Int]:
    var dims = List[Int](capacity=n)
    for k in range(rank - n, rank):
        dims.append(k)
    return dims^


def _on_accelerator(device: Int) raises -> Bool:
    return not dev(device)[].is_cpu


def _middle_direct_ok(a: T, dims: List[Int]) raises -> Bool:
    """Whether the scalar reductions read this layout in place.

    Mirrors aten_fast._reduce_middle_direct_ok: a contiguous operand of a
    supported dtype (checked by the caller) reduced over one adjacent,
    ascending, NON-trailing dim interval on an accelerator. Trailing dims are
    excluded because the ordinary rows/cols path already owns them.
    """
    if not a.contig or not _on_accelerator(a.device):
        return False
    if _is_trailing(dims, a.rank):
        return False
    return _is_adjacent(dims)


def _arg_direct_ok(a: T, dims: List[Int]) raises -> Bool:
    """The same question for the (value, index) kernels, which additionally
    need enough contiguous inner elements for neighbouring lanes to coalesce
    (aten_fast._arg_strided_direct_ok)."""
    if not _middle_direct_ok(a, dims):
        return False
    var inner = 1
    for d in range(dims[len(dims) - 1] + 1, a.rank):
        inner *= a.dim(d)
    return inner >= ARG_DIRECT_MIN_INNER


def _permuted_contiguous(t: T, dims: List[Int]) raises -> T:
    """A fresh contiguous copy of `t` with the reduce dims moved to the end
    (kept dims ascending, then reduce dims ascending) — the one layout
    `_reduce_spec_geom` accepts. The caller owns the handle.

    The source of the copy is `t`'s own storage read through permuted
    shape/strides: the strided-copy kernel reads only the data pointer (which
    already carries the storage offset) and the two stride vectors, so
    rewriting those fields of a `T` copy is the whole permutation — no torch
    view object, no second allocation.
    """
    var is_red = InlineArray[Bool, MAX_RANK](fill=False)
    for d in dims:
        is_red[d] = True
    var shape = IndexList[MAX_RANK](1)
    var strides = IndexList[MAX_RANK](0)
    var pad = MAX_RANK - t.rank
    var w = 0
    for d in range(t.rank):
        if not is_red[d]:
            shape[pad + w] = t.dim(d)
            strides[pad + w] = t.stride(d)
            w += 1
    for d in dims:
        shape[pad + w] = t.dim(d)
        strides[pad + w] = t.stride(d)
        w += 1
    var src = t.copy()
    src.shape = shape
    src.strides = strides
    src.contig = False
    var out = own(new_tensor(shape, t.rank, t.stype, t.device))
    copy_strided_into(out.t, src)
    return out.take()


def _ready_operand(
    a: T, mut dims: List[Int], arg_route: Bool
) raises -> Operand:
    """The operand and dims the kernel is called with: `a` untouched when it
    is already in a layout the kernel reads, else a permuted contiguous copy
    whose reduce dims are the trailing ones.

    Two layouts are read in place: trailing reduce dims of a contiguous
    operand (the ordinary rows/cols kernels) and, on an accelerator, an
    adjacent ascending interval anywhere else (the strided-axis kernels).
    `arg_route` picks the second gate's arg-reduction form, which adds the
    coalescing floor its column kernel needs.
    """
    var direct = _arg_direct_ok(a, dims) if arg_route else _middle_direct_ok(
        a, dims
    )
    if direct or (a.contig and _is_trailing(dims, a.rank)):
        return _borrow(a)
    var n = len(dims)
    var materialized = _permuted_contiguous(a, dims)
    dims = _trailing_dims(a.rank, n)
    return Operand(materialized^, True)


# ---------------------------------------------------------------------------
# launches
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


def _reduce_into(
    family: StaticString,
    op: StaticString,
    a: T,
    var dims: List[Int],
    keepdim: Bool,
    dst: T,
    with_correction: Bool,
    correction: Float64,
) raises:
    """One scalar reduction into the preallocated `dst`.

    Slot list of `_rowred_spec_into_go` / `_var_spec_into_go`: operand spec,
    reduce-dim tuple, keepdim, the accumulator's extra payload (var's
    correction), output spec.
    """
    _one_device(a, dst)
    var src = _ready_operand(a, dims, False)
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall(String(family), String(op))
    call.arg_dtype(0, src.t.dtype)
    call.out_dtype(dst.dtype)
    call.spec(src.t.spec(cp))
    call.tuple(dims)
    call.int(1 if keepdim else 0)
    if with_correction:
        call.f64(correction)
    call.spec(dst.spec(cp))
    call.run()
    _ = ctx
    _ = src^


def _arg_reduce_into(
    family: StaticString,
    op: StaticString,
    a: T,
    var dims: List[Int],
    keepdim: Bool,
    dst: T,
) raises:
    """argmax / argmin: same slots as a scalar reduction, but the in-place
    route has the extra coalescing floor."""
    _one_device(a, dst)
    var src = _ready_operand(a, dims, True)
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall(String(family), String(op))
    call.arg_dtype(0, src.t.dtype)
    call.out_dtype(dst.dtype)
    call.spec(src.t.spec(cp))
    call.tuple(dims)
    call.int(1 if keepdim else 0)
    call.spec(dst.spec(cp))
    call.run()
    _ = ctx
    _ = src^


def _min_dim_into(
    a: T, var dims: List[Int], keepdim: Bool, dst_v: T, dst_i: T
) raises:
    """min.dim in one call: `_min_dim_spec_into_go` fills both preallocated
    outputs, so values and indices come out of a single pass with torch's
    first-min-wins tie rule and its NaN propagation."""
    _one_device(a, dst_v)
    _one_device(a, dst_i)
    var src = _ready_operand(a, dims, True)
    var ctx = ctx_for(dst_v.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("reduction_ops", "MinDimSpec")
    call.arg_dtype(0, src.t.dtype)
    call.out_dtype_i(0, dst_v.dtype)
    call.out_dtype_i(1, dst_i.dtype)
    call.spec(src.t.spec(cp))
    call.tuple(dims)
    call.int(1 if keepdim else 0)
    call.spec(dst_v.spec(cp))
    call.spec(dst_i.spec(cp))
    call.run()
    _ = ctx
    _ = src^


def _scalar_reduction(
    family: StaticString,
    op: StaticString,
    a: T,
    dims: List[Int],
    keepdim: Bool,
    out_stype: Int32,
    with_correction: Bool,
    correction: Float64,
) raises -> Owned:
    """Allocate the output for one scalar reduction and fill it."""
    var shape = IndexList[MAX_RANK](1)
    var rank = 0
    _reduced_shape(a, dims, keepdim, shape, rank)
    var out = own(new_tensor(shape, rank, out_stype, a.device))
    _reduce_into(
        family, op, a, dims.copy(), keepdim, out.t, with_correction, correction
    )
    return out^


# ---------------------------------------------------------------------------
# out= variants
# ---------------------------------------------------------------------------


def _can_cast(frm: DType, to: DType) raises -> Bool:
    """torch's `at::canCast`: no float -> integral, and nothing but bool ->
    bool (the categories type promotion is built on)."""
    if frm.is_floating_point() and not to.is_floating_point():
        return False
    if frm != DType.bool and to == DType.bool:
        return False
    return True


def _check_out_dtype(
    op_name: StaticString,
    policy: StaticString,
    result: DType,
    target: DType,
) raises:
    """The old `_out_variant` dtype policies, one per registration site:
    `exact` (linalg_vector_norm.out), `bool_or_uint8` (any.out) and the
    default `safe_cast` (mean.out)."""
    var ok = _can_cast(result, target)  # the default `safe_cast` policy
    if policy == "exact":
        ok = result == target
    elif policy == "bool_or_uint8":
        ok = target == DType.bool or target == DType.uint8
    if not ok:
        raise Error(
            op_name,
            ": result type ",
            result,
            " can't be cast to the desired output type ",
            target,
        )


def _copy_result_into(dst: T, src: T) raises:
    """`out[...] = src` with a dtype cast, for any `out` layout. `src` is a
    freshly allocated contiguous result, so the cast kernel's contiguity
    requirement is already met."""
    _one_device(src, dst)
    if not dst.same_shape(src):
        raise Error(
            "out= tensor of rank ",
            dst.rank,
            " and ",
            dst.numel,
            " elements does not match the result's rank ",
            src.rank,
            " / ",
            src.numel,
            " elements",
        )
    if dst.stype == src.stype:
        copy_strided_into(dst, src)
        return
    if dst.contig:
        cast_into(dst, src)
        return
    var tmp = own(cast_to(src, dst.stype))
    copy_strided_into(dst, tmp.t)


def _out_ready(dst: T, a: T, stype: Int32, numel: Int) -> Bool:
    """Whether the kernel can write straight into the caller's tensor
    (`_check_into_sized`: element count, contiguity, device and dtype)."""
    return (
        dst.contig
        and dst.stype == stype
        and dst.numel == numel
        and dst.device == a.device
        and dst.on_mojo()
    )


def _shape_numel(shape: IndexList[MAX_RANK], rank: Int) -> Int:
    var n = 1
    for i in range(MAX_RANK - rank, MAX_RANK):
        n *= shape[i]
    return n


def _shape_matches(t: T, shape: IndexList[MAX_RANK], rank: Int) -> Bool:
    if t.rank != rank:
        return False
    for i in range(rank):
        if t.dim(i) != shape[MAX_RANK - rank + i]:
            return False
    return True


def _scalar_reduction_out(
    family: StaticString,
    op: StaticString,
    op_name: StaticString,
    policy: StaticString,
    a: T,
    dims: List[Int],
    keepdim: Bool,
    out_stype: Int32,
    mut dst: T,
) raises:
    """Compute into `out` when its shape, dtype, layout and device already
    match; otherwise compute into a fresh tensor and copy across.

    An `out=` whose SHAPE differs from the reduced shape is resized first
    (`resize_output`, the same rule every ATen out= op follows). Matching the
    element count alone is not enough: the copy below reads the source
    through the destination's extents, so a (2,3) result poured into a (3,2)
    out would walk off the end of the source.
    """
    _one_device(a, dst)
    _check_out_dtype(op_name, policy, max_dtype(out_stype), dst.dtype)
    var shape = IndexList[MAX_RANK](1)
    var rank = 0
    _reduced_shape(a, dims, keepdim, shape, rank)
    var numel = _shape_numel(shape, rank)
    if not _shape_matches(dst, shape, rank):
        resize_out(dst, shape, rank)
    if _out_ready(dst, a, out_stype, numel):
        _reduce_into(family, op, a, dims.copy(), keepdim, dst, False, 0.0)
        return
    var tmp = own(new_tensor(shape, rank, out_stype, a.device))
    _reduce_into(family, op, a, dims.copy(), keepdim, tmp.t, False, 0.0)
    _copy_result_into(dst, tmp.t)


# ---------------------------------------------------------------------------
# sum
# ---------------------------------------------------------------------------


# aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *,
#   ScalarType? dtype=None) -> Tensor
def op_sum_dim_intlist(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a)
    var src = _borrow(a)
    var want = _opt_dtype(args[unsafe_offset=3])
    if want >= 0:
        _promote(src, want)
    elif not src.t.dtype.is_floating_point():
        # torch promotes bool / sub-int64 integer sums to int64.
        _promote(src, ST_INT64)
    if not _is_sum_dtype(src.t.dtype):
        unsupported("sum of dtype " + String(src.t.dtype))
    var dims = _reduce_dims(args[unsafe_offset=1], src.t.rank, True)
    if len(dims) == 0:
        unsupported("sum with no reduce dim (a rank-0 operand)")
    var keepdim = v_bool_or(args[unsafe_offset=2], False)
    var out = _scalar_reduction(
        "reduction_ops",
        "SumSpec",
        src.t,
        dims,
        keepdim,
        src.t.stype,
        False,
        0.0,
    )
    ret_owned(rets, 0, out)
    _ = src^


# ---------------------------------------------------------------------------
# mean
# ---------------------------------------------------------------------------


def _mean(
    a: T, dim_v: Value, keepdim: Bool, dtype_v: Value, rets: Values
) raises:
    _require_mojo(a)
    var src = _borrow(a)
    var want = _opt_dtype(dtype_v)
    if want >= 0:
        if not _is_float3(max_dtype(want)):
            unsupported("mean with dtype=" + String(max_dtype(want)))
        _promote(src, want)
    if not _is_float3(src.t.dtype):
        unsupported("mean of dtype " + String(src.t.dtype))
    var dims = _reduce_dims(dim_v, src.t.rank, True)
    if len(dims) == 0:
        unsupported("mean with no reduce dim (a rank-0 operand)")
    var out = _scalar_reduction(
        "nn_ops", "MeanSpec", src.t, dims, keepdim, src.t.stype, False, 0.0
    )
    ret_owned(rets, 0, out)
    _ = src^


# aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
def op_mean(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _mean(
        v_tensor(args[unsafe_offset=0]),
        Value(TAG_NONE, 0, 0, 0),
        False,
        args[unsafe_offset=1],
        rets,
    )


# aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *,
#   ScalarType? dtype=None) -> Tensor
def op_mean_dim(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _mean(
        v_tensor(args[unsafe_offset=0]),
        args[unsafe_offset=1],
        v_bool_or(args[unsafe_offset=2], False),
        args[unsafe_offset=3],
        rets,
    )


# aten::mean.out(Tensor self, int[1]? dim, bool keepdim=False, *,
#   ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
def op_mean_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=4])
    _require_mojo(a)
    _require_mojo(out)
    var src = _borrow(a)
    var want = _opt_dtype(args[unsafe_offset=3])
    if want >= 0:
        if not _is_float3(max_dtype(want)):
            unsupported("mean with dtype=" + String(max_dtype(want)))
        _promote(src, want)
    if not _is_float3(src.t.dtype):
        unsupported("mean of dtype " + String(src.t.dtype))
    var dims = _reduce_dims(args[unsafe_offset=1], src.t.rank, True)
    if len(dims) == 0:
        unsupported("mean with no reduce dim (a rank-0 operand)")
    _scalar_reduction_out(
        "nn_ops",
        "MeanSpec",
        "aten::mean.out",
        "safe_cast",
        src.t,
        dims,
        v_bool_or(args[unsafe_offset=2], False),
        src.t.stype,
        out,
    )
    ret_ref(rets, 0, out)
    _ = src^


# ---------------------------------------------------------------------------
# amax / amin / max / min (values only)
# ---------------------------------------------------------------------------


def _refuse_empty_extremum(op: StaticString, t: T, dims: List[Int]) raises:
    """amax/amin/max/min over a zero-length axis: torch refuses ("Expected
    reduction dim to have non-zero size") and so does the accumulator
    (`errors_on_empty_axis`). Declining on the host gives the caller the
    actionable NotImplementedError the old fast path gave, rather than the
    kernel's own message. A reduction with no OUTPUTS is an error for nobody."""
    var is_red = InlineArray[Bool, MAX_RANK](fill=False)
    var extent = 1
    for d in dims:
        is_red[d] = True
        extent *= t.dim(d)
    if extent != 0:
        return
    var outputs = 1
    for d in range(t.rank):
        if not is_red[d]:
            outputs *= t.dim(d)
    if outputs > 0:
        unsupported(
            String(op) + " over a reduce dim of size 0 (torch refuses it too)"
        )


def _amax_amin(op: StaticString, args: Values, rets: Values) raises:
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a)
    if not _is_row_reduce(a.dtype):
        unsupported(String(op) + " of dtype " + String(a.dtype))
    var dims = _reduce_dims(args[unsafe_offset=1], a.rank, True)
    if len(dims) == 0:
        unsupported("amax/amin with no reduce dim (a rank-0 operand)")
    _refuse_empty_extremum(op, a, dims)
    var out = _scalar_reduction(
        "reduction_ops",
        op,
        a,
        dims,
        v_bool_or(args[unsafe_offset=2], False),
        a.stype,
        False,
        0.0,
    )
    ret_owned(rets, 0, out)


# aten::amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
def op_amax(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _amax_amin("AmaxSpec", args, rets)


# aten::amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
def op_amin(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _amax_amin("AminSpec", args, rets)


def _full_extremum(
    family: StaticString, op: StaticString, args: Values, rets: Values
) raises:
    """max(Tensor) / min(Tensor): the values-only full reduction."""
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a)
    if not _is_row_reduce(a.dtype):
        unsupported(String(op) + " of dtype " + String(a.dtype))
    if a.rank == 0:
        unsupported("max()/min() of a rank-0 tensor")
    var dims = _trailing_dims(a.rank, a.rank)
    _refuse_empty_extremum(op, a, dims)
    var out = _scalar_reduction(family, op, a, dims, False, a.stype, False, 0.0)
    ret_owned(rets, 0, out)


# aten::max(Tensor self) -> Tensor
def op_max(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _full_extremum("nn_ops", "MaxSpec", args, rets)


# aten::min(Tensor self) -> Tensor
def op_min(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _full_extremum("reduction_ops", "AminSpec", args, rets)


# ---------------------------------------------------------------------------
# min.dim (values + indices)
# ---------------------------------------------------------------------------


def _min_dim_gate(a: T, dim: Int) raises -> List[Int]:
    _require_mojo(a)
    if not _is_row_reduce(a.dtype):
        unsupported("min.dim of dtype " + String(a.dtype))
    if a.rank == 0:
        unsupported("min.dim of a rank-0 tensor")
    if a.numel == 0:
        unsupported("min.dim of an empty tensor")
    var dims = List[Int]()
    dims.append(_norm_dim(dim, a.rank))
    return dims^


def _min_dim(a: T, dim: Int, keepdim: Bool) raises -> Tuple[Owned, Owned]:
    var dims = _min_dim_gate(a, dim)
    var shape = IndexList[MAX_RANK](1)
    var rank = 0
    _reduced_shape(a, dims, keepdim, shape, rank)
    var values = own(new_tensor(shape, rank, a.stype, a.device))
    var indices = own(new_tensor(shape, rank, ST_INT64, a.device))
    _min_dim_into(a, dims.copy(), keepdim, values.t, indices.t)
    return (values^, indices^)


# aten::min.dim(Tensor self, int dim, bool keepdim=False)
#   -> (Tensor values, Tensor indices)
def op_min_dim(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var pair = _min_dim(
        v_tensor(args[unsafe_offset=0]),
        v_int(args[unsafe_offset=1]),
        v_bool_or(args[unsafe_offset=2], False),
    )
    ret_owned(rets, 0, pair[0])
    ret_owned(rets, 1, pair[1])


# aten::min.dim_min(Tensor self, int dim, bool keepdim=False, *,
#   Tensor(a!) min, Tensor(b!) min_indices)
#   -> (Tensor(a!) values, Tensor(b!) indices)
def op_min_dim_min(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var keepdim = v_bool_or(args[unsafe_offset=2], False)
    var out_v = v_tensor(args[unsafe_offset=3])
    var out_i = v_tensor(args[unsafe_offset=4])
    _require_mojo(out_v)
    _require_mojo(out_i)
    var dims = _min_dim_gate(a, v_int(args[unsafe_offset=1]))
    var shape = IndexList[MAX_RANK](1)
    var rank = 0
    _reduced_shape(a, dims, keepdim, shape, rank)
    var numel = _shape_numel(shape, rank)
    _one_device(a, out_v)
    _one_device(a, out_i)
    if not _shape_matches(out_v, shape, rank):
        resize_out(out_v, shape, rank)
    if not _shape_matches(out_i, shape, rank):
        resize_out(out_i, shape, rank)
    if _out_ready(out_v, a, a.stype, numel) and _out_ready(
        out_i, a, ST_INT64, numel
    ):
        _min_dim_into(a, dims.copy(), keepdim, out_v, out_i)
    else:
        var values = own(new_tensor(shape, rank, a.stype, a.device))
        var indices = own(new_tensor(shape, rank, ST_INT64, a.device))
        _min_dim_into(a, dims.copy(), keepdim, values.t, indices.t)
        _copy_result_into(out_v, values.t)
        _copy_result_into(out_i, indices.t)
    ret_ref(rets, 0, out_v)
    ret_ref(rets, 1, out_i)


# ---------------------------------------------------------------------------
# argmax / argmin
# ---------------------------------------------------------------------------


def _argreduce(
    family: StaticString, op: StaticString, args: Values, rets: Values
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a)
    if not _is_row_reduce(a.dtype):
        unsupported(String(op) + " of dtype " + String(a.dtype))
    if a.numel == 0:
        unsupported("argmax/argmin of an empty tensor")
    var dims = _reduce_dims(args[unsafe_offset=1], a.rank, True)
    if len(dims) == 0:
        unsupported("argmax/argmin of a rank-0 tensor")
    var keepdim = v_bool_or(args[unsafe_offset=2], False)
    var shape = IndexList[MAX_RANK](1)
    var rank = 0
    _reduced_shape(a, dims, keepdim, shape, rank)
    var out = own(new_tensor(shape, rank, ST_INT64, a.device))
    _arg_reduce_into(family, op, a, dims.copy(), keepdim, out.t)
    ret_owned(rets, 0, out)


# aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
def op_argmax(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _argreduce("nn_ops", "ArgmaxSpec", args, rets)


# aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
def op_argmin(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _argreduce("reduction_ops", "ArgminSpec", args, rets)


# ---------------------------------------------------------------------------
# any / all
# ---------------------------------------------------------------------------


def _bool_full_reduce(op: StaticString, a: T) raises -> Owned:
    """The single-block bool full reduction (nn_ops AllBool / AnyBool): one
    scalar out of a contiguous bool buffer, below aten's 4.2M-element cap.
    Slots are raw pointers, not specs."""
    var src = Operand(a.copy(), False)
    if not a.contig:
        src.replace(
            _permuted_contiguous(a, _trailing_dims(a.rank, a.rank)), True
        )
    var out = own(new_tensor(IndexList[MAX_RANK](1), 0, ST_BOOL, a.device))
    var ctx = ctx_for(a.device)
    var call = KernelCall("nn_ops", String(op))
    call.arg_dtype(0, src.t.dtype)
    call.out_dtype(out.t.dtype)
    call.int(out.t.ptr)
    call.int(src.t.ptr)
    call.int(src.t.numel)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx
    _ = src^
    return out^


def _any_all(
    spec: StaticString,
    bool_op: StaticString,
    a: T,
    dim_v: Value,
    keepdim: Bool,
    rets: Values,
) raises:
    _require_mojo(a)
    if not _is_truthy(a.dtype):
        unsupported(String(spec) + " of dtype " + String(a.dtype))
    if (
        dim_v.tag == TAG_NONE
        and not keepdim
        and a.dtype == DType.bool
        and a.numel > 0
        and a.numel < BOOL_FULL_REDUCE_MAX
    ):
        var scalar = _bool_full_reduce(bool_op, a)
        ret_owned(rets, 0, scalar)
        return
    # any.dims / all.dims: an EXPLICIT empty dim list reduces nothing.
    var dims = _reduce_dims(dim_v, a.rank, False)
    if len(dims) == 0:
        unsupported("any/all with an empty dim list")
    var out = _scalar_reduction(
        "reduction_ops", spec, a, dims, keepdim, ST_BOOL, False, 0.0
    )
    ret_owned(rets, 0, out)


def _none_value() -> Value:
    return Value(TAG_NONE, 0, 0, 0)


# aten::all(Tensor self) -> Tensor
def op_all(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _any_all(
        "AllSpec",
        "AllBool",
        v_tensor(args[unsafe_offset=0]),
        _none_value(),
        False,
        rets,
    )


# aten::all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
# aten::all.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor
def op_all_dim(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _any_all(
        "AllSpec",
        "AllBool",
        v_tensor(args[unsafe_offset=0]),
        args[unsafe_offset=1],
        v_bool_or(args[unsafe_offset=2], False),
        rets,
    )


# aten::any(Tensor self) -> Tensor
def op_any(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _any_all(
        "AnySpec",
        "AnyBool",
        v_tensor(args[unsafe_offset=0]),
        _none_value(),
        False,
        rets,
    )


# aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
# aten::any.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor
def op_any_dim(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _any_all(
        "AnySpec",
        "AnyBool",
        v_tensor(args[unsafe_offset=0]),
        args[unsafe_offset=1],
        v_bool_or(args[unsafe_offset=2], False),
        rets,
    )


# aten::any.out(Tensor self, int dim, bool keepdim=False, *,
#   Tensor(a!) out) -> Tensor(a!)
def op_any_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=3])
    _require_mojo(a)
    _require_mojo(out)
    if not _is_truthy(a.dtype):
        unsupported("any of dtype " + String(a.dtype))
    var dims = _reduce_dims(args[unsafe_offset=1], a.rank, False)
    if len(dims) == 0:
        unsupported("any with an empty dim list")
    _scalar_reduction_out(
        "reduction_ops",
        "AnySpec",
        "aten::any.out",
        "bool_or_uint8",
        a,
        dims,
        v_bool_or(args[unsafe_offset=2], False),
        ST_BOOL,
        out,
    )
    ret_ref(rets, 0, out)


# ---------------------------------------------------------------------------
# var
# ---------------------------------------------------------------------------


# aten::var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None,
#   bool keepdim=False) -> Tensor
def op_var_correction(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a)
    if not _is_float3(a.dtype):
        unsupported("var of dtype " + String(a.dtype))
    if a.numel == 0:
        unsupported("var of an empty tensor")
    var dims = _reduce_dims(args[unsafe_offset=1], a.rank, True)
    if len(dims) == 0:
        unsupported("var with no reduce dim (a rank-0 operand)")
    var correction = v_f64_or(args[unsafe_offset=2], 1.0)
    var out = _scalar_reduction(
        "reduction_ops",
        "VarSpec",
        a,
        dims,
        v_bool_or(args[unsafe_offset=3], False),
        a.stype,
        True,
        correction,
    )
    ret_owned(rets, 0, out)


# ---------------------------------------------------------------------------
# linalg_vector_norm (ord=2 only: one pass, the root folded into the finalize)
# ---------------------------------------------------------------------------


def _vector_norm_operand(ord_v: Value, dtype_v: Value, mut src: Operand) raises:
    """The `ord` / `dtype=` gates shared by the functional and out= forms:
    only the ord-2 one-pass accumulator exists, and `dtype=` selects the
    accumulation type by casting first (clip_grad_norm_ asks for float32)."""
    if v_scalar_is_bool(ord_v) or v_f64(ord_v) != 2.0:
        unsupported("linalg_vector_norm with ord != 2")
    var want = _opt_dtype(dtype_v)
    if want >= 0:
        if not _is_float3(max_dtype(want)):
            unsupported(
                "linalg_vector_norm with dtype=" + String(max_dtype(want))
            )
        _promote(src, want)
    if not _is_float3(src.t.dtype):
        unsupported("linalg_vector_norm of dtype " + String(src.t.dtype))


# aten::linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None,
#   bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
def op_linalg_vector_norm(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a)
    var src = _borrow(a)
    _vector_norm_operand(args[unsafe_offset=1], args[unsafe_offset=4], src)
    var dims = _reduce_dims(args[unsafe_offset=2], src.t.rank, True)
    if len(dims) == 0:
        unsupported("linalg_vector_norm with no reduce dim (a rank-0 operand)")
    var out = _scalar_reduction(
        "reduction_ops",
        "NormSpec",
        src.t,
        dims,
        v_bool_or(args[unsafe_offset=3], False),
        src.t.stype,
        False,
        0.0,
    )
    ret_owned(rets, 0, out)
    _ = src^


# aten::linalg_vector_norm.out(Tensor self, Scalar ord=2, int[1]? dim=None,
#   bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
def op_linalg_vector_norm_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=5])
    _require_mojo(a)
    _require_mojo(out)
    var src = _borrow(a)
    _vector_norm_operand(args[unsafe_offset=1], args[unsafe_offset=4], src)
    var dims = _reduce_dims(args[unsafe_offset=2], src.t.rank, True)
    if len(dims) == 0:
        unsupported("linalg_vector_norm with no reduce dim (a rank-0 operand)")
    _scalar_reduction_out(
        "reduction_ops",
        "NormSpec",
        "aten::linalg_vector_norm.out",
        "exact",
        src.t,
        dims,
        v_bool_or(args[unsafe_offset=3], False),
        src.t.stype,
        out,
    )
    ret_ref(rets, 0, out)
    _ = src^


# ---------------------------------------------------------------------------
# cumsum
# ---------------------------------------------------------------------------


# aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
def op_cumsum(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a)
    if a.numel == 0 or a.rank == 0:
        unsupported("cumsum of an empty or rank-0 tensor")
    # The fast cumsum family (block.prefix_sum INNER, the long-line workspace
    # route, OUTER dim=0) was only ever MEASURED on NVIDIA; anything else gets
    # exactly the pre-existing surface (int64/int32/float32, trailing dim), and
    # the Mojo-side dispatch gates the same way.
    var is_cuda = dev(a.device)[].api == "cuda"
    var src = _borrow(a)
    var want = _opt_dtype(args[unsafe_offset=2])
    if want >= 0:
        if not _is_cumsum_dtype(max_dtype(want), is_cuda):
            unsupported("cumsum with dtype=" + String(max_dtype(want)))
        _promote(src, want)
    elif not src.t.dtype.is_floating_point():
        # torch promotes bool / sub-int64 integer cumsum to int64.
        _promote(src, ST_INT64)
    if not _is_cumsum_dtype(src.t.dtype, is_cuda):
        unsupported("cumsum of dtype " + String(src.t.dtype))
    var dim = _norm_dim(v_int(args[unsafe_offset=1]), src.t.rank)
    var rank = src.t.rank
    if dim != rank - 1 and not (is_cuda and rank == 2 and dim == 0):
        unsupported(
            "cumsum over dim "
            + String(dim)
            + " of a rank-"
            + String(rank)
            + " tensor"
        )
    if not src.t.contig:
        src.replace(
            _permuted_contiguous(src.t, _trailing_dims(rank, rank)), True
        )
    var out = own(new_tensor(src.t.shape, rank, src.t.stype, src.t.device))
    var ctx = ctx_for(out.t.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("nn_ops", "CumsumSpec")
    call.arg_dtype(0, src.t.dtype)
    call.out_dtype(out.t.dtype)
    call.spec(src.t.spec(cp))
    call.int(dim)
    call.spec(out.t.spec(cp))
    call.run()
    _ = ctx
    ret_owned(rets, 0, out)
    _ = src^


def register_reductions(site: Site) raises:
    impl[op_all, "all"](site)
    impl[op_all_dim, "all.dim"](site)
    impl[op_all_dim, "all.dims"](site)
    impl[op_amax, "amax"](site)
    impl[op_amin, "amin"](site)
    impl[op_any, "any"](site)
    impl[op_any_dim, "any.dim"](site)
    impl[op_any_dim, "any.dims"](site)
    impl[op_any_out, "any.out"](site)
    impl[op_argmax, "argmax"](site)
    impl[op_argmin, "argmin"](site)
    impl[op_cumsum, "cumsum"](site)
    impl[op_linalg_vector_norm, "linalg_vector_norm"](site)
    impl[op_linalg_vector_norm_out, "linalg_vector_norm.out"](site)
    impl[op_max, "max"](site)
    impl[op_mean, "mean"](site)
    impl[op_mean_dim, "mean.dim"](site)
    impl[op_mean_out, "mean.out"](site)
    impl[op_min, "min"](site)
    impl[op_min_dim, "min.dim"](site)
    impl[op_min_dim_min, "min.dim_min"](site)
    impl[op_sum_dim_intlist, "sum.dim_IntList"](site)
    impl[op_var_correction, "var.correction"](site)


@export
def tmb_op_address() abi("C") -> Int:
    """Entry of this file's one-op extension: the address of the op the
    TMB_OP define selected (registry.mojo)."""
    return op_address_of[register_reductions]()
