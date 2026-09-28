"""ATen ops: reductions (sum, nansum, mean, amax/amin, max/min, the
arg-reductions, any/all, count_nonzero, var, the vector norm (any ord),
cumsum, and sort/topk).

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
from std.utils.numerics import max_or_inf, min_or_neg_inf, nan

from tmb.backend.abi import (
    ST_BOOL,
    ST_INT32,
    ST_INT64,
    ST_UINT8,
    TAG_INT,
    TAG_INT_LIST,
    TAG_NONE,
    TAG_SCALAR_INT,
    IntList,
    Owned,
    T,
    Value,
    Values,
    dtype_code,
    dtype_itemsize,
    dtype_name,
    index_error,
    max_dtype,
    new_scalar,
    new_tensor,
    own,
    own_if_new,
    release,
    ret_owned,
    ret_ref,
    torch_dtype,
    unsupported,
    v_bool_or,
    v_dtype_or,
    v_f64_or,
    v_int,
    v_int_or,
    v_scalar_is_bool,
    v_tensor,
)
from tmb.backend.device import ctx_for, ctx_ptr, dev
from tmb.backend.kernel_call import KernelCall
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.common import (
    assert_no_internal_overlap,
    assert_no_overlap,
    cast_into,
    cast_to,
    check_out,
    contiguous,
    copy_strided_into,
    elementwise_direct,
    fill_value,
    is_cast_dtype_on,
    one_device,
    resize_out,
)
from tmb.backend.registry import Site, impl

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


def _check_extremum_dtype(a: T, op: StaticString) raises:
    """reduce_skeleton EXTREMUM_DTYPES: amax/amin and full max/min; float64
    is declined on Apple GPUs, which have none."""
    if a.dtype == DType.float64:
        if dev(a.device)[].api == "metal":
            unsupported(String(op) + ": float64 is unavailable on Apple GPUs")
        return
    if not _is_row_reduce(a.dtype):
        unsupported(String(op) + " of dtype " + String(a.dtype))


def _is_truthy(dt: DType) -> Bool:
    """reduce_skeleton TRUTHY_DTYPES: the operand dtypes any()/all() accept."""
    return (
        _is_row_reduce(dt)
        or dt == DType.float64
        or dt == DType.int16
        or dt == DType.int8
        or dt == DType.uint8
        or dt == DType.bool
    )


def _is_float_or_double(dt: DType) -> Bool:
    """_is_float3 plus float64: mean's and the vector-norm's dtype gate, NOT
    var's (its separate moments kernel, entry.mojo's FLOAT_DTYPES, has no
    float64 specialization)."""
    return _is_float3(dt) or dt == DType.float64


def _is_castable(dt: DType, t: T) raises -> Bool:
    """aten_fast._CAST_DTYPES: what the cast kernel dispatches on, and so the
    dtype pairs a promotion can go through -- exactly `is_cast_dtype_on`
    (ops/common.mojo), which `_promote` (sum/nansum/mean/prod/cumsum/the
    vector-norm dtype= path -- every caller of `_promote`) uses through this
    one check rather than needing its own device guard."""
    return is_cast_dtype_on(dt, t)


def _is_sum_dtype(dt: DType) -> Bool:
    """What `fast_aten_sum` accumulates in (int32 is promoted to int64 by
    torch before it gets here, and an explicit `dtype=int32` was declined)."""
    return _is_float3(dt) or dt == DType.float64 or dt == DType.int64


def _decline_metal_float64(t: T, op: StaticString) raises:
    """float64 has no Apple GPU execution: every dtype gate that admits it
    below (sum/nansum/prod, mean, norm, any/all) declines it here the same
    way `_check_extremum_dtype` already does for amax/amin/max/min."""
    _decline_metal_float64_dtype(t.dtype, t, op)


def _decline_metal_float64_dtype(dt: DType, t: T, op: StaticString) raises:
    """Same decline, ALSO checked against a dtype not yet on `t` (a `dtype=`
    target about to be promoted into): declining before the promotion means
    an explicit `dtype=torch.float64` on an Apple GPU gets a clean
    `unsupported()` instead of the cast kernel's own raw Error.

    Checks `t`'s CURRENT dtype too, not just `dt`: a float64 self promoted
    to a non-float64 target (`mean(x_float64, dtype=torch.float32)`, and the
    same shape for sum/nansum/prod) still cast_to()s through the float64
    side of that pair, which `_cast` raises on Apple GPUs precisely because
    it's float64, regardless of which side. Declining on either dtype here
    means the call site above needn't separately check `t.dtype`."""
    if (dt == DType.float64 or t.dtype == DType.float64) and dev(
        t.device
    )[].api == "metal":
        unsupported(String(op) + ": float64 is unavailable on Apple GPUs")


def _is_cumsum_dtype(dt: DType, fast_ok: Bool) -> Bool:
    """cumsum_kernels CUMSUM_DTYPES, minus the bf16/f16 entries on a device
    where the bf16/f16 route was never measured. Measured correct on NVIDIA
    (H100, the fast block.prefix_sum kernels) and on AMD MI300A and Apple M4 (gfx942/Metal, the
    portable one-thread-per-line fallback -- see `_cumsum_inner_into` /
    `_cumsum_outer_into` in tmb/kernels/nn/entry.mojo)."""
    if dt == DType.float32 or dt == DType.int32 or dt == DType.int64:
        return True
    return fast_ok and (dt == DType.bfloat16 or dt == DType.float16)


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
    if not _is_castable(op.t.dtype, op.t) or not _is_castable(target, op.t):
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


def _dim_range_message(d: Int, ndim: Int) -> String:
    """torch's own wording, verbatim (`maybe_wrap_dim_slow`, WrapDimMinimal.cpp).
    """
    return (
        "Dimension out of range (expected to be in range of ["
        + String(-ndim)
        + ", "
        + String(ndim - 1)
        + "], but got "
        + String(d)
        + ")"
    )


def _norm_dim(d: Int, rank: Int) raises -> Int:
    """torch's `maybe_wrap_dim`, including its 0-d exception: a rank-0
    operand is wrapped as if it were a 1-d tensor of size 1, so dim 0 / -1
    are valid (and normalize to 0) while anything else is out of range.
    Raises IndexError, matching stock CUDA (`TORCH_CHECK_INDEX`)."""
    var ndim = max(rank, 1)
    if d < -ndim or d >= ndim:
        index_error(_dim_range_message(d, ndim))
    return d + ndim if d < 0 else d


def _reduce_dims(v: Value, rank: Int, empty_is_all: Bool) raises -> List[Int]:
    """Sorted, unique, normalized reduce dims (aten_fast._norm_reduce_dims).

    `None` always reduces every dim. An EMPTY dim list reduces every dim for
    sum/mean/amax/amin/var, and nothing for any.dims/all.dims. A duplicate or
    out-of-range dim is declined. A rank-0 operand has no dim to mark (its
    single element is already the whole reduction) so it always returns
    empty, once every given dim has been validated against `_norm_dim`'s 0-d
    exception ({-1, 0}) -- `empty_is_all` makes no difference there, since
    "reduce everything" and "reduce nothing" coincide when there is nothing
    to reduce over.
    """
    var dims = List[Int]()
    if v.tag == TAG_NONE:
        for d in range(rank):
            dims.append(d)
        return dims^
    if v.tag == TAG_INT or v.tag == TAG_SCALAR_INT:
        var d = _norm_dim(v_int(v), rank)
        if rank > 0:
            dims.append(d)
        return dims^
    if v.tag != TAG_INT_LIST:
        raise Error("expected an int or int[] dim argument, got tag ", v.tag)
    var given = IntList(v)
    if len(given) == 0:
        if empty_is_all:
            for d in range(rank):
                dims.append(d)
        return dims^
    if rank == 0:
        # Every valid entry normalizes to the same (only) dim, 0: a second
        # one is necessarily a duplicate, same as torch's own refusal
        # (plain RuntimeError, matching `dim_list_to_bitset`'s TORCH_CHECK).
        var seen0 = False
        for i in range(len(given)):
            _ = _norm_dim(given[i], rank)
            if seen0:
                raise Error("dim 0 appears multiple times in the list of dims")
            seen0 = True
        return dims^
    var seen = Array[Bool, MAX_RANK](fill=False)
    for i in range(len(given)):
        var d = _norm_dim(given[i], rank)
        if seen[d]:
            raise Error(
                "dim "
                + String(d)
                + " appears multiple times in the list of dims"
            )
        seen[d] = True
    for d in range(rank):
        if seen[d]:
            dims.append(d)
    return dims^


def _reduce_dim_single(d: Int, rank: Int) raises -> List[Int]:
    """One explicit reduce dim (min.dim, prod.dim_int/.int_out) as a dims
    list: empty for a rank-0 operand once `_norm_dim` has validated it."""
    var norm = _norm_dim(d, rank)
    var dims = List[Int]()
    if rank > 0:
        dims.append(norm)
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
    var is_red = Array[Bool, MAX_RANK](fill=False)
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


def _middle_direct_ok(a: T, dims: List[Int]) raises -> Bool:
    """Whether the scalar reductions read this layout in place.

    Mirrors aten_fast._reduce_middle_direct_ok: a contiguous operand of a
    supported dtype (checked by the caller) reduced over one adjacent,
    ascending, NON-trailing dim interval on an accelerator. Trailing dims are
    excluded because the ordinary rows/cols path already owns them.
    """
    if not a.contig:
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
    var is_red = Array[Bool, MAX_RANK](fill=False)
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
    coalescing floor its column kernel needs. An empty `dims` (a rank-0
    operand, the only case `_reduce_dims` produces one) has nothing to check
    or permute -- `a` is already ready -- and skips straight past
    `_arg_direct_ok`'s `dims[-1]`, which an empty list can't index.
    """
    if len(dims) == 0:
        return _borrow(a)
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
    correction, the general vector norm's ord), output spec.
    """
    one_device(a, dst)
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
    one_device(a, dst)
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
    one_device(a, dst_v)
    one_device(a, dst_i)
    var src = _ready_operand(a, dims, True)
    var ctx = ctx_for(dst_v.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("reduction", "MinDimSpec")
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
    one_device(src, dst)
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
    _ = tmp^  # alive past the launch


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


def _decline_aliasing_out(op_name: StaticString, a: T, dst: T) raises:
    """`out=` sharing storage with the input is declined outright, before any
    resize is even considered. Resizing an `out=` that shares `a`'s storage
    is not one well-defined case: the resize hook can grow the shared
    allocation, which reallocates the block `a`'s already-cached tensor info
    points at -- verified on stock CUDA torch that this has no single
    behavior worth reproducing: `torch.max(x, out=x[x.numel():])` is fine,
    but `torch.max(x, out=x)` (`resize_output` shrinking `self`'s own
    metadata out from under the reduction that is about to read it) comes
    back with a silently WRONG answer on real CUDA, not merely a stale
    pointer. Shared with the sibling out= overload fixes on other reduce ops
    (see #565) so both land on the identical check."""
    var a_storage = a.storage_ptr()
    if a_storage != 0 and a_storage == dst.storage_ptr():
        unsupported(String(op_name) + ": out= aliasing the input")


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
    with_correction: Bool = False,
    correction: Float64 = 0.0,
) raises:
    """Compute into `out` when its shape, dtype, layout and device already
    match; otherwise compute into a fresh tensor and copy across.

    An `out=` whose SHAPE differs from the reduced shape is resized first
    (`resize_output`, the same rule every ATen out= op follows). Matching the
    element count alone is not enough: the copy below reads the source
    through the destination's extents, so a (2,3) result poured into a (3,2)
    out would walk off the end of the source.

    `assert_no_internal_overlap` (checked once here for every out= reduction,
    after the resize decision since a resize that leaves the shape alone --
    e.g. an already-right-shaped but `.expand()`ed `out=` -- is exactly the
    case it must catch) declines an `out=` that repeats elements: several
    logical output positions would alias one physical address, so distinct
    reduction results written there would silently collapse into whichever
    write lands last. `_decline_aliasing_out`, a separate concern, catches
    `out=` sharing storage with the INPUT.
    """
    one_device(a, dst)
    _decline_aliasing_out(op_name, a, dst)
    _check_out_dtype(op_name, policy, max_dtype(out_stype), dst.dtype)
    var shape = IndexList[MAX_RANK](1)
    var rank = 0
    _reduced_shape(a, dims, keepdim, shape, rank)
    var numel = _shape_numel(shape, rank)
    if not _shape_matches(dst, shape, rank):
        resize_out(dst, shape, rank)
    assert_no_internal_overlap(dst)
    if _out_ready(dst, a, out_stype, numel):
        _reduce_into(
            family,
            op,
            a,
            dims.copy(),
            keepdim,
            dst,
            with_correction,
            correction,
        )
        return
    var tmp = own(new_tensor(shape, rank, out_stype, a.device))
    _reduce_into(
        family,
        op,
        a,
        dims.copy(),
        keepdim,
        tmp.t,
        with_correction,
        correction,
    )
    _copy_result_into(dst, tmp.t)
    _ = tmp^  # alive past the launch


def _out_reduce_dtype(
    dtype_v: Value, dst: T, op_name: StaticString
) raises -> DType:
    """The dtype an out= reduction with an optional `dtype=` computes and
    rounds into: `dtype_v` if given -- which torch requires to equal `dst`'s
    dtype exactly ("Expected out tensor to have dtype X, but got dtype Y
    instead", the structured-kernel `set_output` check every `.out`/
    `.dtype_out` reduction shares) -- otherwise `dst`'s own dtype
    (ReduceOps.cpp: `ScalarType dtype = result.scalar_type();`), which is
    authoritative regardless of the input's own dtype: unlike the no-`out=`
    path, an integer input never overrides it with the bool/sub-int64 ->
    int64 default. Callers still validate the resolved dtype is one their
    op/kernel supports (mean/norm: float only; sum: float or int64)."""
    var want = _opt_dtype(dtype_v)
    if want < 0:
        return dst.dtype
    var target = max_dtype(want)
    _check_out_dtype(op_name, "exact", target, dst.dtype)
    return target


def _promote_for_out_reduction(
    mut src: Operand, target: DType, op_name: StaticString
) raises:
    """Cast `src` to `target` before reducing: both mean and sum round every
    element to `target` first (their CUDA kernels build the reduction
    directly from it), then accumulate in float32 via their own `acc_dtype`
    -- mirroring CUDA, not CPU torch's separate half-precision-avoiding
    `mean_out` path.

    Declines float64 on an Apple GPU UNCONDITIONALLY, before the `src.t.dtype
    != target` check below: when they're already equal (both float64, no
    cast needed) `_promote` -- and so its own device-aware `_is_castable`
    guard -- is never reached, so this is the one place that decision has to
    be made for every out= reduction (sum.IntList_out, mean.out/dtype_out,
    prod.int_out, nansum.out)."""
    _decline_metal_float64_dtype(target, src.t, op_name)
    if src.t.dtype != target:
        _promote(src, torch_dtype(target))


# ---------------------------------------------------------------------------
# sum
# ---------------------------------------------------------------------------


def _sum(
    a: T, dim_v: Value, keepdim: Bool, dtype_v: Value, rets: Values
) raises:
    _require_mojo(a)
    var src = _borrow(a)
    var want = _opt_dtype(dtype_v)
    if want >= 0:
        _promote(src, want)
    elif not src.t.dtype.is_floating_point():
        # torch promotes bool / sub-int64 integer sums to int64.
        _promote(src, ST_INT64)
    if not _is_sum_dtype(src.t.dtype):
        unsupported("sum of dtype " + String(src.t.dtype))
    _decline_metal_float64(src.t, "sum")
    var dims = _reduce_dims(dim_v, src.t.rank, True)
    if len(dims) == 0 and src.t.rank != 0:
        unsupported("sum with no reduce dim (a rank-0 operand)")
    var out = _scalar_reduction(
        "reduction",
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


# aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
def op_sum(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _sum(
        v_tensor(args[unsafe_offset=0]),
        Value(TAG_NONE, 0, 0, 0),
        False,
        args[unsafe_offset=1],
        rets,
    )


# aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *,
#   ScalarType? dtype=None) -> Tensor
def op_sum_dim_intlist(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _sum(
        v_tensor(args[unsafe_offset=0]),
        args[unsafe_offset=1],
        v_bool_or(args[unsafe_offset=2], False),
        args[unsafe_offset=3],
        rets,
    )


# aten::sum.IntList_out(Tensor self, int[1]? dim, bool keepdim=False, *,
#   ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
def op_sum_intlist_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=4])
    _require_mojo(a)
    _require_mojo(out)
    var src = _borrow(a)
    # `out` always exists here, so its dtype is the compute dtype for ANY
    # self (float or int) with no explicit dtype= -- the bool/sub-int64 ->
    # int64 default (`_sum`, above) only applies when there is no out tensor
    # to take a dtype from.
    var target = _out_reduce_dtype(
        args[unsafe_offset=3], out, "aten::sum.IntList_out"
    )
    if not _is_sum_dtype(target):
        unsupported("sum with dtype=" + String(target))
    _promote_for_out_reduction(src, target, "aten::sum.IntList_out")
    var dims = _reduce_dims(args[unsafe_offset=1], src.t.rank, True)
    if len(dims) == 0 and src.t.rank != 0:
        unsupported("sum with no reduce dim (a rank-0 operand)")
    _scalar_reduction_out(
        "reduction",
        "SumSpec",
        "aten::sum.IntList_out",
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
# nansum
# ---------------------------------------------------------------------------


def _nansum_prep(
    mut src: Operand, dtype_v: Value, dim_v: Value
) raises -> List[Int]:
    """Prep for `nansum` (the no-`out=` overload): dtype promotion, the dtype
    gate, and the reduce-dim list.

    Mirrors `sum`'s own promotion (`_promote` / `_is_sum_dtype`) exactly for
    integral/bool inputs -- they carry no NaN, so nansum and sum must agree on
    them -- except an explicit INTEGRAL `dtype=` on a FLOATING input: torch
    zeroes NaN before that cast (`nan_to_num` then `sum`, ReduceOps.cpp), and
    this backend has no `nan_to_num` kernel to do that ahead of `_promote`'s
    plain cast, which would truncate NaN to an arbitrary integer instead.
    Declined rather than risking a silently wrong value.
    """
    var want = _opt_dtype(dtype_v)
    if want >= 0:
        var target = max_dtype(want)
        if src.t.dtype.is_floating_point() and not target.is_floating_point():
            unsupported(
                "nansum with dtype=" + String(target) + " from a floating input"
            )
        _promote(src, want)
    elif not src.t.dtype.is_floating_point():
        _promote(src, ST_INT64)
    if not _is_sum_dtype(src.t.dtype):
        unsupported("nansum of dtype " + String(src.t.dtype))
    _decline_metal_float64(src.t, "nansum")
    var dims = _reduce_dims(dim_v, src.t.rank, True)
    if len(dims) == 0 and src.t.rank != 0:
        unsupported("nansum with no reduce dim (a rank-0 operand)")
    return dims^


def _nansum_out_target(
    dtype_v: Value, src_dtype: DType, dst: T, op_name: StaticString
) raises -> DType:
    """nansum.out's compute dtype: `_out_reduce_dtype` (`dtype_v` if given,
    else `dst`'s own -- so dtype=None computes and rounds into the caller's
    out dtype, not the default's bool/sub-int64 -> int64 rule), gated to
    `_is_sum_dtype` and declined outright when the ORIGINAL operand is
    floating and the target is not: torch either `nan_to_num`s this
    combination explicitly (an explicit integral `dtype=`) or fails at its
    own dispatch (an integral `out` with no `dtype=`, since its floating
    nansum kernel has no integral specialization). This backend has no
    `nan_to_num` kernel, so both are declined rather than risking a silently
    wrong truncation.
    """
    var target = _out_reduce_dtype(dtype_v, dst, op_name)
    if not _is_sum_dtype(target):
        unsupported("nansum with dtype=" + String(target))
    if src_dtype.is_floating_point() and not target.is_floating_point():
        unsupported(
            "nansum with dtype=" + String(target) + " from a floating input"
        )
    return target


# aten::nansum(Tensor self, int[1]? dim=None, bool keepdim=False, *,
#   ScalarType? dtype=None) -> Tensor
def op_nansum(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a)
    var src = _borrow(a)
    var dims = _nansum_prep(src, args[unsafe_offset=3], args[unsafe_offset=1])
    var keepdim = v_bool_or(args[unsafe_offset=2], False)
    var out = _scalar_reduction(
        "reduction",
        "NanSumSpec",
        src.t,
        dims,
        keepdim,
        src.t.stype,
        False,
        0.0,
    )
    ret_owned(rets, 0, out)
    _ = src^


# aten::nansum.out(Tensor self, int[1]? dim=None, bool keepdim=False, *,
#   ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
def op_nansum_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=4])
    _require_mojo(a)
    _require_mojo(out)
    var src = _borrow(a)
    var target = _nansum_out_target(
        args[unsafe_offset=3], src.t.dtype, out, "aten::nansum.out"
    )
    _promote_for_out_reduction(src, target, "aten::nansum.out")
    var dims = _reduce_dims(args[unsafe_offset=1], src.t.rank, True)
    if len(dims) == 0 and src.t.rank != 0:
        unsupported("nansum with no reduce dim (a rank-0 operand)")
    _scalar_reduction_out(
        "reduction",
        "NanSumSpec",
        "aten::nansum.out",
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
# mean
# ---------------------------------------------------------------------------


def _mean(
    a: T, dim_v: Value, keepdim: Bool, dtype_v: Value, rets: Values
) raises:
    _require_mojo(a)
    var src = _borrow(a)
    var want = _opt_dtype(dtype_v)
    if want >= 0:
        if not _is_float_or_double(max_dtype(want)):
            unsupported("mean with dtype=" + String(max_dtype(want)))
        _promote(src, want)
    if not _is_float_or_double(src.t.dtype):
        unsupported("mean of dtype " + String(src.t.dtype))
    _decline_metal_float64(src.t, "mean")
    var dims = _reduce_dims(dim_v, src.t.rank, True)
    if len(dims) == 0 and src.t.rank != 0:
        unsupported("mean with no reduce dim (a rank-0 operand)")
    var out = _scalar_reduction(
        "nn", "MeanSpec", src.t, dims, keepdim, src.t.stype, False, 0.0
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


def _mean_out(
    a: T,
    dim_v: Value,
    keepdim: Bool,
    dtype_v: Value,
    op_name: StaticString,
    mut dst: T,
    rets: Values,
) raises:
    _require_mojo(a)
    _require_mojo(dst)
    var src = _borrow(a)
    var target = _out_reduce_dtype(dtype_v, dst, op_name)
    if _opt_dtype(dtype_v) < 0 and not _is_float_or_double(src.t.dtype):
        # No explicit dtype=: torch requires self itself to be float/complex.
        # An explicit dtype= bypasses this -- self is cast to it below, so an
        # int64 self with dtype=torch.float32 is valid.
        unsupported("mean of dtype " + String(src.t.dtype))
    if not _is_float_or_double(target):
        unsupported("mean with dtype=" + String(target))
    _promote_for_out_reduction(src, target, op_name)
    var dims = _reduce_dims(dim_v, src.t.rank, True)
    if len(dims) == 0 and src.t.rank != 0:
        unsupported("mean with no reduce dim (a rank-0 operand)")
    _scalar_reduction_out(
        "nn",
        "MeanSpec",
        op_name,
        "safe_cast",
        src.t,
        dims,
        keepdim,
        src.t.stype,
        dst,
    )
    ret_ref(rets, 0, dst)
    _ = src^


# aten::mean.out(Tensor self, int[1]? dim, bool keepdim=False, *,
#   ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
def op_mean_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var out = v_tensor(args[unsafe_offset=4])
    _mean_out(
        v_tensor(args[unsafe_offset=0]),
        args[unsafe_offset=1],
        v_bool_or(args[unsafe_offset=2], False),
        args[unsafe_offset=3],
        "aten::mean.out",
        out,
        rets,
    )


# aten::mean.dtype_out(Tensor self, *, ScalarType? dtype=None,
#   Tensor(a!) out) -> Tensor(a!)
def op_mean_dtype_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    # CompositeExplicitAutograd forwards to mean.out with dim=[] (full
    # reduce), keepdim=False: aten/src/ATen/native/ReduceOps.cpp mean_dtype_out.
    var out = v_tensor(args[unsafe_offset=2])
    _mean_out(
        v_tensor(args[unsafe_offset=0]),
        Value(TAG_NONE, 0, 0, 0),
        False,
        args[unsafe_offset=1],
        "aten::mean.dtype_out",
        out,
        rets,
    )


# ---------------------------------------------------------------------------
# amax / amin / max / min (values only)
# ---------------------------------------------------------------------------


def _refuse_empty_extremum(op: StaticString, t: T, dims: List[Int]) raises:
    """amax/amin/max/min over a zero-length axis: torch refuses ("Expected
    reduction dim to have non-zero size") and so does the accumulator
    (`errors_on_empty_axis`). Declining on the host gives the caller the
    actionable NotImplementedError the old fast path gave, rather than the
    kernel's own message. Torch refuses this EVEN WHEN THE OUTPUT ITSELF IS
    EMPTY (e.g. amin(empty(0, 0), dim=1) still raises), so the output count
    plays no part here."""
    var extent = 1
    for d in dims:
        extent *= t.dim(d)
    if extent == 0:
        unsupported(
            String(op) + " over a reduce dim of size 0 (torch refuses it too)"
        )


def _amax_amin(op: StaticString, args: Values, rets: Values) raises:
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a)
    _check_extremum_dtype(a, op)
    var dims = _reduce_dims(args[unsafe_offset=1], a.rank, True)
    if len(dims) == 0 and a.rank != 0:
        unsupported("amax/amin with no reduce dim (a rank-0 operand)")
    _refuse_empty_extremum(op, a, dims)
    var out = _scalar_reduction(
        "reduction",
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


def _amax_amin_out(
    op: StaticString, op_name: StaticString, args: Values, rets: Values
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=3])
    _require_mojo(a)
    _require_mojo(out)
    _check_extremum_dtype(a, op)
    var dims = _reduce_dims(args[unsafe_offset=1], a.rank, True)
    if len(dims) == 0 and a.rank != 0:
        unsupported("amax/amin with no reduce dim (a rank-0 operand)")
    _refuse_empty_extremum(op, a, dims)
    _scalar_reduction_out(
        "reduction",
        op,
        op_name,
        "exact",  # torch's amax/amin meta: out dtype must equal input dtype
        a,
        dims,
        v_bool_or(args[unsafe_offset=2], False),
        a.stype,
        out,
    )
    ret_ref(rets, 0, out)


# aten::amax.out(Tensor self, int[1] dim=[], bool keepdim=False, *,
#   Tensor(a!) out) -> Tensor(a!)
def op_amax_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _amax_amin_out("AmaxSpec", "aten::amax.out", args, rets)


# aten::amin.out(Tensor self, int[1] dim=[], bool keepdim=False, *,
#   Tensor(a!) out) -> Tensor(a!)
def op_amin_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _amax_amin_out("AminSpec", "aten::amin.out", args, rets)


def _full_extremum_dims(op: StaticString, a: T) raises -> List[Int]:
    """Shared gate for max(Tensor)/min(Tensor) and min.unary_out: dtype and
    empty-reduce-dim checks, then every dim is reduced. `_trailing_dims` of a
    rank-0 operand is already empty, so `_refuse_empty_extremum`'s size
    product over zero dims stays 1 (never the size-0 case torch refuses)."""
    _require_mojo(a)
    _check_extremum_dtype(a, op)
    var dims = _trailing_dims(a.rank, a.rank)
    _refuse_empty_extremum(op, a, dims)
    return dims^


def _full_extremum(
    family: StaticString, op: StaticString, args: Values, rets: Values
) raises:
    """max(Tensor) / min(Tensor): the values-only full reduction."""
    var a = v_tensor(args[unsafe_offset=0])
    var dims = _full_extremum_dims(op, a)
    var out = _scalar_reduction(family, op, a, dims, False, a.stype, False, 0.0)
    ret_owned(rets, 0, out)


# aten::max(Tensor self) -> Tensor
def op_max(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _full_extremum("nn", "MaxSpec", args, rets)


# aten::min(Tensor self) -> Tensor
def op_min(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _full_extremum("reduction", "AminSpec", args, rets)


def _full_extremum_out(
    family: StaticString,
    op: StaticString,
    op_name: StaticString,
    args: Values,
    rets: Values,
) raises:
    """max.unary_out / min.unary_out: `_full_extremum_dims`'s gate, then an
    `exact`-dtype `out=` -- stock CUDA's `make_reduction` requires the output
    dtype to equal the input's exactly (verified on real CUDA: an int64
    input with a float32 out raises "provided dtype must match dtype of
    result"), unlike mean.out/any.out's `canCast` policy."""
    var a = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=1])
    _require_mojo(out)
    var dims = _full_extremum_dims(op, a)
    _scalar_reduction_out(
        family,
        op,
        op_name,
        "exact",
        a,
        dims,
        False,
        a.stype,
        out,
    )
    ret_ref(rets, 0, out)


# aten::max.unary_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_max_unary_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _full_extremum_out("nn", "MaxSpec", "aten::max.unary_out", args, rets)


# aten::min.unary_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_min_unary_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _full_extremum_out(
        "reduction", "AminSpec", "aten::min.unary_out", args, rets
    )


# ---------------------------------------------------------------------------
# min.dim (values + indices)
# ---------------------------------------------------------------------------


def _min_dim_gate(a: T, dim: Int) raises -> List[Int]:
    _require_mojo(a)
    if not _is_row_reduce(a.dtype):
        unsupported("min.dim of dtype " + String(a.dtype))
    if a.numel == 0:
        unsupported("min.dim of an empty tensor")
    return _reduce_dim_single(dim, a.rank)


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
    one_device(a, out_v)
    one_device(a, out_i)
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
        _ = values^  # alive past the launch
        _copy_result_into(out_i, indices.t)
        _ = indices^  # alive past the launch
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
    if len(dims) == 0 and a.rank != 0:
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
    _argreduce("nn", "ArgmaxSpec", args, rets)


# aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
def op_argmin(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _argreduce("reduction", "ArgminSpec", args, rets)


# ---------------------------------------------------------------------------
# any / all
# ---------------------------------------------------------------------------


def _bool_full_reduce(op: StaticString, a: T) raises -> Owned:
    """The single-block bool full reduction (nn AllBool / AnyBool): one
    scalar out of a contiguous bool buffer, below aten's 4.2M-element cap.
    Slots are raw pointers, not specs."""
    var src = Operand(a.copy(), False)
    if not a.contig:
        src.replace(
            _permuted_contiguous(a, _trailing_dims(a.rank, a.rank)), True
        )
    var out = own(new_tensor(IndexList[MAX_RANK](1), 0, ST_BOOL, a.device))
    var ctx = ctx_for(a.device)
    var call = KernelCall("nn", String(op))
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
    _decline_metal_float64(a, spec)
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
    if len(dims) == 0 and a.rank != 0:
        unsupported("any/all with an empty dim list")
    var out = _scalar_reduction(
        "reduction", spec, a, dims, keepdim, _any_all_out_stype(a), False, 0.0
    )
    ret_owned(rets, 0, out)


def _any_all_out_stype(a: T) -> Int32:
    """The dtype AnyOp/AllOp's kernel actually writes: torch's uint8
    compatibility keeps a uint8 input's dtype instead of narrowing to bool
    (ReduceOps.cpp, Note "[all, any : uint8 compatibility]")."""
    return ST_UINT8 if a.dtype == DType.uint8 else ST_BOOL


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


# aten::any.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_any_all_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=1])
    _require_mojo(a)
    _require_mojo(out)
    _check_truthy_dtype("any", a)
    _scalar_reduction_out(
        "reduction",
        "AnySpec",
        "aten::any.all_out",
        "bool_or_uint8",
        a,
        _reduce_dims(_none_value(), a.rank, False),
        False,
        _any_all_out_stype(a),
        out,
    )
    ret_ref(rets, 0, out)


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


def _check_truthy_dtype(name: StaticString, a: T) raises:
    if not _is_truthy(a.dtype):
        unsupported(String(name) + " of dtype " + String(a.dtype))
    _decline_metal_float64(a, name)


def _truthy_reduce_dims(
    name: StaticString, a: T, dim_v: Value
) raises -> List[Int]:
    """Shared any/all `.out` checks for the overloads that take a dim
    argument: truthy dtype, non-empty dim list. The full-reduction *_out
    overloads (any.all_out/all.all_out) use `_check_truthy_dtype` alone --
    their dims come from reducing a rank-0 input over zero axes, a
    legitimate no-op, not a user-requested empty dim list to decline."""
    _check_truthy_dtype(name, a)
    var dims = _reduce_dims(dim_v, a.rank, False)
    if len(dims) == 0 and a.rank != 0:
        unsupported(String(name) + " with an empty dim list")
    return dims^


# aten::any.out(Tensor self, int dim, bool keepdim=False, *,
#   Tensor(a!) out) -> Tensor(a!)
# aten::any.dims_out(Tensor self, int[]? dim=None, bool keepdim=False, *,
#   Tensor(a!) out) -> Tensor(a!)
# `_reduce_dims` reads the dim arg's Value tag (plain int for any.out, an
# optional int list for any.dims_out), so one body serves both overloads.
def op_any_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=3])
    _require_mojo(a)
    _require_mojo(out)
    var dims = _truthy_reduce_dims("any", a, args[unsafe_offset=1])
    _scalar_reduction_out(
        "reduction",
        "AnySpec",
        "aten::any.out",
        "bool_or_uint8",
        a,
        dims,
        v_bool_or(args[unsafe_offset=2], False),
        _any_all_out_stype(a),
        out,
    )
    ret_ref(rets, 0, out)


# aten::all.out(Tensor self, int dim, bool keepdim=False, *,
#   Tensor(a!) out) -> Tensor(a!)
# aten::all.dims_out(Tensor self, int[]? dim=None, bool keepdim=False, *,
#   Tensor(a!) out) -> Tensor(a!)
# One body serves both, exactly like op_any_out above.
def op_all_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=3])
    _require_mojo(a)
    _require_mojo(out)
    var dims = _truthy_reduce_dims("all", a, args[unsafe_offset=1])
    _scalar_reduction_out(
        "reduction",
        "AllSpec",
        "aten::all.out",
        "bool_or_uint8",
        a,
        dims,
        v_bool_or(args[unsafe_offset=2], False),
        _any_all_out_stype(a),
        out,
    )
    ret_ref(rets, 0, out)


# aten::all.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
def op_all_all_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=1])
    _require_mojo(a)
    _require_mojo(out)
    _check_truthy_dtype("all", a)
    _scalar_reduction_out(
        "reduction",
        "AllSpec",
        "aten::all.all_out",
        "bool_or_uint8",
        a,
        _reduce_dims(_none_value(), a.rank, False),
        False,
        _any_all_out_stype(a),
        out,
    )
    ret_ref(rets, 0, out)


# ---------------------------------------------------------------------------
# count_nonzero
# ---------------------------------------------------------------------------


# aten::count_nonzero.dim_IntList(Tensor self, int[] dim) -> Tensor
def op_count_nonzero(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a)
    if not _is_truthy(a.dtype):
        unsupported("count_nonzero of dtype " + String(a.dtype))
    _decline_metal_float64(a, "count_nonzero")
    # An explicit empty dim list reduces every dim (unlike any.dims/all.dims):
    # `count_nonzero.default(self, dim=None)` redispatches here with `dim=[]`.
    var dims = _reduce_dims(args[unsafe_offset=1], a.rank, True)
    if len(dims) == 0 and a.rank != 0:
        unsupported("count_nonzero with no reduce dim (a rank-0 operand)")
    var out = _scalar_reduction(
        "reduction", "CountNonzeroSpec", a, dims, False, ST_INT64, False, 0.0
    )
    ret_owned(rets, 0, out)


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
    if len(dims) == 0 and a.rank != 0:
        unsupported("var with no reduce dim (a rank-0 operand)")
    var correction = v_f64_or(args[unsafe_offset=2], 1.0)
    var out = _scalar_reduction(
        "reduction",
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
# linalg_vector_norm / norm: ord=2 (sum of squares), ord=1 (sum of |x|),
# ord=+inf (max of |x|), ord=-inf (min of |x|), ord=0 (count of nonzero) and
# any other ord p (sum of |x|^p, then ^(1/p): `NormPOp`, p passed at run time)
# share everything but the accumulator. `norm`'s six
# legacy overloads are torch's own redispatch onto this op (`impl_func_norm`
# in ATen's ReduceOps.cpp: p=None -> 2, dim=[] -> every dim), so they share
# every helper below with `linalg_vector_norm`.
# ---------------------------------------------------------------------------


def _vector_norm_spec(ord_v: Value) raises -> StaticString:
    """The kernel op token for `ord`, shared by every overload of both ops:
    a missing `ord` (legacy `norm`'s `p=None`) means 2, same as torch's
    `impl_func_norm`. The ords CUDA gives a dedicated kernel
    (`norm_kernel_cuda_impl`: 0, 1, 2, inf, -inf) keep theirs; every other
    ord is `NormPSpec`."""
    if v_scalar_is_bool(ord_v):
        unsupported("vector_norm with an unsupported ord")
    var ord_f = v_f64_or(ord_v, 2.0)
    if ord_f == 2.0:
        return "NormSpec"
    if ord_f == 1.0:
        return "NormL1Spec"
    if ord_f == max_or_inf[DType.float64]():
        return "NormInfSpec"
    if ord_f == min_or_neg_inf[DType.float64]():
        return "NormNegInfSpec"
    if ord_f == 0.0:
        return "NormL0Spec"
    return "NormPSpec"


def _decline_normp_float64(
    op: StaticString, op_label: StaticString, src_dtype: DType, dtype_v: Value
) raises:
    """NormPOp (general ord=p) has a fixed float32 accumulator, so a float64
    self or `dtype=torch.float64` would silently compute in less precision
    than asked for; declined on every device, before `_vector_norm_operand`
    promotes anything."""
    if op != "NormPSpec":
        return
    if src_dtype == DType.float64:
        unsupported(
            String(op_label) + " of dtype float64 with a general ord (p)"
        )
    var want = _opt_dtype(dtype_v)
    if want >= 0 and max_dtype(want) == DType.float64:
        unsupported(
            String(op_label) + " with dtype=float64 and a general ord (p)"
        )


def _vector_norm_operand(
    op_label: StaticString, dtype_v: Value, mut src: Operand
) raises:
    """The `dtype=` gate shared by every ord and every overload of both ops:
    `dtype=` selects the accumulation type by casting first (clip_grad_norm_
    asks for float32).

    torch validates the INPUT's own dtype unconditionally
    (`checkFloatingOrComplex` in `TORCH_META_FUNC(linalg_vector_norm)`)
    before ever looking at `dtype=`, so an integer/bool input is declined
    even when `dtype=float32` would make the cast well-defined; and `dtype=`
    may only WIDEN (`check_linalg_norm_dtype`'s
    `promoteTypes(self_dtype, dtype) == dtype`): a float32 input with
    `dtype=float16` is declined, and (verified on real CUDA) so is a float64
    input with `dtype=float32` -- float64 sits ABOVE float32 in this
    promotion order, so it is a valid target from any of the other three
    floats but never a valid source down to float32.
    """
    if not _is_float_or_double(src.t.dtype):
        unsupported(String(op_label) + " of dtype " + String(src.t.dtype))
    var want = _opt_dtype(dtype_v)
    if want >= 0:
        var target = max_dtype(want)
        if not _is_float_or_double(target):
            unsupported(String(op_label) + " with dtype=" + String(target))
        var is_widen = (
            target == src.t.dtype
            or target == DType.float64
            or (target == DType.float32 and src.t.dtype != DType.float64)
        )
        if not is_widen:
            unsupported(
                String(op_label)
                + ": the dtype of the input ("
                + String(src.t.dtype)
                + ") can't convert without narrowing to dtype="
                + String(target)
            )
        _promote(src, want)


def _all_reduced_dims_size_one(a: T, dims: List[Int]) -> Bool:
    """torch's `is_reduce_over_1D_vector`: every dim BEING REDUCED has extent
    1 (a dim that is kept may be any size). Squaring a lone element can
    overflow a magnitude the un-squared `abs` represents exactly (float32
    1e20: `(1e20)**2` overflows to inf, `sqrt(inf)` stays inf), so torch
    special-cases this to `abs()` instead of routing it through the
    square-then-sqrt accumulator -- see `linalg_vector_norm_out` in ATen's
    LinearAlgebra.cpp. That special case fires for every ord != 0 (torch maps
    `ord == 0` to `ne(0)` there instead, since counting is not magnitude), so
    it is correct for every other ord here; callers must skip it for
    ord=0, where the general reduce path already computes `ne(0)` correctly
    through `NormL0Op` for a size-one reduction, same as any other size.
    """
    for d in dims:
        if a.dim(d) != 1:
            return False
    return True


def _vector_norm_abs(a: T, dims: List[Int], keepdim: Bool) raises -> Owned:
    """abs(a), reshaped to the reduced-and-squeezed (or kept-dims) output
    shape; always a FRESH tensor, so this never reads and writes overlapping
    memory regardless of what the caller does with the result."""
    var shape = IndexList[MAX_RANK](1)
    var rank = 0
    _reduced_shape(a, dims, keepdim, shape, rank)
    var src_c = own_if_new(contiguous(a), a)
    var out = own(new_tensor(shape, rank, a.stype, a.device))
    elementwise_direct("elementwise", "AbsSpec", src_c.t, out.t, out.t.dtype)
    _ = src_c^
    return out^


def _vector_norm_abs_out(
    op_name: StaticString, a: T, dims: List[Int], keepdim: Bool, mut dst: T
) raises:
    """Same policy as `_scalar_reduction_out`: dtype (`exact`), `out=`
    aliasing the input declined outright before any resize
    (`_decline_aliasing_out`), and `out=` repeating elements declined too
    (`assert_no_internal_overlap`, after the resize decision).

    The result is always computed into a FRESH tensor first (`_vector_norm_abs`,
    never a direct launch into `dst`) and then copied in: with
    `_decline_aliasing_out` already ruling out any shared storage between
    `dst` and `a`, this is just the same "copy the result across" tail
    `_scalar_reduction_out` uses.
    """
    one_device(a, dst)
    _decline_aliasing_out(op_name, a, dst)
    _check_out_dtype(op_name, "exact", max_dtype(a.stype), dst.dtype)
    var shape = IndexList[MAX_RANK](1)
    var rank = 0
    _reduced_shape(a, dims, keepdim, shape, rank)
    if not _shape_matches(dst, shape, rank):
        resize_out(dst, shape, rank)
    assert_no_internal_overlap(dst)
    var result = _vector_norm_abs(a, dims, keepdim)
    _copy_result_into(dst, result.t)
    _ = result^


def _vector_norm(
    op_label: StaticString,
    a: T,
    ord_v: Value,
    dim_v: Value,
    keepdim: Bool,
    dtype_v: Value,
    rets: Values,
) raises:
    _require_mojo(a)
    var op = _vector_norm_spec(ord_v)
    var src = _borrow(a)
    _decline_normp_float64(op, op_label, src.t.dtype, dtype_v)
    _vector_norm_operand(op_label, dtype_v, src)
    _decline_metal_float64(src.t, op_label)
    var dims = _reduce_dims(dim_v, src.t.rank, True)
    if len(dims) == 0 and src.t.rank != 0:
        unsupported(String(op_label) + " with no reduce dim (a rank-0 operand)")
    if op != "NormL0Spec" and _all_reduced_dims_size_one(src.t, dims):
        var out = _vector_norm_abs(src.t, dims, keepdim)
        ret_owned(rets, 0, out)
        _ = src^
        return
    var ord_f = v_f64_or(ord_v, 2.0)
    if ord_f < 0.0 or ord_f == max_or_inf[DType.float64]():
        # No identity: torch refuses a zero-length reduce dim even when the
        # output itself is empty. Declining on the host gives a clean
        # NotImplementedError; the skeleton's own guard for +-inf
        # (`_rowred_spec_into_go`, unconditional on `reduce_n == 0` too) is
        # a plain `raise Error(...)` -> RuntimeError, not reached here.
        _refuse_empty_extremum(op_label, src.t, dims)
    var out = _scalar_reduction(
        "reduction",
        op,
        src.t,
        dims,
        keepdim,
        src.t.stype,
        op == "NormPSpec",
        ord_f,
    )
    ret_owned(rets, 0, out)
    _ = src^


def _vector_norm_out(
    op_label: StaticString,
    op_name: StaticString,
    a: T,
    ord_v: Value,
    dim_v: Value,
    keepdim: Bool,
    dtype_v: Value,
    mut out: T,
    rets: Values,
) raises:
    _require_mojo(a)
    _require_mojo(out)
    var op = _vector_norm_spec(ord_v)
    var src = _borrow(a)
    _decline_normp_float64(op, op_label, src.t.dtype, dtype_v)
    _vector_norm_operand(op_label, dtype_v, src)
    _decline_metal_float64(src.t, op_label)
    var dims = _reduce_dims(dim_v, src.t.rank, True)
    if len(dims) == 0 and src.t.rank != 0:
        unsupported(String(op_label) + " with no reduce dim (a rank-0 operand)")
    if op != "NormL0Spec" and _all_reduced_dims_size_one(src.t, dims):
        _vector_norm_abs_out(op_name, src.t, dims, keepdim, out)
        ret_ref(rets, 0, out)
        _ = src^
        return
    var ord_f = v_f64_or(ord_v, 2.0)
    if ord_f < 0.0 or ord_f == max_or_inf[DType.float64]():
        _refuse_empty_extremum(op_label, src.t, dims)
    _scalar_reduction_out(
        "reduction",
        op,
        op_name,
        "exact",
        src.t,
        dims,
        keepdim,
        src.t.stype,
        out,
        op == "NormPSpec",
        ord_f,
    )
    ret_ref(rets, 0, out)
    _ = src^


# aten::linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None,
#   bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
def op_linalg_vector_norm(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _vector_norm(
        "linalg_vector_norm",
        v_tensor(args[unsafe_offset=0]),
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        v_bool_or(args[unsafe_offset=3], False),
        args[unsafe_offset=4],
        rets,
    )


# aten::linalg_vector_norm.out(Tensor self, Scalar ord=2, int[1]? dim=None,
#   bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
def op_linalg_vector_norm_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var out = v_tensor(args[unsafe_offset=5])
    _vector_norm_out(
        "linalg_vector_norm",
        "aten::linalg_vector_norm.out",
        v_tensor(args[unsafe_offset=0]),
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        v_bool_or(args[unsafe_offset=3], False),
        args[unsafe_offset=4],
        out,
        rets,
    )


# ---------------------------------------------------------------------------
# norm (legacy overloads): all six redispatch to `linalg_vector_norm_out`
# with `dim=[]` treated as "reduce every dim" and a missing `p` as 2 (see
# `impl_func_norm` in ATen's ReduceOps.cpp) -- the same helpers as
# `linalg_vector_norm` above, gated the same way.
# ---------------------------------------------------------------------------


# aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor
def op_norm_scalar(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _vector_norm(
        "norm",
        v_tensor(args[unsafe_offset=0]),
        args[unsafe_offset=1],
        _none_value(),
        False,
        _none_value(),
        rets,
    )


# aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor
def op_norm_scalaropt_dtype(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _vector_norm(
        "norm",
        v_tensor(args[unsafe_offset=0]),
        args[unsafe_offset=1],
        _none_value(),
        False,
        args[unsafe_offset=2],
        rets,
    )


# aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False)
#   -> Tensor
def op_norm_scalaropt_dim(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _vector_norm(
        "norm",
        v_tensor(args[unsafe_offset=0]),
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        v_bool_or(args[unsafe_offset=3], False),
        _none_value(),
        rets,
    )


# aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *,
#   ScalarType dtype) -> Tensor
def op_norm_scalaropt_dim_dtype(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _vector_norm(
        "norm",
        v_tensor(args[unsafe_offset=0]),
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        v_bool_or(args[unsafe_offset=3], False),
        args[unsafe_offset=4],
        rets,
    )


# aten::norm.out(Tensor self, Scalar? p, int[1] dim, bool keepdim=False, *,
#   Tensor(a!) out) -> Tensor(a!)
def op_norm_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var out = v_tensor(args[unsafe_offset=4])
    _vector_norm_out(
        "norm",
        "aten::norm.out",
        v_tensor(args[unsafe_offset=0]),
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        v_bool_or(args[unsafe_offset=3], False),
        _none_value(),
        out,
        rets,
    )


# aten::norm.dtype_out(Tensor self, Scalar? p, int[1] dim, bool keepdim, *,
#   ScalarType dtype, Tensor(a!) out) -> Tensor(a!)
def op_norm_dtype_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var out = v_tensor(args[unsafe_offset=5])
    _vector_norm_out(
        "norm",
        "aten::norm.dtype_out",
        v_tensor(args[unsafe_offset=0]),
        args[unsafe_offset=1],
        args[unsafe_offset=2],
        v_bool_or(args[unsafe_offset=3], False),
        args[unsafe_offset=4],
        out,
        rets,
    )


# ---------------------------------------------------------------------------
# cumsum
# ---------------------------------------------------------------------------


# aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
def op_cumsum(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a)
    if a.numel == 0 or a.rank == 0:
        unsupported("cumsum of an empty or rank-0 tensor")
    # CUDA uses block prefix sums; HIP and Metal use the portable per-line
    # route. Half dtypes and rank-2 dim 0 are validated on all three APIs.
    # Preserve the CPU device's existing trailing-dimension surface.
    var fast_ok = dev(a.device)[].api != "cpu"
    var src = _borrow(a)
    var want = _opt_dtype(args[unsafe_offset=2])
    if want >= 0:
        if not _is_cumsum_dtype(max_dtype(want), fast_ok):
            unsupported("cumsum with dtype=" + String(max_dtype(want)))
        _promote(src, want)
    elif not src.t.dtype.is_floating_point():
        # torch promotes bool / sub-int64 integer cumsum to int64.
        _promote(src, ST_INT64)
    if not _is_cumsum_dtype(src.t.dtype, fast_ok):
        unsupported("cumsum of dtype " + String(src.t.dtype))
    var dim = _norm_dim(v_int(args[unsafe_offset=1]), src.t.rank)
    var rank = src.t.rank
    if dim != rank - 1 and not (fast_ok and rank == 2 and dim == 0):
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
    var call = KernelCall("nn", "CumsumSpec")
    call.arg_dtype(0, src.t.dtype)
    call.out_dtype(out.t.dtype)
    call.spec(src.t.spec(cp))
    call.int(dim)
    call.spec(out.t.spec(cp))
    call.run()
    _ = ctx
    ret_owned(rets, 0, out)
    _ = src^


# ---------------------------------------------------------------------------
# prod
# ---------------------------------------------------------------------------


# aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
def op_prod(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a)
    var src = _borrow(a)
    var want = _opt_dtype(args[unsafe_offset=1])
    if want >= 0:
        _promote(src, want)
    elif not src.t.dtype.is_floating_point():
        # torch promotes bool / sub-int64 integer prod to int64.
        _promote(src, ST_INT64)
    if not _is_sum_dtype(src.t.dtype):
        unsupported("prod of dtype " + String(src.t.dtype))
    _decline_metal_float64(src.t, "prod")
    var dims = _trailing_dims(src.t.rank, src.t.rank)
    var out = _scalar_reduction(
        "reduction",
        "ProdSpec",
        src.t,
        dims,
        False,
        src.t.stype,
        False,
        0.0,
    )
    ret_owned(rets, 0, out)
    _ = src^


# aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *,
#   ScalarType? dtype=None) -> Tensor
def op_prod_dim_int(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a)
    var src = _borrow(a)
    var want = _opt_dtype(args[unsafe_offset=3])
    if want >= 0:
        _promote(src, want)
    elif not src.t.dtype.is_floating_point():
        _promote(src, ST_INT64)
    if not _is_sum_dtype(src.t.dtype):
        unsupported("prod of dtype " + String(src.t.dtype))
    _decline_metal_float64(src.t, "prod")
    var dims = _reduce_dim_single(v_int(args[unsafe_offset=1]), src.t.rank)
    var out = _scalar_reduction(
        "reduction",
        "ProdSpec",
        src.t,
        dims,
        v_bool_or(args[unsafe_offset=2], False),
        src.t.stype,
        False,
        0.0,
    )
    ret_owned(rets, 0, out)
    _ = src^


# aten::prod.int_out(Tensor self, int dim, bool keepdim=False, *,
#   ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
def op_prod_int_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=4])
    _require_mojo(a)
    _require_mojo(out)
    var src = _borrow(a)
    # `out` always exists here, so its dtype is the compute dtype for ANY
    # self with no explicit dtype= -- the bool/sub-int64 -> int64 default
    # (`op_prod`, above) only applies when there is no out tensor to take a
    # dtype from.
    var target = _out_reduce_dtype(
        args[unsafe_offset=3], out, "aten::prod.int_out"
    )
    if not _is_sum_dtype(target):
        unsupported("prod with dtype=" + String(target))
    _promote_for_out_reduction(src, target, "aten::prod.int_out")
    var dims = _reduce_dim_single(v_int(args[unsafe_offset=1]), src.t.rank)
    _scalar_reduction_out(
        "reduction",
        "ProdSpec",
        "aten::prod.int_out",
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
# sort / topk: one segmented (key, index) sort behind both ops
# (tmb/kernels/sort/entry.mojo).
#
# The kernel orders (order-preserving unsigned key, original index) pairs, so
# the order is total: ties resolve by index, which makes every result the
# STABLE one. `sort(stable=True)` therefore costs nothing extra, and `topk` is
# the same primitive run as a tournament. That is stronger than ATen asks
# for: `sort(stable=False)` and `topk` leave the order of equal elements
# unspecified, and CPU ATen may pick a different one.
# ---------------------------------------------------------------------------

# Elements one block sorts in shared memory, mirrored from `_TILE` in
# tmb/kernels/sort/entry.mojo: the op picks the launch route and sizes the
# workspace, so the two have to agree.
comptime SORT_TILE_32 = 4096
comptime SORT_TILE_64 = 2048
# grid.y carries the row, and CUDA/Metal cap that dimension at 65535; more
# rows run as several launches over row chunks.
comptime SORT_MAX_GRID_ROWS = 65535
# The index rides through the kernel as int32 (halving shared-memory traffic
# against int64), so a longer row cannot be addressed.
comptime SORT_MAX_ROW = 2147483647


def _sort_kernel_dtype(a: T, op: StaticString) raises -> DType:
    """The kernel specialization for `a`'s dtype: bool rides uint8 (its ties
    are broken by index, so the stable answer is well defined); the unsigned
    16/32/64-bit dtypes and float64 on Apple GPUs are declined."""
    var dt = a.dtype
    if dt == DType.bool:
        return DType.uint8
    if (
        dt == DType.float32
        or dt == DType.bfloat16
        or dt == DType.float16
        or dt == DType.int64
        or dt == DType.int32
        or dt == DType.int16
        or dt == DType.int8
        or dt == DType.uint8
    ):
        return dt
    if dt == DType.float64:
        if dev(a.device)[].api == "metal":
            unsupported(String(op) + ": float64 is unavailable on Apple GPUs")
        return dt
    unsupported(String(op) + " of dtype " + String(dt))
    return dt


def _sort_dim(a: T, dim: Int) raises -> Int:
    """ATen's `maybe_wrap_dim` for sort/topk: a 0-d tensor has one dim."""
    return _norm_dim(dim, a.rank)


def _selected_shape(a: T, dim: Int, k: Int) -> IndexList[MAX_RANK]:
    """`a`'s shape with `dim` replaced by `k` (a 0-d tensor stays 0-d)."""
    var shape = a.shape
    if a.rank > 0:
        shape[MAX_RANK - a.rank + dim] = k
    return shape


def _dim_last_view(t: T, dim: Int) -> T:
    """`t` read with `dim` moved to the end, the other dims in order: the
    same storage, only shape/strides rewritten (see `_permuted_contiguous`).
    """
    var shape = IndexList[MAX_RANK](1)
    var strides = IndexList[MAX_RANK](0)
    var pad = MAX_RANK - t.rank
    var w = 0
    for d in range(t.rank):
        if d != dim:
            shape[pad + w] = t.dim(d)
            strides[pad + w] = t.stride(d)
            w += 1
    shape[MAX_RANK - 1] = t.dim(dim)
    strides[MAX_RANK - 1] = t.stride(dim)
    var view = t.copy()
    view.shape = shape
    view.strides = strides
    view.contig = False
    return view^


def _sort_rows_into(
    src: T,
    kdt: DType,
    rows: Int,
    n: Int,
    out_k: Int,
    topk: Bool,
    descending: Bool,
    values: T,
    indices: T,
    select_mode: Int = -1,
    select_pos: Int = 0,
) raises:
    """Run the kernel over `rows` contiguous rows of `n` elements of `src`,
    writing the first `out_k` of each sorted row into the contiguous
    `values` / `indices` -- or, with a `select_mode` (the sort family's
    `SELECT_*`, ascending only), the one element that mode reads from each
    row of `out_k` sorted elements into `values` / `indices` of `rows`."""
    var select = select_mode >= 0
    var per_row = 1 if select else out_k
    var tile = SORT_TILE_64 if dtype_itemsize(kdt) == 8 else SORT_TILE_32
    var tiles = (n + tile - 1) // tile
    var n_pow2 = 1
    while n_pow2 < n:
        n_pow2 <<= 1
    var route: Int
    var pad: Int
    if n_pow2 <= tile:
        route = 0
        pad = n_pow2
    elif topk and tiles * out_k <= tile:
        route = 1
        pad = tiles * out_k
    else:
        route = 2
        pad = n_pow2
    var chunk = min(rows, SORT_MAX_GRID_ROWS)
    var ws = IndexList[MAX_RANK](1)
    ws[MAX_RANK - 1] = chunk * pad
    # Only the key's byte width matters (the kernel reinterprets the buffer
    # as unsigned); int32/int64 are those widths.
    var key_stype = ST_INT64 if dtype_itemsize(kdt) == 8 else ST_INT32
    var keys = own(new_tensor(ws, 1, key_stype, src.device))
    var scratch = own(new_tensor(ws, 1, ST_INT32, src.device))
    var ctx = ctx_for(src.device)
    var cp = ctx_ptr(ctx)
    var r0 = 0
    while r0 < rows:
        var r = min(chunk, rows - r0)
        var call = KernelCall("sort", "SortSelect" if select else "Sort")
        call.arg_dtype(0, kdt)
        call.int(values.ptr + r0 * per_row * values.itemsize)
        call.int(indices.ptr + r0 * per_row * 8)
        call.int(src.ptr + r0 * n * src.itemsize)
        call.int(keys.t.ptr)
        call.int(scratch.t.ptr)
        call.int(r)
        call.int(n)
        call.int(out_k)
        call.int(pad)
        call.int(route)
        if select:
            call.int(select_pos)
            call.int(select_mode)
        else:
            call.int(1 if descending else 0)
        call.int(dtype_code(kdt))
        call.int(cp)
        call.run()
        r0 += r
    _ = ctx
    _ = keys^  # alive past the launches
    _ = scratch^


def _select_into(
    a: T,
    op: StaticString,
    dim_in: Int,
    k: Int,
    topk: Bool,
    descending: Bool,
    values: T,
    indices: T,
) raises:
    """sort (`topk` False, `k` the whole dim) or topk along `dim_in` of `a`,
    into the fresh contiguous `values` / `indices` of the result's shape."""
    var kdt = _sort_kernel_dtype(a, op)
    var dim = _sort_dim(a, dim_in)
    var n = a.dim(dim) if a.rank > 0 else 1
    if n > SORT_MAX_ROW:
        unsupported(String(op) + " of a row longer than 2**31 - 1")
    if values.numel == 0:
        return
    var rows = a.numel // n
    var last = a.rank <= 1 or dim == a.rank - 1
    var dims = List[Int]()
    dims.append(dim)
    var src = _borrow(a)
    if a.rank > 0 and not (last and a.contig):
        src.replace(_permuted_contiguous(a, dims), True)
    if last:
        _sort_rows_into(
            src.t, kdt, rows, n, k, topk, descending, values, indices
        )
        _ = src^
        return
    var moved = src.t.shape
    moved[MAX_RANK - 1] = k
    var tv = own(new_tensor(moved, a.rank, a.stype, a.device))
    var ti = own(new_tensor(moved, a.rank, ST_INT64, a.device))
    _sort_rows_into(src.t, kdt, rows, n, k, topk, descending, tv.t, ti.t)
    _ = src^  # alive past the launch
    copy_strided_into(_dim_last_view(values, dim), tv.t)
    copy_strided_into(_dim_last_view(indices, dim), ti.t)
    _ = tv^
    _ = ti^


def _select(
    a: T, op: StaticString, dim_in: Int, k: Int, topk: Bool, descending: Bool
) raises -> Tuple[Owned, Owned]:
    _require_mojo(a)
    var dim = _sort_dim(a, dim_in)
    var shape = _selected_shape(a, dim, k)
    var values = own(new_tensor(shape, a.rank, a.stype, a.device))
    var indices = own(new_tensor(shape, a.rank, ST_INT64, a.device))
    _select_into(a, op, dim, k, topk, descending, values.t, indices.t)
    return (values^, indices^)


def _select_out(
    a: T,
    op: StaticString,
    dim_in: Int,
    k: Int,
    topk: Bool,
    descending: Bool,
    var out_v: T,
    var out_i: T,
    rets: Values,
) raises:
    """The `out=` overloads: check, resize, then compute straight into the
    caller's tensors when they are contiguous, else compute and copy."""
    _require_mojo(a)
    check_out(out_v, a)
    if out_i.stype != ST_INT64:
        raise Error(
            "Expected out tensor to have dtype long int, but got ",
            dtype_name(out_i.stype),
            " instead",
        )
    one_device(a, out_v)
    one_device(a, out_i)
    var dim = _sort_dim(a, dim_in)
    var shape = _selected_shape(a, dim, k)
    var numel = _shape_numel(shape, a.rank)
    if not _shape_matches(out_v, shape, a.rank):
        resize_out(out_v, shape, a.rank)
    if not _shape_matches(out_i, shape, a.rank):
        resize_out(out_i, shape, a.rank)
    if _out_ready(out_v, a, a.stype, numel) and _out_ready(
        out_i, a, ST_INT64, numel
    ):
        _select_into(a, op, dim, k, topk, descending, out_v, out_i)
    else:
        var pair = _select(a, op, dim, k, topk, descending)
        copy_strided_into(out_v, pair[0].t)
        copy_strided_into(out_i, pair[1].t)
        _ = pair^  # alive past the launches
    ret_ref(rets, 0, out_v)
    ret_ref(rets, 1, out_i)


def _topk_k(a: T, dim_in: Int, k: Int) raises -> Int:
    var dim = _sort_dim(a, dim_in)
    var size = a.dim(dim) if a.rank > 0 else 1
    if k < 0 or k > size:
        raise Error("selected index k out of range")
    return k


def _sort_size(a: T, dim_in: Int) raises -> Int:
    var dim = _sort_dim(a, dim_in)
    return a.dim(dim) if a.rank > 0 else 1


# aten::topk(Tensor self, SymInt k, int dim=-1, bool largest=True,
#   bool sorted=True) -> (Tensor values, Tensor indices)
# `sorted=False` only permits any order among the k results; the sorted
# order is one of them.
def op_topk(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var dim = v_int_or(args[unsafe_offset=2], -1)
    var k = _topk_k(a, dim, v_int(args[unsafe_offset=1]))
    var largest = v_bool_or(args[unsafe_offset=3], True)
    var pair = _select(a, "topk", dim, k, True, largest)
    ret_owned(rets, 0, pair[0])
    ret_owned(rets, 1, pair[1])


# aten::topk.values(Tensor self, SymInt k, int dim=-1, bool largest=True,
#   bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices)
#   -> (Tensor(a!) values, Tensor(b!) indices)
def op_topk_values(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var dim = v_int_or(args[unsafe_offset=2], -1)
    var k = _topk_k(a, dim, v_int(args[unsafe_offset=1]))
    var largest = v_bool_or(args[unsafe_offset=3], True)
    _select_out(
        a,
        "topk",
        dim,
        k,
        True,
        largest,
        v_tensor(args[unsafe_offset=5]),
        v_tensor(args[unsafe_offset=6]),
        rets,
    )


# aten::sort.stable(Tensor self, *, bool? stable, int dim=-1,
#   bool descending=False) -> (Tensor values, Tensor indices)
# Every result is the stable one, so `stable` needs no branch.
def op_sort_stable(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var dim = v_int_or(args[unsafe_offset=2], -1)
    var descending = v_bool_or(args[unsafe_offset=3], False)
    var pair = _select(a, "sort", dim, _sort_size(a, dim), False, descending)
    ret_owned(rets, 0, pair[0])
    ret_owned(rets, 1, pair[1])


# aten::sort.values_stable(Tensor self, *, bool? stable, int dim=-1,
#   bool descending=False, Tensor(a!) values, Tensor(b!) indices)
#   -> (Tensor(a!) values, Tensor(b!) indices)
def op_sort_values_stable(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var dim = v_int_or(args[unsafe_offset=2], -1)
    var descending = v_bool_or(args[unsafe_offset=3], False)
    _select_out(
        a,
        "sort",
        dim,
        _sort_size(a, dim),
        False,
        descending,
        v_tensor(args[unsafe_offset=4]),
        v_tensor(args[unsafe_offset=5]),
        rets,
    )


# ---------------------------------------------------------------------------
# kthvalue / median / nanmedian: one order statistic per row, read off the
# sort kernel's total (key, index) order by its `SortSelect` op -- the same
# routes as sort/topk (kthvalue takes the topk tournament when k is small),
# then one element per row instead of the gather. That order is also what
# makes the answers ATen's: a tie resolves to the LOWEST index of the tied
# value (CPU's `nth_element` comparator for median), every NaN sorts after
# every number, and `median` returns the lower middle and propagates NaN as
# the row's first NaN (value and index), as CPU torch does.
# ---------------------------------------------------------------------------

# Mirrored from SELECT_* in tmb/kernels/sort/entry.mojo.
comptime SELECT_KTH = 0
comptime SELECT_MEDIAN = 1
comptime SELECT_NANMEDIAN = 2


def _order_stat_dtype(a: T, what: StaticString) raises -> DType:
    """ATen dispatches these over ALL_TYPES + half/bfloat16: bool raises the
    dispatcher's own error, everything else rides the sort specializations."""
    if a.dtype == DType.bool:
        raise Error('"', what, "\" not implemented for 'Bool'")
    return _sort_kernel_dtype(a, what)


def _order_stat_shape(
    a: T, dim: Int, keepdim: Bool
) -> Tuple[IndexList[MAX_RANK], Int]:
    """The (shape, rank) of a one-per-row result along `dim`."""
    if a.rank == 0:
        return (a.shape, 0)
    if keepdim:
        return (_selected_shape(a, dim, 1), a.rank)
    var shape = IndexList[MAX_RANK](1)
    var rank = a.rank - 1
    var w = 0
    for d in range(a.rank):
        if d != dim:
            shape[MAX_RANK - rank + w] = a.dim(d)
            w += 1
    return (shape, rank)


def _order_stat_rows_into(
    src: T,
    kdt: DType,
    rows: Int,
    n: Int,
    k: Int,
    mode: Int,
    values: T,
    indices: T,
) raises:
    """The k-th smallest (1-based; median's lower middle is k = (n+1)//2) of
    each of the `rows` contiguous rows of `src`. Only kthvalue can stop at a
    top-k tournament; the median modes need the whole row sorted, and on a
    non-float dtype (no NaN) they are that plain k-th smallest."""
    var m = mode if kdt.is_floating_point() else SELECT_KTH
    var topk = mode == SELECT_KTH
    _sort_rows_into(
        src,
        kdt,
        rows,
        n,
        k if topk else n,
        topk,
        False,
        values,
        indices,
        m,
        k - 1,
    )


def _order_stat_into(
    a: T,
    what: StaticString,
    dim: Int,
    k: Int,
    mode: Int,
    values: T,
    indices: T,
) raises:
    """Along the normalized `dim` of `a`, into the fresh contiguous `values`
    / `indices` of the result shape (whose element order is the row order of
    `a` with `dim` moved last, whether or not `dim` was kept)."""
    var kdt = _order_stat_dtype(a, what)
    var n = a.dim(dim) if a.rank > 0 else 1
    if n > SORT_MAX_ROW:
        unsupported(String(what) + " of a row longer than 2**31 - 1")
    if values.numel == 0:
        return
    var rows = a.numel // n
    var last = a.rank <= 1 or dim == a.rank - 1
    var src = _borrow(a)
    if a.rank > 0 and not (last and a.contig):
        var dims = List[Int]()
        dims.append(dim)
        src.replace(_permuted_contiguous(a, dims), True)
    _order_stat_rows_into(src.t, kdt, rows, n, k, mode, values, indices)
    _ = src^  # alive past the launches


def _order_stat_dim(a: T, dim_in: Int) raises -> Int:
    """`maybe_wrap_dim` plus ATen's `zero_numel_check_dims`."""
    var dim = _sort_dim(a, dim_in)
    if a.rank > 0 and a.dim(dim) == 0:
        raise Error(
            "median(): Expected reduction dim ",
            dim,
            " to have non-zero size.",
        )
    return dim


def _order_stat(
    a: T, what: StaticString, dim: Int, keepdim: Bool, k: Int, mode: Int
) raises -> Tuple[Owned, Owned]:
    _require_mojo(a)
    var sr = _order_stat_shape(a, dim, keepdim)
    var values = own(new_tensor(sr[0], sr[1], a.stype, a.device))
    var indices = own(new_tensor(sr[0], sr[1], ST_INT64, a.device))
    _order_stat_into(a, what, dim, k, mode, values.t, indices.t)
    return (values^, indices^)


def _order_stat_out(
    a: T,
    what: StaticString,
    dim: Int,
    keepdim: Bool,
    k: Int,
    mode: Int,
    var out_v: T,
    var out_i: T,
    rets: Values,
) raises:
    """The `values=`/`indices=` overloads, as `_select_out` does them."""
    _require_mojo(a)
    check_out(out_v, a)
    if out_i.stype != ST_INT64:
        raise Error(
            "Expected out tensor to have dtype long int, but got ",
            dtype_name(out_i.stype),
            " instead",
        )
    one_device(a, out_v)
    one_device(a, out_i)
    var sr = _order_stat_shape(a, dim, keepdim)
    var numel = _shape_numel(sr[0], sr[1])
    if not _shape_matches(out_v, sr[0], sr[1]):
        resize_out(out_v, sr[0], sr[1])
    if not _shape_matches(out_i, sr[0], sr[1]):
        resize_out(out_i, sr[0], sr[1])
    if _out_ready(out_v, a, a.stype, numel) and _out_ready(
        out_i, a, ST_INT64, numel
    ):
        _order_stat_into(a, what, dim, k, mode, out_v, out_i)
    else:
        var pair = _order_stat(a, what, dim, keepdim, k, mode)
        copy_strided_into(out_v, pair[0].t)
        copy_strided_into(out_i, pair[1].t)
        _ = pair^  # alive past the launches
    ret_ref(rets, 0, out_v)
    ret_ref(rets, 1, out_i)


def _kth_k(a: T, dim: Int, k: Int) raises -> Int:
    var size = a.dim(dim) if a.rank > 0 else 1
    if a.rank > 0 and size == 0:
        raise Error(
            "kthvalue(): Expected reduction dim ",
            dim,
            " to have non-zero size.",
        )
    if k < 1 or k > size:
        raise Error(
            "kthvalue(): selected number k out of range for dimension ", dim
        )
    return k


def _median_k(a: T, dim: Int) -> Int:
    var size = a.dim(dim) if a.rank > 0 else 1
    return (size + 1) // 2


# aten::kthvalue(Tensor self, SymInt k, int dim=-1, bool keepdim=False)
#   -> (Tensor values, Tensor indices)
def op_kthvalue(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var dim = _sort_dim(a, v_int_or(args[unsafe_offset=2], -1))
    var k = _kth_k(a, dim, v_int(args[unsafe_offset=1]))
    var keepdim = v_bool_or(args[unsafe_offset=3], False)
    var pair = _order_stat(a, "kthvalue_cpu", dim, keepdim, k, SELECT_KTH)
    ret_owned(rets, 0, pair[0])
    ret_owned(rets, 1, pair[1])


# aten::kthvalue.values(Tensor self, SymInt k, int dim=-1, bool keepdim=False,
#   *, Tensor(a!) values, Tensor(b!) indices)
#   -> (Tensor(a!) values, Tensor(b!) indices)
def op_kthvalue_values(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var dim = _sort_dim(a, v_int_or(args[unsafe_offset=2], -1))
    var k = _kth_k(a, dim, v_int(args[unsafe_offset=1]))
    assert_no_overlap(v_tensor(args[unsafe_offset=4]), a)
    _order_stat_out(
        a,
        "kthvalue_cpu",
        dim,
        v_bool_or(args[unsafe_offset=3], False),
        k,
        SELECT_KTH,
        v_tensor(args[unsafe_offset=4]),
        v_tensor(args[unsafe_offset=5]),
        rets,
    )


def _median_dim(
    args: Values, rets: Values, mode: Int, what: StaticString
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var dim = _order_stat_dim(a, v_int(args[unsafe_offset=1]))
    var keepdim = v_bool_or(args[unsafe_offset=2], False)
    var pair = _order_stat(a, what, dim, keepdim, _median_k(a, dim), mode)
    ret_owned(rets, 0, pair[0])
    ret_owned(rets, 1, pair[1])


def _median_dim_values(
    args: Values, rets: Values, mode: Int, what: StaticString
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var dim = _order_stat_dim(a, v_int(args[unsafe_offset=1]))
    _order_stat_out(
        a,
        what,
        dim,
        v_bool_or(args[unsafe_offset=2], False),
        _median_k(a, dim),
        mode,
        v_tensor(args[unsafe_offset=3]),
        v_tensor(args[unsafe_offset=4]),
        rets,
    )


def _median_all(args: Values, rets: Values, mode: Int) raises:
    """median() / nanmedian(): the whole tensor as one row, a 0-d value (NaN
    for an empty float tensor, as ATen returns)."""
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a)
    var kdt = _order_stat_dtype(a, "median_cpu")
    var value = own(new_scalar(a.stype, a.device))
    if a.numel == 0:
        if not kdt.is_floating_point():
            unsupported("median of an empty integer tensor")
        fill_value(value.t, nan[DType.float64]())
        ret_owned(rets, 0, value)
        return
    if a.numel > SORT_MAX_ROW:
        unsupported("median of more than 2**31 - 1 elements")
    var index = own(new_scalar(ST_INT64, a.device))
    var src = Operand(contiguous(a), not a.contig)
    _order_stat_rows_into(
        src.t, kdt, 1, a.numel, (a.numel + 1) // 2, mode, value.t, index.t
    )
    _ = src^  # alive past the launches
    _ = index^
    ret_owned(rets, 0, value)


# aten::median.dim(Tensor self, int dim, bool keepdim=False)
#   -> (Tensor values, Tensor indices)
def op_median_dim(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _median_dim(args, rets, SELECT_MEDIAN, "median_out")


# aten::median.dim_values(Tensor self, int dim, bool keepdim=False, *,
#   Tensor(a!) values, Tensor(b!) indices)
#   -> (Tensor(a!) values, Tensor(b!) indices)
def op_median_dim_values(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _median_dim_values(args, rets, SELECT_MEDIAN, "median_out")


# aten::median(Tensor self) -> Tensor
def op_median(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _median_all(args, rets, SELECT_MEDIAN)


# aten::nanmedian.dim(Tensor self, int dim, bool keepdim=False)
#   -> (Tensor values, Tensor indices)
def op_nanmedian_dim(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _median_dim(args, rets, SELECT_NANMEDIAN, "median_out")


# aten::nanmedian.dim_values(Tensor self, int dim, bool keepdim=False, *,
#   Tensor(a!) values, Tensor(b!) indices)
#   -> (Tensor(a!) values, Tensor(b!) indices)
def op_nanmedian_dim_values(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _median_dim_values(args, rets, SELECT_NANMEDIAN, "median_out")


# aten::nanmedian(Tensor self) -> Tensor
def op_nanmedian(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _median_all(args, rets, SELECT_NANMEDIAN)


def register_reductions(site: Site) raises:
    impl[op_all, "all"](site)
    impl[op_all_all_out, "all.all_out"](site)
    impl[op_all_dim, "all.dim"](site)
    impl[op_all_dim, "all.dims"](site)
    impl[op_all_out, "all.dims_out"](site)
    impl[op_all_out, "all.out"](site)
    impl[op_amax, "amax"](site)
    impl[op_amax_out, "amax.out"](site)
    impl[op_amin, "amin"](site)
    impl[op_amin_out, "amin.out"](site)
    impl[op_any, "any"](site)
    impl[op_any_all_out, "any.all_out"](site)
    impl[op_any_dim, "any.dim"](site)
    impl[op_any_dim, "any.dims"](site)
    impl[op_any_out, "any.dims_out"](site)
    impl[op_any_out, "any.out"](site)
    impl[op_argmax, "argmax"](site)
    impl[op_argmin, "argmin"](site)
    impl[op_count_nonzero, "count_nonzero.dim_IntList"](site)
    impl[op_cumsum, "cumsum"](site)
    impl[op_kthvalue, "kthvalue"](site)
    impl[op_kthvalue_values, "kthvalue.values"](site)
    impl[op_linalg_vector_norm, "linalg_vector_norm"](site)
    impl[op_linalg_vector_norm_out, "linalg_vector_norm.out"](site)
    impl[op_max, "max"](site)
    impl[op_max_unary_out, "max.unary_out"](site)
    impl[op_mean, "mean"](site)
    impl[op_mean_dim, "mean.dim"](site)
    impl[op_mean_dtype_out, "mean.dtype_out"](site)
    impl[op_mean_out, "mean.out"](site)
    impl[op_median, "median"](site)
    impl[op_median_dim, "median.dim"](site)
    impl[op_median_dim_values, "median.dim_values"](site)
    impl[op_min, "min"](site)
    impl[op_min_dim, "min.dim"](site)
    impl[op_min_dim_min, "min.dim_min"](site)
    impl[op_min_unary_out, "min.unary_out"](site)
    impl[op_nanmedian, "nanmedian"](site)
    impl[op_nanmedian_dim, "nanmedian.dim"](site)
    impl[op_nanmedian_dim_values, "nanmedian.dim_values"](site)
    impl[op_nansum, "nansum"](site)
    impl[op_nansum_out, "nansum.out"](site)
    impl[op_norm_dtype_out, "norm.dtype_out"](site)
    impl[op_norm_out, "norm.out"](site)
    impl[op_norm_scalar, "norm.Scalar"](site)
    impl[op_norm_scalaropt_dim, "norm.ScalarOpt_dim"](site)
    impl[op_norm_scalaropt_dim_dtype, "norm.ScalarOpt_dim_dtype"](site)
    impl[op_norm_scalaropt_dtype, "norm.ScalarOpt_dtype"](site)
    impl[op_prod, "prod"](site)
    impl[op_prod_dim_int, "prod.dim_int"](site)
    impl[op_prod_int_out, "prod.int_out"](site)
    impl[op_sort_stable, "sort.stable"](site)
    impl[op_sort_values_stable, "sort.values_stable"](site)
    impl[op_sum, "sum"](site)
    impl[op_sum_dim_intlist, "sum.dim_IntList"](site)
    impl[op_sum_intlist_out, "sum.IntList_out"](site)
    impl[op_topk, "topk"](site)
    impl[op_topk_values, "topk.values"](site)
    impl[op_var_correction, "var.correction"](site)
