"""ATen ops: unique group (see agents_docs/native_backend.md).

_unique / _unique2 / unique_consecutive / unique_dim /
unique_dim_consecutive, on the device like ATen's native/cuda/UniqueCub.cu
and Unique.cu: a stable sort (the reductions group's sort.stable; rows by
the unique family's lexicographic merge sort of row indices), then
the `unique` kernel family's adjacent difference, a cumsum of it (the group
of every row), and one pass that writes each group's representative, the
inverse indices and the run starts the counts come from. The output length
is data-dependent: the group count is read back to the host, one sync, as on
CUDA.

CUDA's choices this follows: `sorted=False` sorts anyway; a run keeps its
first element, or its last when counts are requested (cub's
run_length_encode), which is visible for -0.0 / 0.0 runs; NaN never equals
anything; outputs the caller did not ask for are empty, except the flat
ops' inverse of an empty input, which has the input's shape, and a bool
input's counts, which CUDA always returns. (Where CUDA returns an undefined
inverse for bool, this returns an empty one.) Rows sort with NaN after
every number, as torch.sort orders values (CUDA's `<` / `>` row comparator
is no strict weak order with NaN, so its NaN placement is undefined), and
NaN never equals NaN, so a row holding one is always its own group, as on
CPU.
"""
from std.utils import IndexList

from tmb.backend.abi import (
    Owned,
    ST_INT64,
    T,
    TAG_BOOL,
    TAG_INT,
    TAG_SCALAR_INT,
    Value,
    Values,
    contiguous_strides,
    dtype_code,
    index_error,
    new_tensor,
    own,
    own_if_new,
    ret_owned,
    tensor_arg,
    unsupported,
    v_bool,
    v_int,
    v_is_none,
    v_tensor,
    view_strided,
)
from tmb.backend.device import ctx_for, ctx_ptr
from tmb.backend.kernel_call import KernelCall
from tmb.backend.registry import Site, impl
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.common import call_op, contiguous


def _vec(n: Int) -> IndexList[MAX_RANK]:
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 1] = n
    return shape


def _empty(n: Int, stype: Int32, device: Int) raises -> Owned:
    return own(new_tensor(_vec(n), 1, stype, device))


def _op(name: String, overload: String, var args: List[Value]) raises -> Owned:
    var r = call_op(name, overload, args^, 1)
    return own(r.take_tensor(0))


def _int(x: Int) -> Value:
    return Value(TAG_INT, 0, Int64(x), 0)


def _stable_sort_perm(keys: T) raises -> Owned:
    """The permutation of a stable ascending sort of the 1-D `keys`."""
    var r = call_op(
        "aten::sort",
        "stable",
        [
            tensor_arg(keys),
            Value(TAG_BOOL, 0, 1, 0),
            _int(0),
            Value(TAG_BOOL, 0, 0, 0),
        ],
        2,
    )
    return own(r.take_tensor(1))


def _read_last(t: T) raises -> Int:
    """The last element of the contiguous 1-D int64 `t` (one sync)."""
    var one = own(
        view_strided(
            t,
            IndexList[MAX_RANK](1),
            IndexList[MAX_RANK](0),
            0,
            t.offset + t.numel - 1,
        )
    )
    var r = call_op("aten::_local_scalar_dense", "", [tensor_arg(one.t)], 1)
    _ = one^
    return v_int(r[0])


struct Groups(Movable):
    """The runs of a sorted row sequence: `out_vals` (when asked) holds one
    representative row per run, `out_rows` its source row, `inverse` the
    run of every source row, `counts` the run lengths."""

    var m: Int
    var out_vals: Optional[Owned]
    var out_rows: Optional[Owned]
    var inverse: Optional[Owned]
    var counts: Optional[Owned]

    def __init__(out self):
        self.m = 0
        self.out_vals = None
        self.out_rows = None
        self.inverse = None
        self.counts = None


def _groups(
    data: T,
    perm: Optional[T],
    n: Int,
    inner: Int,
    want_vals: Bool,
    want_rows: Bool,
    want_inverse: Bool,
    want_counts: Bool,
    take_last: Bool,
) raises -> Groups:
    """Group the `n` rows of `inner` elements of the contiguous `data`, read
    in `perm` order (identity when absent), into runs of equal rows."""
    var device = data.device
    var ctx = ctx_for(device)
    var marks = _empty(n, ST_INT64, device)
    var mc = KernelCall("unique", "UniqueMarks")
    mc.arg_dtype(0, data.dtype)
    mc.int(marks.t.ptr)
    mc.int(data.ptr)
    mc.int(perm.value().ptr if perm else 0)
    mc.tuple([n, inner])
    mc.int(dtype_code(data.dtype))
    mc.int(ctx_ptr(ctx))
    mc.run()
    var gid = _op(
        "aten::cumsum",
        "",
        [tensor_arg(marks.t), _int(0), Value(0, 0, 0, 0)],
    )
    _ = marks^
    var g = Groups()
    g.m = _read_last(gid.t) + 1
    if want_vals:
        g.out_vals = own(new_tensor(_vec(g.m * inner), 1, data.stype, device))
    if want_rows:
        g.out_rows = _empty(g.m, ST_INT64, device)
    if want_inverse:
        g.inverse = _empty(n, ST_INT64, device)
    var starts = Optional[Owned](None)
    if want_counts:
        starts = _empty(g.m, ST_INT64, device)
    var sc = KernelCall("unique", "UniqueSelect")
    sc.arg_dtype(0, data.dtype)
    sc.tuple(
        [
            g.out_vals.value().t.ptr if g.out_vals else 0,
            g.out_rows.value().t.ptr if g.out_rows else 0,
            g.inverse.value().t.ptr if g.inverse else 0,
            starts.value().t.ptr if starts else 0,
            data.ptr,
            perm.value().ptr if perm else 0,
            gid.t.ptr,
        ]
    )
    sc.tuple([1 if take_last else 0, n, inner])
    sc.int(dtype_code(data.dtype))
    sc.int(ctx_ptr(ctx))
    sc.run()
    if want_counts:
        g.counts = _empty(g.m, ST_INT64, device)
        var cc = KernelCall("unique", "UniqueCounts")
        cc.int(g.counts.value().t.ptr)
        cc.int(starts.value().t.ptr)
        cc.int(g.m)
        cc.int(n)
        cc.int(ctx_ptr(ctx))
        cc.run()
    _ = starts^
    _ = gid^
    _ = ctx
    return g^


struct RunBounds(Movable):
    """For a sorted 1-D sequence: the run of every position (`gid`), and the
    first and last position of every run (`first[g]`, `last[g]`); buffers of
    the sequence's length, valid up to the run count, never read back."""

    var gid: Owned
    var first: Owned
    var last: Owned

    def __init__(out self, var gid: Owned, var first: Owned, var last: Owned):
        self.gid = gid^
        self.first = first^
        self.last = last^


def run_bounds(sorted: T) raises -> RunBounds:
    """`RunBounds` of the contiguous 1-D `sorted` (no host read)."""
    var n = sorted.numel
    var device = sorted.device
    var ctx = ctx_for(device)
    var marks = _empty(n, ST_INT64, device)
    var mc = KernelCall("unique", "UniqueMarks")
    mc.arg_dtype(0, sorted.dtype)
    mc.int(marks.t.ptr)
    mc.int(sorted.ptr)
    mc.int(0)
    mc.tuple([n, 1])
    mc.int(dtype_code(sorted.dtype))
    mc.int(ctx_ptr(ctx))
    mc.run()
    var gid = _op(
        "aten::cumsum",
        "",
        [tensor_arg(marks.t), _int(0), Value(0, 0, 0, 0)],
    )
    _ = marks^
    var first = _empty(n, ST_INT64, device)
    var last = _empty(n, ST_INT64, device)
    for take_last in range(2):
        var sc = KernelCall("unique", "UniqueSelect")
        sc.arg_dtype(0, sorted.dtype)
        sc.tuple(
            [
                0,
                last.t.ptr if take_last else 0,
                0,
                0 if take_last else first.t.ptr,
                sorted.ptr,
                0,
                gid.t.ptr,
            ]
        )
        sc.tuple([take_last, n, 1])
        sc.int(dtype_code(sorted.dtype))
        sc.int(ctx_ptr(ctx))
        sc.run()
    _ = ctx
    return RunBounds(gid^, first^, last^)


def _check_dtype(t: T, what: String) raises:
    var dt = t.dtype
    if not (
        dt.is_floating_point()
        or dt == DType.int64
        or dt == DType.int32
        or dt == DType.int16
        or dt == DType.int8
        or dt == DType.uint8
        or dt == DType.bool
    ):
        unsupported("aten::" + what + " of dtype " + String(dt))
    if dt == DType.float64 and ctx_for(t.device).api() == "metal":
        unsupported("aten::" + what + " of float64 on Apple GPU")


def _flat_view(t: T) raises -> Owned:
    """A contiguous 1-D tensor over every element of `t` (a copy when `t`
    is not contiguous)."""
    var c = own_if_new(contiguous(t), t)
    var v = own(view_strided(c.t, _vec(c.t.numel), _vec(1), 1, c.t.offset))
    _ = c^
    return v^


def unique_flat(
    t: T,
    consecutive: Bool,
    want_inverse: Bool,
    want_counts: Bool,
) raises -> Tuple[Owned, Owned, Owned]:
    """`unique_cuda_template`: (values, inverse, counts) of the flattened
    `t`, sorted unless `consecutive`."""
    var n = t.numel
    var device = t.device
    # UniqueCub<bool> counts the trues instead of sorting, and always
    # returns the counts.
    var counts_out = want_counts or (t.dtype == DType.bool and not consecutive)
    if n == 0:
        var inv = own(new_tensor(t.shape, t.rank, ST_INT64, device))
        return (_empty(0, t.stype, device), inv^, _empty(0, ST_INT64, device))
    var flat = _flat_view(t)
    var perm = Optional[Owned](None)
    if not consecutive:
        perm = _stable_sort_perm(flat.t)
    var perm_t = Optional[T](None)
    if perm:
        perm_t = perm.value().t.copy()
    var g = _groups(
        flat.t,
        perm_t,
        n,
        1,
        True,
        False,
        want_inverse,
        counts_out,
        want_counts,
    )
    _ = perm^
    _ = flat^
    var inv: Owned
    if want_inverse:
        var flat_inv = g.inverse.take()
        inv = own(
            view_strided(
                flat_inv.t,
                t.shape,
                contiguous_strides(t.shape, t.rank),
                t.rank,
                0,
            )
        )
        _ = flat_inv^
    else:
        inv = _empty(0, ST_INT64, device)
    var counts = g.counts.take() if counts_out else _empty(0, ST_INT64, device)
    return (g.out_vals.take(), inv^, counts^)


def _wrap_dim(dim: Int, rank: Int) raises -> Int:
    var r = max(rank, 1)
    if dim < -r or dim >= r:
        index_error(
            "Dimension out of range (expected to be in range of ["
            + String(-r)
            + ", "
            + String(r - 1)
            + "], but got "
            + String(dim)
            + ")"
        )
    return dim + r if dim < 0 else dim


def unique_rows(
    t: T,
    dim_in: Int,
    consecutive: Bool,
    want_inverse: Bool,
    want_counts: Bool,
) raises -> Tuple[Owned, Owned, Owned]:
    """`unique_dim_cuda_template`: the unique slices of `t` along `dim`."""
    var device = t.device
    if t.rank == 0:
        index_error(
            "Dimension specified as "
            + String(dim_in)
            + " but tensor has no dimensions"
        )
    var dim = _wrap_dim(dim_in, t.rank)
    var zero_dims = 0
    for d in range(t.rank):
        if t.dim(d) == 0:
            zero_dims += 1
    if t.dim(dim) == 0:
        if zero_dims != 1:
            raise Error(
                "Number of zero sized dimensions is more than one, so unique"
                " cannot be applied "
            )
        return (
            own(new_tensor(t.shape, t.rank, t.stype, device)),
            _empty(0, ST_INT64, device),
            _empty(0, ST_INT64, device),
        )
    if zero_dims != 0:
        raise Error(
            "There are 0 sized dimensions, and they aren't selected, so unique"
            " cannot be applied"
        )
    var num_inp = t.dim(dim)
    var inner = t.numel // num_inp
    # input_flat = self.moveaxis(dim, 0).contiguous().view({num_inp, -1})
    var moved = _op("aten::movedim", "int", [tensor_arg(t), _int(dim), _int(0)])
    var c = own_if_new(contiguous(moved.t), moved.t)
    var rows_shape = IndexList[MAX_RANK](1)
    rows_shape[MAX_RANK - 2] = num_inp
    rows_shape[MAX_RANK - 1] = inner
    var flat = own(
        view_strided(
            c.t, rows_shape, contiguous_strides(rows_shape, 2), 2, c.t.offset
        )
    )
    var perm = Optional[Owned](None)
    if not consecutive:
        var order = _empty(num_inp, ST_INT64, device)
        _ = call_op(
            "aten::arange",
            "start_out",
            [
                Value(TAG_SCALAR_INT, 0, 0, 0),
                Value(TAG_SCALAR_INT, 0, Int64(num_inp), 0),
                Value(TAG_SCALAR_INT, 0, 1, 0),
                tensor_arg(order.t),
            ],
            1,
        )
        # One stable merge sort of the row indices by lexicographic row
        # order (CUDA sorts them with a row comparator): log2(n) passes.
        var other = _empty(num_inp, ST_INT64, device)
        var ctx = ctx_for(device)
        var width = 1
        while width < num_inp:
            var mc = KernelCall("unique", "RowMergePass")
            mc.arg_dtype(0, flat.t.dtype)
            mc.int(other.t.ptr)
            mc.int(order.t.ptr)
            mc.int(flat.t.ptr)
            mc.tuple([num_inp, inner, width])
            mc.int(dtype_code(flat.t.dtype))
            mc.int(ctx_ptr(ctx))
            mc.run()
            var tmp = order^
            order = other^
            other = tmp^
            width *= 2
        _ = other^
        _ = ctx
        perm = order^
    var perm_t = Optional[T](None)
    if perm:
        perm_t = perm.value().t.copy()
    var g = _groups(
        flat.t,
        perm_t,
        num_inp,
        inner,
        False,
        True,
        want_inverse,
        want_counts,
        False,
    )
    _ = perm^
    _ = flat^
    _ = c^
    _ = moved^
    var values = _op(
        "aten::index_select",
        "",
        [tensor_arg(t), _int(dim), tensor_arg(g.out_rows.value().t)],
    )
    var inv = g.inverse.take() if want_inverse else _empty(0, ST_INT64, device)
    var counts = g.counts.take() if want_counts else _empty(0, ST_INT64, device)
    return (values^, inv^, counts^)


def _ret3(rets: Values, n_rets: Int, var r: Tuple[Owned, Owned, Owned]):
    var a = own(r[0].take())
    var b = own(r[1].take())
    var c = own(r[2].take())
    ret_owned(rets, 0, a)
    ret_owned(rets, 1, b)
    if n_rets > 2:
        ret_owned(rets, 2, c)


# aten::_unique(Tensor self, bool sorted=True, bool return_inverse=False)
#   -> (Tensor, Tensor)
def op_unique(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    _check_dtype(t, "_unique")
    var r = unique_flat(t, False, v_bool(args[unsafe_offset=2]), False)
    _ret3(rets, 2, r^)


# aten::_unique2(Tensor self, bool sorted=True, bool return_inverse=False,
#   bool return_counts=False) -> (Tensor, Tensor, Tensor)
def op_unique2(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    _check_dtype(t, "_unique2")
    var r = unique_flat(
        t, False, v_bool(args[unsafe_offset=2]), v_bool(args[unsafe_offset=3])
    )
    _ret3(rets, 3, r^)


# aten::unique_dim(Tensor self, int dim, bool sorted=True,
#   bool return_inverse=False, bool return_counts=False)
#   -> (Tensor, Tensor, Tensor)
def op_unique_dim(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    _check_dtype(t, "unique_dim")
    var r = unique_rows(
        t,
        v_int(args[unsafe_offset=1]),
        False,
        v_bool(args[unsafe_offset=3]),
        v_bool(args[unsafe_offset=4]),
    )
    _ret3(rets, 3, r^)


# aten::unique_consecutive(Tensor self, bool return_inverse=False,
#   bool return_counts=False, int? dim=None) -> (Tensor, Tensor, Tensor)
def op_unique_consecutive(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    _check_dtype(t, "unique_consecutive")
    var inverse = v_bool(args[unsafe_offset=1])
    var counts = v_bool(args[unsafe_offset=2])
    if v_is_none(args[unsafe_offset=3]):
        var r = unique_flat(t, True, inverse, counts)
        _ret3(rets, 3, r^)
    else:
        var r = unique_rows(
            t, v_int(args[unsafe_offset=3]), True, inverse, counts
        )
        _ret3(rets, 3, r^)


# aten::unique_dim_consecutive(Tensor self, int dim, bool return_inverse=False,
#   bool return_counts=False) -> (Tensor, Tensor, Tensor)
def op_unique_dim_consecutive(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    _check_dtype(t, "unique_dim_consecutive")
    var r = unique_rows(
        t,
        v_int(args[unsafe_offset=1]),
        True,
        v_bool(args[unsafe_offset=2]),
        v_bool(args[unsafe_offset=3]),
    )
    _ret3(rets, 3, r^)


def register_unique(site: Site) raises:
    impl[op_unique, "_unique"](site)
    impl[op_unique2, "_unique2"](site)
    impl[op_unique_dim, "unique_dim"](site)
    impl[op_unique_consecutive, "unique_consecutive"](site)
    impl[op_unique_dim_consecutive, "unique_dim_consecutive"](site)
