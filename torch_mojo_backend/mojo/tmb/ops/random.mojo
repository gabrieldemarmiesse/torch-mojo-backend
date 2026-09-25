"""Random group of aten ops: every op that consumes the device Philox stream.

Draws are bit-identical to stock CUDA (torch 2.11, curand 12.8) for the same
seed and generator state: the generator counts in curand's unit
(shim_runtime.cpp), elements are visited in TensorIterator's order (`_plan`:
dims sorted by stride, coalesced), the launch is ATen's (256 threads, grid
capped at sm_count * max_threads_per_sm / 256, one reservation of
`((n - 1) / (256 * grid * unroll) + 1) * 4` per launch) and the kernels are
curand's (tmb/kernels/random). A tensor whose
byte extent exceeds INT32_MAX is split the way `with_32bit_indexing` does:
largest-extent dim halved, first half first, one launch and reservation per
piece after the reservation the unsplit call made. Consequences worth
knowing: the grid cap makes draws above ~sm_count * 8 * 256 * unroll elements
depend on the GPU model, exactly as they do on CUDA, and a `bernoulli_.Tensor`
call advances the offset by 12 (curand asks for 10, rounded up).
"""
from std.math import ceildiv
from std.utils import IndexList

from max.gpu.host import DeviceAttribute

from tmb.backend.abi import (
    Owned,
    TAG_SCALAR_INT,
    TAG_TENSOR,
    ST_BOOL,
    ST_FLOAT32,
    ST_FLOAT64,
    ST_INT32,
    ST_INT64,
    T,
    Value,
    Values,
    bool_arg,
    cpu_empty,
    dtype_code,
    call_op,
    int_arg,
    contiguous_strides,
    new_like,
    new_scalar,
    new_strided,
    new_tensor,
    own,
    own_if_new,
    retain,
    ret_owned,
    ret_ref,
    unsupported,
    v_bool_or,
    v_f64,
    v_f64_or,
    v_generator,
    v_int,
    v_is_none,
    v_tensor,
)
from tmb.backend.device import copy_d2d, copy_to_host, ctx_for, ctx_ptr, dev
from tmb.backend.kernel_call import KernelCall
from tmb.kernels.common.op_utils import MAX_RANK, _device_attr_cached
from tmb.ops.common import (
    broadcast_shape,
    cast_to,
    contiguous,
    copy_strided_into,
    fill_value,
    philox_reserve,
    resize_out,
)
from tmb.ops.data_movement import _scalar_type_name
from tmb.backend.registry import Site, impl

comptime INT32_MAX = 2147483647
comptime INT64_MIN = -9223372036854775808
comptime INT64_MAX = 9223372036854775807
comptime BLOCK = 256


# ---------------------------------------------------------------------------
# TensorIterator's view of one output tensor: reorder_dimensions +
# coalesce_dimensions (aten/src/ATen/TensorIterator.cpp), fastest dim first.
# ---------------------------------------------------------------------------


struct Plan(Copyable, Movable):
    var ndim: Int
    var sizes: IndexList[MAX_RANK]
    var strides: IndexList[MAX_RANK]  # elements
    var numel: Int
    var base: Int  # element offset of this piece (split halves)

    def __init__(out self):
        self.ndim = 0
        self.sizes = IndexList[MAX_RANK](1)
        self.strides = IndexList[MAX_RANK](0)
        self.numel = 1
        self.base = 0

    def size_list(self) -> List[Int]:
        var out = List[Int]()
        for d in range(self.ndim):
            out.append(self.sizes[d])
        return out^

    def stride_list(self) -> List[Int]:
        var out = List[Int]()
        for d in range(self.ndim):
            out.append(self.strides[d])
        return out^


def _plan(t: T) raises -> Plan:
    var p = Plan()
    var r = t.rank
    p.numel = t.numel
    if r == 0:
        p.ndim = 1
        p.sizes[0] = 1
        p.strides[0] = 1
        return p^
    var sizes = IndexList[MAX_RANK](1)
    var strides = IndexList[MAX_RANK](0)
    for i in range(r):
        sizes[i] = t.dim(i)
        strides[i] = t.stride(i)
        if sizes[i] > 1 and strides[i] == 0:
            raise Error(
                "unsupported operation: more than one element of the written-to"
                " tensor refers to a single memory location. Please clone()"
                " the tensor before performing the operation."
            )
    # reorder_dimensions: insertion sort of perm (initially reversed) with
    # ambiguous comparisons; stride 0 skips, equal strides break ties on size.
    var perm = IndexList[MAX_RANK](0)
    for i in range(r):
        perm[i] = r - 1 - i

    @always_inline
    @parameter
    def should_swap(d0: Int, d1: Int) -> Int:
        var s0 = strides[d0]
        var s1 = strides[d1]
        if s0 == 0 or s1 == 0:
            return 0
        if s0 < s1:
            return -1
        if s0 > s1:
            return 1
        if sizes[d0] > sizes[d1]:
            return 1
        return 0

    for i in range(1, r):
        var dim1 = i
        var dim0 = i - 1
        while dim0 >= 0:
            var c = should_swap(perm[dim0], perm[dim1])
            if c > 0:
                var tmp = perm[dim0]
                perm[dim0] = perm[dim1]
                perm[dim1] = tmp
                dim1 = dim0
            elif c < 0:
                break
            dim0 -= 1
    for k in range(r):
        p.sizes[k] = sizes[perm[k]]
        p.strides[k] = strides[perm[k]]
    # coalesce_dimensions
    var prev = 0
    for dim in range(1, r):
        var s0 = p.sizes[prev]
        var s1 = p.sizes[dim]
        var can = s0 == 1 or s1 == 1 or s0 * p.strides[prev] == p.strides[dim]
        if can:
            if s0 == 1:
                p.strides[prev] = p.strides[dim]
            p.sizes[prev] = s0 * s1
        else:
            prev += 1
            if prev != dim:
                p.strides[prev] = p.strides[dim]
                p.sizes[prev] = p.sizes[dim]
    p.ndim = prev + 1
    for d in range(p.ndim, MAX_RANK):
        p.sizes[d] = 1
        p.strides[d] = 0
    return p^


def _fits_32bit(p: Plan, itemsize: Int) -> Bool:
    if p.numel > INT32_MAX:
        return False
    var max_offset = 1
    for d in range(p.ndim):
        max_offset += (p.sizes[d] - 1) * p.strides[d] * itemsize
    return max_offset <= INT32_MAX


def _dim_to_split(p: Plan, itemsize: Int) -> Int:
    var max_extent = -1
    var dim_to_split = -1
    var dim = p.ndim - 1
    while dim >= 0:
        var size = p.sizes[dim]
        if size != 0:
            var extent = (size - 1) * abs(p.strides[dim] * itemsize)
            if extent > max_extent:
                max_extent = extent
                dim_to_split = dim
        dim -= 1
    return dim_to_split


def _pieces(p: Plan, itemsize: Int, mut out: List[Plan]):
    """SplitUntil32Bit: halve the largest-extent dim, first half first."""
    if _fits_32bit(p, itemsize) or p.numel == 0:
        out.append(p.copy())
        return
    var dim = _dim_to_split(p, itemsize)
    var first = p.copy()
    var second = p.copy()
    var first_size = p.sizes[dim] // 2
    first.sizes[dim] = first_size
    first.numel = p.numel // p.sizes[dim] * first_size
    second.sizes[dim] = p.sizes[dim] - first_size
    second.numel = p.numel - first.numel
    second.base = p.base + first_size * p.strides[dim]
    _pieces(first, itemsize, out)
    _pieces(second, itemsize, out)


# ---------------------------------------------------------------------------
# calc_execution_policy (cuda/DistributionTemplates.h)
# ---------------------------------------------------------------------------


def _grid(device: Int, numel: Int) raises -> Int:
    # Constant fallbacks (an H100 SXM) for a query that fails to answer;
    # every real accelerator answers these, so they are not expected to fire.
    var ctx = ctx_for(device)
    var max_threads = _device_attr_cached["maxthr"](
        ctx, DeviceAttribute.MAX_THREADS_PER_MULTIPROCESSOR, 2048
    )
    var sm = _device_attr_cached["sm"](
        ctx, DeviceAttribute.MULTIPROCESSOR_COUNT, 132
    )
    return min(ceildiv(numel, BLOCK), sm * (max_threads // BLOCK))


def _counter_offset(numel: Int, grid: Int, unroll: Int) -> Int:
    return ((numel - 1) // (BLOCK * grid * unroll) + 1) * 4


def _unroll(op: StaticString, dtype: DType) -> Int:
    if op == "RandomFromTo64" or op == "RandomFull64" or op == "Random64":
        return 2
    if op == "RandomFromTo32" or op == "Random32":
        return 4
    return 2 if dtype == DType.float64 else 4


def _draw(
    t: T,
    op: StaticString,
    pf0: Float64,
    pf1: Float64,
    pi0: Int,
    pi1: Int,
    generator: Int,
) raises:
    """Run distribution `op` over `t` in TensorIterator order, reserving the
    generator exactly as ATen's distribution_nullary_kernel does."""
    if t.numel == 0:
        return
    var plan = _plan(t)
    var pieces = List[Plan]()
    _pieces(plan, t.itemsize, pieces)
    var unroll = _unroll(op, t.dtype)
    var ctx = ctx_for(t.device)
    var cp = ctx_ptr(ctx)
    if len(pieces) > 1:
        # The unsplit call reserves before it discovers it must split.
        _ = philox_reserve(
            generator,
            t.device,
            _counter_offset(plan.numel, _grid(t.device, plan.numel), unroll),
        )
    for piece in pieces:
        var grid = _grid(t.device, piece.numel)
        var seed_offset = philox_reserve(
            generator, t.device, _counter_offset(piece.numel, grid, unroll)
        )
        var call = KernelCall("random", String(op))
        call.out_dtype(t.dtype)
        call.int(t.ptr + piece.base * t.itemsize)
        call.int(piece.numel)
        call.int(piece.ndim)
        call.tuple(piece.size_list())
        call.tuple(piece.stride_list())
        call.int(grid)
        call.f64(pf0)
        call.f64(pf1)
        call.int(pi0)
        call.int(pi1)
        call.int(Int(seed_offset[0] & 0xFFFFFFFF))
        call.int(Int((seed_offset[0] >> 32) & 0xFFFFFFFF))
        call.int(Int(seed_offset[1] & 0xFFFFFFFF))
        call.int(Int((seed_offset[1] >> 32) & 0xFFFFFFFF))
        call.int(dtype_code(t.dtype))
        call.int(cp)
        call.run()
    _ = ctx


def _is_floating(dt: DType) -> Bool:
    return (
        dt == DType.float32
        or dt == DType.float16
        or dt == DType.bfloat16
        or dt == DType.float64
    )


def _is_wide_unsigned(dt: DType) -> Bool:
    return dt == DType.uint16 or dt == DType.uint32 or dt == DType.uint64


def _check_device_dtype(
    t: T, op: StaticString, floating_only: Bool, wide_unsigned: Bool = False
) raises:
    if floating_only and not _is_floating(t.dtype):
        unsupported(String(op) + " of dtype " + String(t.dtype))
    if not (
        _is_floating(t.dtype)
        or t.dtype == DType.int64
        or t.dtype == DType.int32
        or t.dtype == DType.int16
        or t.dtype == DType.int8
        or t.dtype == DType.uint8
        or t.dtype == DType.bool
        or (wide_unsigned and _is_wide_unsigned(t.dtype))
    ):
        unsupported(String(op) + " of dtype " + String(t.dtype))
    if t.dtype == DType.float64 and dev(t.device)[].api == "metal":
        unsupported(String(op) + " of dtype float64 on Apple GPU")


def _lowest_highest(dtype: DType) -> Tuple[Float64, Float64]:
    if dtype == DType.float16:
        return (-65504.0, 65504.0)
    if dtype == DType.bfloat16:
        return (-3.3895313892515355e38, 3.3895313892515355e38)
    if dtype == DType.float64:
        return (-1.7976931348623157e308, 1.7976931348623157e308)
    return (-3.4028234663852886e38, 3.4028234663852886e38)


# aten::uniform_(Tensor(a!) self, float from=0., float to=1., *, Generator? generator=None) -> Tensor(a!)
def op_uniform_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var from_ = v_f64_or(args[unsafe_offset=1], 0.0)
    var to = v_f64_or(args[unsafe_offset=2], 1.0)
    _check_device_dtype(t, "uniform_", True)
    var bounds = _lowest_highest(t.dtype)
    var lowest = bounds[0]
    var highest = bounds[1]
    # uniform_impl_'s checks, in ATen's order.
    if not (from_ >= lowest and from_ <= highest):
        raise Error("from is out of bounds for ", String(t.dtype))
    if not (to >= lowest and to <= highest):
        raise Error("to is out of bounds for ", String(t.dtype))
    if not from_ <= to:
        raise Error(
            "uniform_ expects to return a [from, to) range, but found from=",
            from_,
            " > to=",
            to,
        )
    if not (to - from_) <= highest:
        raise Error(
            "uniform_ expects to-from <= the numeric limit for ",
            String(t.dtype),
            ", but found to=",
            to,
            " and from=",
            from_,
            " which result in to-from to exceed the limit",
        )
    _draw(t, "Uniform", from_, to, 0, 0, v_generator(args[unsafe_offset=3]))
    ret_ref(rets, 0, t)


# aten::normal_(Tensor(a!) self, float mean=0., float std=1., *, Generator? generator=None) -> Tensor(a!)
def op_normal_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var mean = v_f64_or(args[unsafe_offset=1], 0.0)
    var std = v_f64_or(args[unsafe_offset=2], 1.0)
    _check_device_dtype(t, "normal_", True)
    if not std >= 0.0:
        raise Error("normal expects std >= 0.0, but found std ", std)
    _draw(t, "Normal", mean, std, 0, 0, v_generator(args[unsafe_offset=3]))
    ret_ref(rets, 0, t)


# ---------------------------------------------------------------------------
# torch.normal with tensor arguments (native/DistributionTemplates.h
# normal_impl / normal_out_impl): a fresh contiguous N(0, 1) (or N(0, std))
# draw, then mul_/add_ through the dispatcher, exactly as the composite does.
# ---------------------------------------------------------------------------


def _tensor_value(t: T) -> Value:
    return Value(TAG_TENSOR, 0, Int64(t.h), 0)


def _add_(dst: T, other: T) raises:
    _ = call_op(
        "aten::add_",
        "Tensor",
        [
            _tensor_value(dst),
            _tensor_value(other),
            Value(TAG_SCALAR_INT, 0, 1, 0),
        ],
        1,
    )


def _mul_(dst: T, other: T) raises:
    _ = call_op(
        "aten::mul_", "Tensor", [_tensor_value(dst), _tensor_value(other)], 1
    )


def _check_std_tensor(std: T) raises:
    """CHECK_NORMAL_TENSOR_STD: every element >= 0."""
    if std.numel == 0:
        return
    # `mn` owns the min tensor through the second call and releases it on
    # every path (Results.__deinit__); an Owned local would die before it.
    var mn = call_op("aten::min", "", [_tensor_value(std)], 1)
    var ok = call_op("aten::_local_scalar_dense", "", [mn[0]], 1)
    _ = mn^
    if not v_f64(ok[0]) >= 0.0:
        raise Error("normal expects all elements of std >= 0.0")


# aten::normal.Tensor_float(Tensor mean, float std=1, *, Generator? generator=None) -> Tensor
def op_normal_tensor_float(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var mean = v_tensor(args[unsafe_offset=0])
    var std = v_f64_or(args[unsafe_offset=1], 1.0)
    _check_device_dtype(mean, "normal", True)
    if not std >= 0.0:
        raise Error("normal expects std >= 0.0, but found std ", std)
    var ret = own(new_like(mean))
    _draw(ret.t, "Normal", 0.0, std, 0, 0, v_generator(args[unsafe_offset=2]))
    _add_(ret.t, mean)
    ret_owned(rets, 0, ret)


# aten::normal.float_Tensor(float mean, Tensor std, *, Generator? generator=None) -> Tensor
def op_normal_float_tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var mean = v_f64(args[unsafe_offset=0])
    var std = v_tensor(args[unsafe_offset=1])
    _check_device_dtype(std, "normal", True)
    _check_std_tensor(std)
    var ret = own(new_like(std))
    _draw(ret.t, "Normal", 0.0, 1.0, 0, 0, v_generator(args[unsafe_offset=2]))
    _mul_(ret.t, std)
    var mean_t = own(new_scalar(std.stype, std.device))
    fill_value(mean_t.t, mean)
    _add_(ret.t, mean_t.t)
    ret_owned(rets, 0, ret)


# aten::normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor
def op_normal_tensor_tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var mean = v_tensor(args[unsafe_offset=0])
    var std = v_tensor(args[unsafe_offset=1])
    _check_device_dtype(mean, "normal", True)
    _check_std_tensor(std)
    var shape = broadcast_shape(mean, std)
    var ret = own(
        new_tensor(shape, max(mean.rank, std.rank), mean.stype, mean.device)
    )
    _draw(ret.t, "Normal", 0.0, 1.0, 0, 0, v_generator(args[unsafe_offset=2]))
    _mul_(ret.t, std)
    _add_(ret.t, mean)
    ret_owned(rets, 0, ret)


# aten::normal.Tensor_float_out(Tensor mean, float std=1, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
def op_normal_tensor_float_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var mean = v_tensor(args[unsafe_offset=0])
    var std = v_f64_or(args[unsafe_offset=1], 1.0)
    var out = v_tensor(args[unsafe_offset=3])
    _check_device_dtype(out, "normal", True)
    if not std >= 0.0:
        raise Error("normal expects std >= 0.0, but found std ", std)
    # normal_out_impl: shape = infer_size(mean, empty_like(out)).
    resize_out(out, broadcast_shape(mean, out), max(mean.rank, out.rank))
    _draw(out, "Normal", 0.0, std, 0, 0, v_generator(args[unsafe_offset=2]))
    _add_(out, mean)
    ret_ref(rets, 0, out)


# aten::normal.float_Tensor_out(float mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
def op_normal_float_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var mean = v_f64(args[unsafe_offset=0])
    var std = v_tensor(args[unsafe_offset=1])
    var out = v_tensor(args[unsafe_offset=3])
    _check_device_dtype(out, "normal", True)
    _check_std_tensor(std)
    resize_out(out, std.shape, std.rank)
    _draw(out, "Normal", 0.0, 1.0, 0, 0, v_generator(args[unsafe_offset=2]))
    _mul_(out, std)
    var mean_t = own(new_scalar(out.stype, out.device))
    fill_value(mean_t.t, mean)
    _add_(out, mean_t.t)
    ret_ref(rets, 0, out)


# aten::normal.Tensor_Tensor_out(Tensor mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
def op_normal_tensor_tensor_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var mean = v_tensor(args[unsafe_offset=0])
    var std = v_tensor(args[unsafe_offset=1])
    var out = v_tensor(args[unsafe_offset=3])
    _check_device_dtype(out, "normal", True)
    _check_std_tensor(std)
    resize_out(out, broadcast_shape(mean, std), max(mean.rank, std.rank))
    _draw(out, "Normal", 0.0, 1.0, 0, 0, v_generator(args[unsafe_offset=2]))
    _mul_(out, std)
    _add_(out, mean)
    ret_ref(rets, 0, out)


# aten::log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator? generator=None) -> Tensor(a!)
def op_log_normal_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var mean = v_f64_or(args[unsafe_offset=1], 1.0)
    var std = v_f64_or(args[unsafe_offset=2], 2.0)
    _check_device_dtype(t, "log_normal_", True)
    if not std > 0.0:
        raise Error("log_normal_ expects std > 0.0, but found std=", std)
    _draw(t, "LogNormal", mean, std, 0, 0, v_generator(args[unsafe_offset=3]))
    ret_ref(rets, 0, t)


# aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)
def op_cauchy_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var median = v_f64_or(args[unsafe_offset=1], 0.0)
    var sigma = v_f64_or(args[unsafe_offset=2], 1.0)
    _check_device_dtype(t, "cauchy_", True)
    if not sigma > 0.0:
        raise Error("cauchy_ expects sigma > 0.0, but found sigma=", sigma)
    _draw(t, "Cauchy", median, sigma, 0, 0, v_generator(args[unsafe_offset=3]))
    ret_ref(rets, 0, t)


# aten::exponential_(Tensor(a!) self, float lambd=1, *, Generator? generator=None) -> Tensor(a!)
def op_exponential_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var lambd = v_f64_or(args[unsafe_offset=1], 1.0)
    _check_device_dtype(t, "exponential_", True)
    if not lambd > 0.0:
        raise Error(
            "exponential_ expects lambda > 0.0, but found lambda=", lambd
        )
    _draw(
        t, "Exponential", lambd, 0.0, 0, 0, v_generator(args[unsafe_offset=2])
    )
    ret_ref(rets, 0, t)


# aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
def op_geometric_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var p = v_f64(args[unsafe_offset=1])
    _check_device_dtype(t, "geometric_", False)
    if not (0.0 < p and p < 1.0):
        raise Error("geometric_ expects p to be in (0, 1), but got p=", p)
    _draw(t, "Geometric", p, 0.0, 0, 0, v_generator(args[unsafe_offset=2]))
    ret_ref(rets, 0, t)


# aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)
def op_bernoulli_float(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var p = v_f64_or(args[unsafe_offset=1], 0.5)
    _check_device_dtype(t, "bernoulli_", False)
    if not (0.0 <= p and p <= 1.0):
        raise Error("bernoulli_ expects p to be in [0, 1], but got p=", p)
    _draw(t, "Bernoulli", p, 0.0, 0, 0, v_generator(args[unsafe_offset=2]))
    ret_ref(rets, 0, t)


# aten::bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)
def op_bernoulli_tensor(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var p_in = v_tensor(args[unsafe_offset=1])
    var generator = v_generator(args[unsafe_offset=2])
    _check_device_dtype(t, "bernoulli_", False)
    if not _is_floating(p_in.dtype):
        raise Error(
            "expected probabilities tensor to have floating type, got ",
            String(p_in.dtype),
        )
    if p_in.device != t.device:
        unsupported("bernoulli_.Tensor with p on another device")
    if t.numel == 0:
        ret_ref(rets, 0, t)
        return
    for i in range(t.rank):
        if t.dim(i) > 1 and t.stride(i) == 0:
            raise Error(
                "unsupported operation: more than one element of the written-to"
                " tensor refers to a single memory location. Please clone()"
                " the tensor before performing the operation."
            )
    # p as float (double for a double self), expanded to self's shape. Our
    # copy is contiguous; CUDA's `.to(dtype)` keeps a dense p's strides, and
    # rearrangeDims below decides on THOSE strides, so they are tracked
    # separately from the strides the kernel reads with.
    var p_stype = ST_FLOAT64 if t.dtype == DType.float64 else ST_FLOAT32
    var p = own_if_new(cast_to(p_in, p_stype), p_in)
    var p_keeps_layout = p_in.stype == p_stype or _is_dense(p_in)
    if p_in.rank > t.rank:
        raise Error("bernoulli_: p has more dimensions than self")
    # CUDA_tensor_apply2 computes into a contiguous temporary when self may
    # have overlapping indices, then copies back.
    var overlapping = _maybe_overlapping(t)
    var dst = own(new_like(t)) if overlapping else own(T(retain(t)))
    var r = t.rank
    var sizes = IndexList[MAX_RANK](1)
    var t_strides = IndexList[MAX_RANK](0)
    var p_strides = IndexList[MAX_RANK](0)  # what the kernel reads with
    var p_cuda = IndexList[MAX_RANK](0)  # what CUDA's rearrangeDims sees
    var p_contig = contiguous_strides(p_in.shape, p_in.rank)
    for i in range(r):
        sizes[i] = t.dim(i)
        t_strides[i] = dst.t.stride(i)
        var pi = i - (r - p_in.rank)
        if pi >= 0:
            var ps = p.t.dim(pi)
            if ps == sizes[i]:
                p_strides[i] = p.t.stride(pi)
                p_cuda[i] = p_in.stride(pi) if p_keeps_layout else p_contig[
                    MAX_RANK - p_in.rank + pi
                ]
            elif ps == 1:
                p_strides[i] = 0
                p_cuda[i] = 0
            else:
                raise Error(
                    "bernoulli_: p of size ",
                    ps,
                    " is not broadcastable to self at dim ",
                    i,
                )
    # rearrangeDims: swap (i, j) when every tensor's strides strictly increase.
    for i in range(r - 1):
        if sizes[i] == 1:
            continue
        for j in range(i + 1, r):
            if sizes[j] == 1:
                continue
            var inc = t_strides[i] < t_strides[j] or p_cuda[i] < p_cuda[j]
            var dec = t_strides[i] > t_strides[j] or p_cuda[i] > p_cuda[j]
            if inc and not dec:
                var s = sizes[i]
                sizes[i] = sizes[j]
                sizes[j] = s
                var a = t_strides[i]
                t_strides[i] = t_strides[j]
                t_strides[j] = a
                var b = p_strides[i]
                p_strides[i] = p_strides[j]
                p_strides[j] = b
                var c = p_cuda[i]
                p_cuda[i] = p_cuda[j]
                p_cuda[j] = c
    var size_l = List[Int]()
    var ts_l = List[Int]()
    var ps_l = List[Int]()
    if r == 0:
        size_l.append(1)
        ts_l.append(0)
        ps_l.append(0)
    var i = r - 1
    while i >= 0:  # fastest first
        size_l.append(sizes[i])
        ts_l.append(t_strides[i])
        ps_l.append(p_strides[i])
        i -= 1
    var grid = ceildiv(t.numel, 512 * 4)
    var seed_offset = philox_reserve(generator, t.device, 10)
    var ctx = ctx_for(t.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("random", "BernoulliTensor")
    call.out_dtype(t.dtype)
    call.arg_dtype(0, p.t.dtype)
    call.int(dst.t.ptr)
    call.int(p.t.ptr)
    call.int(t.numel)
    call.int(len(size_l))
    call.tuple(size_l)
    call.tuple(ts_l)
    call.tuple(ps_l)
    call.int(grid)
    call.int(Int(seed_offset[0] & 0xFFFFFFFF))
    call.int(Int((seed_offset[0] >> 32) & 0xFFFFFFFF))
    call.int(Int(seed_offset[1] & 0xFFFFFFFF))
    call.int(Int((seed_offset[1] >> 32) & 0xFFFFFFFF))
    call.int(dtype_code(t.dtype))
    call.int(dtype_code(p.t.dtype))
    call.int(cp)
    call.run()
    _ = ctx
    if overlapping:
        copy_strided_into(t, dst.t)
    _ = dst^
    _ = p^  # alive past the launch
    ret_ref(rets, 0, t)


# ---------------------------------------------------------------------------
# random_ family (native/DistributionTemplates.h random_from_to_impl /
# random_impl, cuda/DistributionTemplates.h random_*_kernel)
# ---------------------------------------------------------------------------


def _digits(dtype: DType) -> Int:
    if dtype == DType.float64:
        return 53
    if dtype == DType.float32:
        return 24
    if dtype == DType.float16:
        return 11
    return 8  # bfloat16


def _round_trip(dtype: DType, v: Int) -> Int:
    """`static_cast<int64_t>(static_cast<scalar_t>(v))` for a float dtype."""
    if dtype == DType.float64:
        return Int(Float64(v))
    if dtype == DType.float32:
        return Int(Float64(Float32(v)))
    if dtype == DType.float16:
        return Int(
            Float64(Float32(v).cast[DType.float16]().cast[DType.float32]())
        )
    return Int(Float64(Float32(v).cast[DType.bfloat16]().cast[DType.float32]()))


def _update_from(dtype: DType, from_: Int) -> Int:
    var from_plus_1 = _round_trip(dtype, from_ + 1)
    if from_plus_1 < from_:
        var f = abs(from_ + 1)
        var n = 0
        while f >> 1 != 0:
            f >>= 1
            n += 1
        return from_plus_1 + (1 << (n - _digits(dtype) + 1))
    return from_


def _update_to(dtype: DType, to: Int) -> Int:
    var to_minus_1 = _round_trip(dtype, to - 1)
    if to_minus_1 >= to:
        var f = abs(to - 1)
        var n = 0
        while f >> 1 != 0:
            f >>= 1
            n += 1
        return to_minus_1 - (1 << (n - _digits(dtype) + 1))
    return to


def _int_lowest_highest(dtype: DType) -> Tuple[Int, Int]:
    if dtype == DType.int64:
        return (INT64_MIN, INT64_MAX)
    if dtype == DType.int32:
        return (-2147483648, 2147483647)
    if dtype == DType.int16:
        return (-32768, 32767)
    if dtype == DType.int8:
        return (-128, 127)
    if dtype == DType.uint8:
        return (0, 255)
    if dtype == DType.uint16:
        return (0, 65535)
    if dtype == DType.uint32:
        return (0, 4294967295)
    if dtype == DType.uint64:
        return (0, INT64_MAX)  # check_from_to_in_range's kUInt64 arm
    return (0, 1)  # bool


def _check_from_to_in_range(from_: Int, to_inc: Int, dtype: DType) raises:
    if _is_floating(dtype):
        var b = _lowest_highest(dtype)
        if not (Float64(from_) >= b[0] and Float64(from_) <= b[1]):
            raise Error("from is out of bounds for ", String(dtype))
        if not (Float64(to_inc) >= b[0] and Float64(to_inc) <= b[1]):
            raise Error("to - 1 is out of bounds for ", String(dtype))
    else:
        var b = _int_lowest_highest(dtype)
        if not (from_ >= b[0] and from_ <= b[1]):
            raise Error("from is out of bounds for ", String(dtype))
        if not (to_inc >= b[0] and to_inc <= b[1]):
            raise Error("to - 1 is out of bounds for ", String(dtype))


def _random_from_to(t: T, range_: UInt64, base: Int, generator: Int) raises:
    var op: StaticString = (
        "RandomFromTo64" if range_ >= (UInt64(1) << 28) else "RandomFromTo32"
    )
    _draw(
        t, op, 0.0, 0.0, Int(Int64(range_.cast[DType.int64]())), base, generator
    )


def _random_from_to_impl(t: T, from_: Int, to_v: Value, generator: Int) raises:
    var lo = from_
    if not v_is_none(to_v):
        var to = v_int(to_v)
        if not lo < to:
            raise Error(
                "random_ expects 'lo' to be less than 'to', but got lo=",
                lo,
                " >= to=",
                to,
            )
        if _is_floating(t.dtype):
            lo = _update_from(t.dtype, lo)
            to = _update_to(t.dtype, to)
            if not lo < to:
                raise Error(
                    (
                        "random_ expects 'lo' casted to dtype to be less than"
                        " 'to' casted to dtype, but got lo="
                    ),
                    lo,
                    " >= to=",
                    to,
                )
        _check_from_to_in_range(lo, to - 1, t.dtype)
        if t.numel == 0:
            return
        var range_ = UInt64(Int64(to).cast[DType.uint64]()) - UInt64(
            Int64(lo).cast[DType.uint64]()
        )
        _random_from_to(t, range_, lo, generator)
    elif lo != INT64_MIN:
        var to_inc: Int
        if _is_floating(t.dtype):
            var d = _digits(t.dtype)
            to_inc = INT64_MAX if d >= 63 else (1 << d)
            lo = _update_from(t.dtype, lo)
            if not lo < to_inc:
                raise Error(
                    (
                        "random_ expects 'lo' casted to dtype to be less than"
                        " or equal to 'to_inc' casted to dtype, but got lo="
                    ),
                    lo,
                    " > to_inc=",
                    to_inc,
                )
        elif t.dtype == DType.uint64:
            to_inc = -1  # static_cast<int64_t>(numeric_limits<uint64_t>::max())
        else:
            to_inc = _int_lowest_highest(t.dtype)[1]
        _check_from_to_in_range(lo, to_inc, t.dtype)
        if t.numel == 0:
            return
        var range_ = (
            UInt64(Int64(to_inc).cast[DType.uint64]())
            - UInt64(Int64(lo).cast[DType.uint64]())
            + 1
        )
        _random_from_to(t, range_, lo, generator)
    else:
        if not (
            t.dtype == DType.int64
            or t.dtype == DType.float64
            or t.dtype == DType.float32
            or t.dtype == DType.bfloat16
        ):
            raise Error(
                "random_full_64_bits_range_kernel_cuda handles only int64,"
                " double, float and bfloat16"
            )
        _draw(t, "RandomFull64", 0.0, 0.0, 0, 0, generator)


# aten::random_.from(Tensor(a!) self, int from, int? to, *, Generator? generator=None) -> Tensor(a!)
def op_random_from(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    _check_device_dtype(t, "random_", False, wide_unsigned=True)
    _random_from_to_impl(
        t,
        v_int(args[unsafe_offset=1]),
        args[unsafe_offset=2].copy(),
        v_generator(args[unsafe_offset=3]),
    )
    ret_ref(rets, 0, t)


# aten::random_.to(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)
def op_random_to(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    _check_device_dtype(t, "random_", False, wide_unsigned=True)
    _random_from_to_impl(
        t, 0, args[unsafe_offset=1].copy(), v_generator(args[unsafe_offset=2])
    )
    ret_ref(rets, 0, t)


# aten::random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)
def op_random_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    _check_device_dtype(t, "random_", False)
    var op: StaticString = "Random64" if (
        t.dtype == DType.int64 or t.dtype == DType.float64
    ) else "Random32"
    _draw(t, op, 0.0, 0.0, 0, 0, v_generator(args[unsafe_offset=1]))
    ret_ref(rets, 0, t)


# ---------------------------------------------------------------------------
# native_dropout (cuda/Dropout.cu: fused_dropout_kernel_vec / fused_dropout_kernel)
# ---------------------------------------------------------------------------


def _is_dense(t: T) -> Bool:
    """is_non_overlapping_and_dense."""
    var r = t.rank
    var perm = IndexList[MAX_RANK](0)
    for i in range(r):
        perm[i] = i
    # dims of size >= 2 sorted by stride; size < 2 dims last
    for i in range(1, r):
        var j = i
        while j > 0:
            var a = perm[j - 1]
            var b = perm[j]
            var swap = False
            if t.dim(a) < 2:
                swap = t.dim(b) >= 2
            elif t.dim(b) >= 2:
                swap = t.stride(a) > t.stride(b)
            if not swap:
                break
            perm[j - 1] = b
            perm[j] = a
            j -= 1
    var require = 1
    for i in range(r):
        var d = perm[i]
        if t.dim(d) < 2:
            return True
        if t.stride(d) != require:
            return False
        require *= t.dim(d)
    return True


def _maybe_overlapping(t: T) -> Bool:
    """cuda::detail::maybeOverlappingIndices: size>1 dims sorted by stride
    must each end before the next one starts."""
    var n = 0
    var sizes = IndexList[MAX_RANK](0)
    var strides = IndexList[MAX_RANK](0)
    for i in range(t.rank):
        if t.dim(i) > 1:
            if t.stride(i) < 1:
                return True
            sizes[n] = t.dim(i)
            strides[n] = t.stride(i)
            n += 1
    for i in range(1, n):
        var j = i
        while j > 0 and strides[j - 1] > strides[j]:
            var a = strides[j - 1]
            strides[j - 1] = strides[j]
            strides[j] = a
            var b = sizes[j - 1]
            sizes[j - 1] = sizes[j]
            sizes[j] = b
            j -= 1
    for i in range(n - 1):
        if (sizes[i] - 1) * strides[i] >= strides[i + 1]:
            return True
    return False


def _dropout_dtype_ok(dtype: DType) -> Bool:
    return (
        dtype == DType.float32
        or dtype == DType.float16
        or dtype == DType.bfloat16
        or dtype == DType.float64
    )


def _bool_mask(a: T, value: Bool) raises -> Owned:
    var m = own(new_tensor(a.shape, a.rank, ST_BOOL, a.device))
    fill_value(m.t, 1.0 if value else 0.0)
    return m^


def _logical_lists(t: T) -> Tuple[List[Int], List[Int]]:
    """(sizes, strides) fastest (last) dim first; rank 0 as one element."""
    var sizes = List[Int]()
    var strides = List[Int]()
    if t.rank == 0:
        sizes.append(1)
        strides.append(0)
    var i = t.rank - 1
    while i >= 0:
        sizes.append(t.dim(i))
        strides.append(t.stride(i))
        i -= 1
    return (sizes^, strides^)


# aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)
def op_native_dropout(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    if not _dropout_dtype_ok(a.dtype):
        unsupported(
            "native_dropout is only implemented for floating tensors on the"
            " mojo GPU device"
        )
    if a.dtype == DType.float64 and dev(a.device)[].api == "metal":
        unsupported("native_dropout of dtype float64 on Apple GPU")
    # `train=None` behaves like `train=True`; only an explicit False takes
    # the inference shortcut.
    var train = v_bool_or(args[unsafe_offset=2], True)
    if not train:
        var output = own(new_like(a))
        if a.contig:
            copy_d2d(
                ctx_for(a.device), output.t.ptr, a.ptr, a.numel * a.itemsize
            )
        else:
            copy_strided_into(output.t, a)
        var mask = _bool_mask(a, True)
        ret_owned(rets, 0, output)
        ret_owned(rets, 1, mask)
        return
    var p = v_f64(args[unsafe_offset=1])
    if not (p >= 0.0 and p <= 1.0):
        raise Error(
            "dropout probability has to be between 0 and 1, but got ", p
        )
    if a.numel == 0:
        var output = own(new_like(a))
        var mask = own(new_tensor(a.shape, a.rank, ST_BOOL, a.device))
        ret_owned(rets, 0, output)
        ret_owned(rets, 1, mask)
        return
    if p == 1.0:
        var output = own(new_like(a))
        fill_value(output.t, 0.0)
        var mask = _bool_mask(a, False)
        ret_owned(rets, 0, output)
        ret_owned(rets, 1, mask)
        return
    # empty_like preserves a dense layout; anything else comes out contiguous.
    var dense = _is_dense(a)
    var output = own(
        new_strided(a.shape, a.strides, a.rank, a.stype, a.device)
    ) if dense else own(new_like(a))
    var mask = own(
        new_strided(a.shape, a.strides, a.rank, ST_BOOL, a.device)
    ) if dense else own(new_tensor(a.shape, a.rank, ST_BOOL, a.device))
    # get_vector_size: alignment of the input pointer, capped at 16 bytes,
    # halved until it divides numel.
    var vec = 1
    if dense:
        var isz = a.itemsize
        var by_align = 8 if a.ptr % (8 * isz) == 0 else (
            4 if a.ptr % (4 * isz)
            == 0 else (2 if a.ptr % (2 * isz) == 0 else 1)
        )
        vec = min(16 // isz, by_align)
        while vec > 1 and a.numel % vec != 0:
            vec //= 2
    var grid = _grid(a.device, a.numel)
    var seed_offset = philox_reserve(
        0, a.device, _counter_offset(a.numel, grid, 4)
    )
    var in_ls = _logical_lists(a)
    var out_ls = _logical_lists(output.t)
    var ctx = ctx_for(a.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("dropout", "NativeDropout")
    call.arg_dtype(0, a.dtype)
    call.int(output.t.ptr)
    call.int(mask.t.ptr)
    call.int(a.ptr)
    call.int(a.numel)
    call.int(len(in_ls[0]))
    call.tuple(in_ls[0])
    call.tuple(in_ls[1])
    call.tuple(out_ls[1])
    call.int(vec)
    call.int(grid)
    call.f64(1.0 - p)
    call.int(Int(seed_offset[0] & 0xFFFFFFFF))
    call.int(Int((seed_offset[0] >> 32) & 0xFFFFFFFF))
    call.int(Int(seed_offset[1] & 0xFFFFFFFF))
    call.int(Int((seed_offset[1] >> 32) & 0xFFFFFFFF))
    call.int(cp)
    call.run()
    _ = ctx
    ret_owned(rets, 0, output)
    ret_owned(rets, 1, mask)


# aten::native_dropout_backward(Tensor grad_output, Tensor mask, float scale) -> Tensor
def op_native_dropout_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var keep = v_tensor(args[unsafe_offset=1])
    var scale = v_f64(args[unsafe_offset=2])
    if (
        not _dropout_dtype_ok(grad.dtype)
        or keep.dtype != DType.bool
        or keep.device != grad.device
        or not grad.same_shape(keep)
    ):
        unsupported(
            "native_dropout_backward is only implemented for a floating"
            " grad_output and a bool mask on the same mojo GPU device"
        )
    var grad_input = own(new_like(grad))
    if grad.numel > 0:
        var gc = own_if_new(contiguous(grad), grad)
        var kc = own_if_new(contiguous(keep), keep)
        var ctx = ctx_for(grad.device)
        var cp = ctx_ptr(ctx)
        var call = KernelCall("dropout", "NativeDropoutBackward")
        call.arg_dtype(0, grad.dtype)
        call.int(grad_input.t.ptr)
        call.int(gc.t.ptr)
        call.int(kc.t.ptr)
        call.int(grad.numel)
        call.f64(scale)
        call.int(cp)
        call.run()
        _ = ctx
        _ = gc^
        _ = kc^
    ret_owned(rets, 0, grad_input)


# ---------------------------------------------------------------------------
# aten::multinomial -- ATen's own algorithm (Distributions.cpp,
# multinomial_out), on the device Philox stream:
#
# * without replacement, or one sample: `argmax(p / q)` / `topk(p / q, n)`
#   with q ~ Exp(1) drawn by the registered exponential_ into a tensor of the
#   input's shape and dtype. Every piece is the native op of the same name,
#   so a seeded draw is the one stock CUDA makes from the same seed (its
#   exponential_ is bit-exact, `div` is IEEE and argmax takes the first
#   maximum). A zero-probability category scores exactly 0 and a positive one
#   scores > 0, so it is drawn only once every positive one is taken -- which
#   only `replacement=False` with more samples than positive entries asks.
# * with replacement and n_sample > 1: the inverse-CDF sampler of
#   tmb/kernels/random/multinomial_kernels.mojo, one Philox word per sample.
#
# Both validate the distribution first with one read of it and a 4-byte
# readback, raising ATen's messages synchronously (CPU's behaviour; CUDA's
# `_assert_async` aborts the context instead).
# ---------------------------------------------------------------------------

# Mirrored from MN_BAD_* in tmb/kernels/random/multinomial_kernels.mojo.
comptime MN_BAD_NEGATIVE = 1
comptime MN_BAD_NONFINITE = 2
comptime MN_BAD_SUM = 4
comptime FLOAT32_MAX_CONSECUTIVE_INT = 16777216


def _multinomial_check(p: T, rows: Int, n: Int, fast_path: Bool) raises:
    var ctx = ctx_for(p.device)
    var flag = own(new_tensor(IndexList[MAX_RANK](1), 1, ST_INT32, p.device))
    fill_value(flag.t, 0.0)
    var call = KernelCall("random", "MultinomialCheck")
    call.arg_dtype(0, p.dtype)
    call.int(flag.t.ptr)
    call.int(p.ptr)
    call.int(rows)
    call.int(n)
    call.int(dtype_code(p.dtype))
    call.int(ctx_ptr(ctx))
    call.run()
    var host = own(cpu_empty(IndexList[MAX_RANK](1), 1, ST_INT32))
    copy_to_host(ctx, flag.t.ptr, host.t.ptr, 4)
    var code = Int(
        Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=host.t.ptr)[]
    )
    _ = host^
    _ = flag^
    _ = ctx
    if code == 0:
        return
    if fast_path:
        if code & (MN_BAD_NEGATIVE | MN_BAD_NONFINITE):
            raise Error(
                "probability tensor contains either `inf`, `nan` or element < 0"
            )
        raise Error(
            "invalid multinomial distribution (sum of probabilities <= 0)"
        )
    if code & MN_BAD_NEGATIVE:
        raise Error(
            "invalid multinomial distribution (encountering probability entry"
            " < 0)"
        )
    if code & MN_BAD_NONFINITE:
        raise Error(
            "invalid multinomial distribution (encountering probability entry"
            " = infinity or NaN)"
        )
    raise Error("invalid multinomial distribution (sum of probabilities <= 0)")


def _multinomial(
    p_in: T, n_sample: Int, replacement: Bool, generator: Int
) raises -> T:
    """The int64 result, an owned handle."""
    if p_in.rank < 1 or p_in.rank > 2:
        raise Error("prob_dist must be 1 or 2 dim")
    if not _is_floating(p_in.dtype):
        raise Error(
            "multinomial only supports floating-point dtypes for input, got: ",
            _scalar_type_name(p_in.dtype),
        )
    if n_sample <= 0:
        raise Error("cannot sample n_sample <= 0 samples")
    var n = p_in.dim(p_in.rank - 1)
    if not replacement and n_sample > n:
        raise Error(
            "cannot sample n_sample > prob_dist.size(-1) samples without"
            " replacement"
        )
    if n > FLOAT32_MAX_CONSECUTIVE_INT:
        raise Error("number of categories cannot exceed 2^24")
    _check_device_dtype(p_in, "multinomial", True)
    var rows = p_in.dim(0) if p_in.rank == 2 else 1
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 1] = n_sample
    if p_in.rank == 2:
        shape[MAX_RANK - 2] = rows
    if rows == 0:
        return new_tensor(shape, p_in.rank, ST_INT64, p_in.device)
    if n == 0:
        # ATen's aminmax / CPU sampler reject an empty row before drawing.
        raise Error(
            "invalid multinomial distribution (sum of probabilities <= 0)"
        )
    var p = own_if_new(contiguous(p_in), p_in)
    var fast_path = not replacement or n_sample == 1
    _multinomial_check(p.t, rows, n, fast_path)
    if fast_path:
        var q = own(new_like(p.t))
        _draw(q.t, "Exponential", 1.0, 0.0, 0, 0, generator)
        var score_r = call_op(
            "aten::div", "Tensor", [_tensor_value(p.t), _tensor_value(q.t)], 1
        )
        var score = own(score_r.take_tensor(0))
        _ = q^
        # The argmax kernel has no float64 specialization; topk(1) is the
        # same answer (first maximum) in the same (rows, 1) shape.
        if n_sample == 1 and p.t.dtype != DType.float64:
            var r = call_op(
                "aten::argmax",
                "",
                [_tensor_value(score.t), int_arg(-1), bool_arg(True)],
                1,
            )
            _ = score^
            _ = p^
            return r.take_tensor(0)
        var r = call_op(
            "aten::topk",
            "",
            [
                _tensor_value(score.t),
                int_arg(n_sample),
                int_arg(-1),
                bool_arg(True),
                bool_arg(True),
            ],
            2,
        )
        _ = score^
        _ = p^
        return r.take_tensor(1)
    var out = own(new_tensor(shape, p_in.rank, ST_INT64, p_in.device))
    var ws = IndexList[MAX_RANK](1)
    ws[MAX_RANK - 1] = rows * n
    var cdf = own(
        new_tensor(
            ws,
            1,
            ST_FLOAT64 if p.t.dtype == DType.float64 else ST_FLOAT32,
            p_in.device,
        )
    )
    var words = rows * n_sample
    var seed_offset = philox_reserve(
        generator, p_in.device, ((words + 3) // 4) * 4
    )
    var ctx = ctx_for(p_in.device)
    var call = KernelCall("random", "MultinomialDraw")
    call.arg_dtype(0, p.t.dtype)
    call.int(out.t.ptr)
    call.int(cdf.t.ptr)
    call.int(p.t.ptr)
    call.int(rows)
    call.int(n)
    call.int(n_sample)
    call.int(Int(seed_offset[0] & 0xFFFFFFFF))
    call.int(Int((seed_offset[0] >> 32) & 0xFFFFFFFF))
    call.int(Int(seed_offset[1] & 0xFFFFFFFF))
    call.int(Int((seed_offset[1] >> 32) & 0xFFFFFFFF))
    call.int(dtype_code(p.t.dtype))
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx
    _ = cdf^  # alive past the launch
    _ = p^
    return out.take()


# aten::multinomial(Tensor self, SymInt num_samples, bool replacement=False, *,
#   Generator? generator=None) -> Tensor
def op_multinomial(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var result = own(
        _multinomial(
            v_tensor(args[unsafe_offset=0]),
            v_int(args[unsafe_offset=1]),
            v_bool_or(args[unsafe_offset=2], False),
            v_generator(args[unsafe_offset=3]),
        )
    )
    ret_owned(rets, 0, result)


# aten::multinomial.out(Tensor self, SymInt num_samples, bool replacement=False,
#   *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
def op_multinomial_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var p = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=4])
    if out.device != p.device:
        raise Error("multinomial arguments must have the same device")
    if out.stype != ST_INT64:
        raise Error(
            "multinomial expects Long tensor out, got: ",
            _scalar_type_name(out.dtype),
        )
    var result = own(
        _multinomial(
            p,
            v_int(args[unsafe_offset=1]),
            v_bool_or(args[unsafe_offset=2], False),
            v_generator(args[unsafe_offset=3]),
        )
    )
    if not (out.rank == result.t.rank and out.shape == result.t.shape):
        resize_out(out, result.t.shape, result.t.rank)
    copy_strided_into(out, result.t)
    _ = result^  # alive past the copy
    ret_ref(rets, 0, out)


def register_random(site: Site) raises:
    impl[op_uniform_, "uniform_"](site)
    impl[op_normal_, "normal_"](site)
    impl[op_normal_tensor_float, "normal.Tensor_float"](site)
    impl[op_normal_float_tensor, "normal.float_Tensor"](site)
    impl[op_normal_tensor_tensor, "normal.Tensor_Tensor"](site)
    impl[op_normal_tensor_float_out, "normal.Tensor_float_out"](site)
    impl[op_normal_float_tensor_out, "normal.float_Tensor_out"](site)
    impl[op_normal_tensor_tensor_out, "normal.Tensor_Tensor_out"](site)
    impl[op_log_normal_, "log_normal_"](site)
    impl[op_cauchy_, "cauchy_"](site)
    impl[op_exponential_, "exponential_"](site)
    impl[op_geometric_, "geometric_"](site)
    impl[op_bernoulli_float, "bernoulli_.float"](site)
    impl[op_bernoulli_tensor, "bernoulli_.Tensor"](site)
    impl[op_random_from, "random_.from"](site)
    impl[op_random_to, "random_.to"](site)
    impl[op_random_, "random_"](site)
    impl[op_native_dropout, "native_dropout"](site)
    impl[op_native_dropout_backward, "native_dropout_backward"](site)
    impl[op_multinomial, "multinomial"](site)
    impl[op_multinomial_out, "multinomial.out"](site)
