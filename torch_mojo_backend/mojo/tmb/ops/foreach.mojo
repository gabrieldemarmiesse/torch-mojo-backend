"""ATen ops: foreach group (see agents_docs/native_backend.md).

Every `_foreach_*_` op below tries ONE batched kernel
launch (the `optimizer` family, ported host-side from
`eager_kernels/aten_fast.py`) when its homogeneous-dtype / contiguous /
no-aliasing preconditions hold, and otherwise falls back to ATen's own
sequential semantics: call the per-tensor op once per list element through
`tmb_call_op`, exactly what ATen's CompositeExplicitAutograd "slow" kernels
do (`aten/src/ATen/native/ForeachOpsKernels.cpp`). That fallback is what
makes "declining" safe here, unlike the general porting rule of raising
`unsupported(...)` on every old `NOT_HANDLED`: the old Python eager path had
no dispatcher to fall through to, but this native op runs as a real
PrivateUse1 kernel, so calling the *same* op name from inside itself would
re-enter our own registration (infinite recursion) -- calling the
*per-tensor* op name instead (`add_.Scalar`, `addcmul_`, ...) does not, and
matches the exact semantics ATen's own slow kernels implement.

GradScaler's two ops, `_amp_foreach_non_finite_check_and_unscale_` (a member
of the same batched family) and `_amp_update_scale_` (one thread), have no
CompositeExplicitAutograd registration either, so they take no sequential
fallback: what the batched kernel cannot take goes through a contiguous
temporary, and what no kernel supports is declined.

The fused optimizers (`_fused_{adam,adamw,sgd,adagrad}_`, what
torch.optim runs with `fused=True`) are the same: no CompositeExplicitAutograd
registration, so no sequential fallback. They validate like stock CUDA and
raise its errors; see their section below.

`_foreach_div_.ScalarList` and `_foreach_addcdiv_.ScalarList` are NOT
registered here: there is no batched kernel for them, and with no
PrivateUse1 override ATen's own CompositeExplicitAutograd decomposition
(`foreach_tensor_div_scalarlist_kernel_slow_` /
`..._addcdiv_scalarlist_kernel_slow_`) already does exactly what the
sequential fallbacks below do -- one `div_.Scalar` / `addcdiv_` per tensor.
(Their `Scalar[]` argument does marshal: `to_record`'s `ListType` branch in
`native/csrc/shim_dispatch.cpp` has a `NumberType` arm.)
"""
from std.builtin.sort import sort
from std.math import ceildiv
from std.utils import IndexList

from tmb.backend.abi import (
    ST_FLOAT32,
    T,
    is_dense,
    Value,
    Values,
    call_op,
    dtype_code,
    f64_bits,
    new_like,
    new_tensor,
    own,
    Owned,
    ret_ref,
    ret_tensor_list,
    unsupported,
    v_bool,
    v_dtype_or,
    v_f64,
    v_int,
    v_scalar_is_bool,
    v_tensor,
    v_tensor_list,
    TAG_NONE,
    TAG_SCALAR_INT,
    TAG_TENSOR,
    TAG_TENSOR_REF,
)
from tmb.backend.device import ctx_for, ctx_ptr, dev, copy_d2d
from tmb.kernels.optimizer.foreach_clip_contract import FOREACH_CHUNK_ELEMENTS
from tmb.backend.kernel_call import KernelCall
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.backend.registry import Site, impl
from tmb.ops.common import copy_strided_into
from tmb.ops.data_movement import _scalar_type_name


# --- the sequential per-tensor fallback ---------------------------------
#
# A foreach op that declines its batched fast path replicates ATen's own
# sequential decomposition by calling the underlying per-tensor op through
# the full dispatcher (never the SAME op name, which would re-enter this very
# registration). Every such call allocates a fresh result handle -- the shim
# wraps even the tensor an in-place op hands back -- so the results go through
# `Results`, which releases whatever is not taken.


def _tensor_value(t: T) -> Value:
    return Value(TAG_TENSOR, 0, Int64(t.h), 0)


def _none_value() -> Value:
    return Value(TAG_NONE, 0, 0, 0)


def _seq_add_scalar_(t: T, scalar_v: Value) raises:
    var args = List[Value](capacity=3)
    args.append(_tensor_value(t))
    args.append(scalar_v.copy())
    args.append(Value(TAG_SCALAR_INT, 0, 1, 0))  # alpha=1
    _ = call_op("aten::add_", "Scalar", args^, 1)


def _seq_mul_scalar_(t: T, scalar_v: Value) raises:
    var args = List[Value](capacity=2)
    args.append(_tensor_value(t))
    args.append(scalar_v.copy())
    _ = call_op("aten::mul_", "Scalar", args^, 1)


def _seq_mul_tensor_(t: T, other: T) raises:
    var args = List[Value](capacity=2)
    args.append(_tensor_value(t))
    args.append(_tensor_value(other))
    _ = call_op("aten::mul_", "Tensor", args^, 1)


def _seq_addcmul_(t: T, t1: T, t2: T, value_v: Value) raises:
    var args = List[Value](capacity=4)
    args.append(_tensor_value(t))
    args.append(_tensor_value(t1))
    args.append(_tensor_value(t2))
    args.append(value_v.copy())
    _ = call_op("aten::addcmul_", "", args^, 1)


def _seq_lerp_scalar_(t: T, end: T, weight_v: Value) raises:
    var args = List[Value](capacity=3)
    args.append(_tensor_value(t))
    args.append(_tensor_value(end))
    args.append(weight_v.copy())
    _ = call_op("aten::lerp_", "Scalar", args^, 1)


def _seq_sqrt(t: T) raises -> T:
    var args = List[Value](capacity=1)
    args.append(_tensor_value(t))
    var rets = call_op("aten::sqrt", "", args^, 1)
    return rets.take_tensor(0)


def _seq_vector_norm(t: T, ord_v: Value, dtype_v: Value) raises -> T:
    var args = List[Value](capacity=5)
    args.append(_tensor_value(t))
    args.append(ord_v.copy())
    args.append(_none_value())  # dim=None
    args.append(Value(TAG_SCALAR_INT, 0, 0, 0))  # keepdim=False
    args.append(dtype_v.copy())
    var rets = call_op("aten::linalg_vector_norm", "", args^, 1)
    return rets.take_tensor(0)


# --- overlap / aliasing (aten_fast._foreach_tensors_overlap /
# _foreach_mutation_hazard, simplified to pairwise interval checks: foreach
# lists are not so long that an O(n^2) sweep matters next to the kernel
# launches around it) ----------------------------------------------------


def _overlaps(a: T, b: T) -> Bool:
    if a.numel == 0 or b.numel == 0:
        return False
    var a_begin = a.ptr
    var a_end = a_begin + a.numel * a.itemsize
    var b_begin = b.ptr
    var b_end = b_begin + b.numel * b.itemsize
    return a_begin < b_end and b_begin < a_end


def _self_overlaps(ts: List[T]) -> Bool:
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            if _overlaps(ts[i], ts[j]):
                return True
    return False


def _overlaps_any(ts: List[T], other: List[T]) -> Bool:
    for a in ts:
        for b in other:
            if _overlaps(a, b):
                return True
    return False


def _scalar_overlaps_any(ts: List[T], scalar: T) -> Bool:
    """Whether the 0-d `scalar`'s bytes intersect ANY list tensor's bytes. If
    so, take the sequential fallback: its per-tensor `mul_.Tensor(a!)` call
    both tolerates true self-aliasing (`t.mul_(t)`) and raises ATen's own
    "Please clone()" error for a genuine partial overlap -- the batched
    kernel, with no ordering between its parallel slots, could do neither."""
    for t in ts:
        if _overlaps(t, scalar):
            return True
    return False


# --- fast-path qualification (aten_fast._foreach_lists): every tensor across
# every list mojo-resident, contiguous, on one shared device, one shared
# dtype, with index-aligned tensors sharing a shape ------------


def _tensor_qualifies(t: T, device: Int, dtype: DType) -> Bool:
    return t.on_mojo() and t.device == device and t.dtype == dtype and t.contig


def _qualifies1(a: List[T], allow_half: Bool) raises -> Bool:
    if len(a) == 0:
        return False
    var first = a[0].copy()
    if not first.on_mojo():
        return False
    # Metal's batched kernels accept only float32. Half lists must use the
    # existing per-tensor scalar operations instead of entering that kernel.
    if dev(first.device)[].api == "metal" and first.dtype != DType.float32:
        return False
    if allow_half:
        if (
            first.dtype != DType.float32
            and first.dtype != DType.float16
            and first.dtype != DType.bfloat16
        ):
            return False
    elif first.dtype != DType.float32:
        return False
    for t in a:
        if not _tensor_qualifies(t, first.device, first.dtype):
            return False
    return True


def _qualifies2(a: List[T], b: List[T], allow_half: Bool) raises -> Bool:
    if not _qualifies1(a, allow_half):
        return False
    if len(b) != len(a):
        return False
    for i in range(len(a)):
        if not _tensor_qualifies(b[i], a[0].device, a[0].dtype):
            return False
        if not b[i].same_shape(a[i]):
            return False
    return True


def _qualifies3(
    a: List[T], b: List[T], c: List[T], allow_half: Bool
) raises -> Bool:
    if not _qualifies2(a, b, allow_half):
        return False
    if len(c) != len(a):
        return False
    for i in range(len(a)):
        if not _tensor_qualifies(c[i], a[0].device, a[0].dtype):
            return False
        if not c[i].same_shape(a[i]):
            return False
    return True


# --- batched launches (optimizer family: ForeachMul/Add/MulTensor/Lerp/
# Addcmul/Sqrt/L2Norm, FusedAdamW -- see tmb/kernels/optimizer/entry.mojo's `tmb_call`
# ladder for the exact slot list each one expects) ------------------------


def _foreach_ew_scalar_launch(
    op_name: StaticString, tensors: List[T], value: Float64
) raises:
    """ForeachMul / ForeachAdd: one repeated per-tensor FP32 scalar (`.Scalar`
    and `.ScalarList` share this kernel code; every tensor here gets the same
    value since only the `.Scalar` overloads reach this launcher)."""
    var metadata = List[Int]()
    for t in tensors:
        metadata.append(t.ptr)
        metadata.append(t.numel)
    var scalars = List[Int]()
    for _ in range(len(tensors)):
        scalars.append(Int(f64_bits(value)))
    var dtype = tensors[0].dtype
    var device = tensors[0].device
    var ctx = ctx_for(device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("optimizer", String(op_name))
    call.arg_dtype(0, dtype)
    call.out_dtype(dtype)
    call.tuple(metadata)
    call.tuple(scalars)
    call.tuple(List[Int]())
    call.int(dtype_code(dtype))
    call.int(cp)
    call.run()
    for t in tensors:
        t.bump_version()
    _ = ctx


def _foreach_mul_tensor_launch(tensors: List[T], scalar: T) raises:
    var metadata = List[Int]()
    for t in tensors:
        metadata.append(t.ptr)
        metadata.append(t.numel)
    var dtype = tensors[0].dtype
    var ctx = ctx_for(tensors[0].device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("optimizer", "ForeachMulTensor")
    call.arg_dtype(0, dtype)
    call.out_dtype(dtype)
    call.tuple(metadata)
    call.tuple(List[Int]())
    var aux = List[Int]()
    aux.append(scalar.ptr)
    call.tuple(aux)
    call.int(dtype_code(dtype))
    call.int(cp)
    call.run()
    for t in tensors:
        t.bump_version()
    _ = ctx


def _foreach_lerp_launch(
    self_list: List[T],
    end_list: List[T],
    weight: Float64,
    bump_versions: Bool = True,
) raises:
    var metadata = List[Int]()
    for i in range(len(self_list)):
        metadata.append(self_list[i].ptr)
        metadata.append(end_list[i].ptr)
        metadata.append(self_list[i].numel)
    var narrowed_weight = Float32(weight)
    var one_minus_weight = Float32(1.0) - narrowed_weight
    var low_branch = 1 if abs(narrowed_weight) < Float32(0.5) else 0
    var scalars = List[Int]()
    scalars.append(Int(f64_bits(Float64(narrowed_weight))))
    scalars.append(Int(f64_bits(Float64(one_minus_weight))))
    var aux = List[Int]()
    aux.append(low_branch)
    var dtype = self_list[0].dtype
    var ctx = ctx_for(self_list[0].device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("optimizer", "ForeachLerp")
    call.arg_dtype(0, dtype)
    call.arg_dtype(1, dtype)
    call.out_dtype(dtype)
    call.tuple(metadata)
    call.tuple(scalars)
    call.tuple(aux)
    call.int(dtype_code(dtype))
    call.int(cp)
    call.run()
    if bump_versions:
        for t in self_list:
            t.bump_version()
    _ = ctx


def _foreach_addc_launch(
    self_list: List[T],
    t1_list: List[T],
    t2_list: List[T],
    value: Float64,
    op: String = "ForeachAddcmul",
    bump_versions: Bool = True,
) raises:
    var metadata = List[Int]()
    for i in range(len(self_list)):
        metadata.append(self_list[i].ptr)
        metadata.append(t1_list[i].ptr)
        metadata.append(t2_list[i].ptr)
        metadata.append(self_list[i].numel)
    var scalars = List[Int]()
    for _ in range(len(self_list)):
        scalars.append(Int(f64_bits(value)))
    var dtype = self_list[0].dtype
    var ctx = ctx_for(self_list[0].device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("optimizer", op)
    call.arg_dtype(0, dtype)
    call.arg_dtype(1, dtype)
    call.arg_dtype(2, dtype)
    call.out_dtype(dtype)
    call.tuple(metadata)
    call.tuple(scalars)
    call.tuple(List[Int]())
    call.int(dtype_code(dtype))
    call.int(cp)
    call.run()
    if bump_versions:
        for t in self_list:
            t.bump_version()
    _ = ctx


def _foreach_sqrt_launch(tensors: List[T]) raises -> List[T]:
    var outs = List[Owned]()
    for t in tensors:
        outs.append(own(new_tensor(t.shape, t.rank, t.stype, t.device)))
    var metadata = List[Int]()
    for i in range(len(tensors)):
        metadata.append(tensors[i].ptr)
        metadata.append(outs[i].t.ptr)
        metadata.append(tensors[i].numel)
    var dtype = tensors[0].dtype
    var ctx = ctx_for(tensors[0].device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("optimizer", "ForeachSqrt")
    call.arg_dtype(0, dtype)
    call.arg_dtype(1, dtype)
    call.out_dtype(dtype)
    call.tuple(metadata)
    call.tuple(List[Int]())
    call.tuple(List[Int]())
    call.int(dtype_code(dtype))
    call.int(cp)
    call.run()
    _ = ctx
    var result = List[T]()
    for i in range(len(outs)):
        result.append(outs[i].take())
    return result^


def _foreach_norm_launch(tensors: List[T]) raises -> List[T]:
    var outs = List[Owned]()
    var total_chunks = 0
    for t in tensors:
        outs.append(
            own(new_tensor(IndexList[MAX_RANK](1), 0, t.stype, t.device))
        )
        total_chunks += ceildiv(t.numel, FOREACH_CHUNK_ELEMENTS)
    var metadata = List[Int]()
    for i in range(len(tensors)):
        metadata.append(tensors[i].ptr)
        metadata.append(outs[i].t.ptr)
        metadata.append(tensors[i].numel)
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 1] = max(total_chunks, 1)
    var partials = own(new_tensor(shape, 1, ST_FLOAT32, tensors[0].device))
    var ctx = ctx_for(tensors[0].device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("optimizer", "ForeachL2Norm")
    call.arg_dtype(0, DType.float32)
    call.arg_dtype(1, DType.float32)
    call.arg_dtype(2, DType.float32)
    call.out_dtype(DType.float32)
    call.tuple(metadata)
    call.int(partials.t.ptr)
    call.int(partials.t.numel)
    call.int(cp)
    call.run()
    _ = partials  # alive past the launch (its last use above is the pointer read)
    _ = ctx
    var result = List[T]()
    for i in range(len(outs)):
        result.append(outs[i].take())
    return result^


# --- ops -------------------------------------------------------------------


# aten::_foreach_add_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
def op_foreach_add_scalar_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var self_list = v_tensor_list(args[unsafe_offset=0])
    if len(self_list) == 0:
        raise Error("Tensor list must have at least one tensor.")
    var scalar_v = args[unsafe_offset=1].copy()
    if (
        not v_scalar_is_bool(scalar_v)
        and _qualifies1(self_list, True)
        and not _self_overlaps(self_list)
    ):
        _foreach_ew_scalar_launch("ForeachAdd", self_list, v_f64(scalar_v))
        return
    for t in self_list:
        _seq_add_scalar_(t, scalar_v)


# aten::_foreach_mul_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
def op_foreach_mul_scalar_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var self_list = v_tensor_list(args[unsafe_offset=0])
    if len(self_list) == 0:
        raise Error("Tensor list must have at least one tensor.")
    var scalar_v = args[unsafe_offset=1].copy()
    if (
        not v_scalar_is_bool(scalar_v)
        and _qualifies1(self_list, True)
        and not _self_overlaps(self_list)
    ):
        _foreach_ew_scalar_launch("ForeachMul", self_list, v_f64(scalar_v))
        return
    for t in self_list:
        _seq_mul_scalar_(t, scalar_v)


# aten::_foreach_mul_.Tensor(Tensor(a!)[] self, Tensor other) -> ()
def op_foreach_mul_tensor_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var self_list = v_tensor_list(args[unsafe_offset=0])
    var other = v_tensor(args[unsafe_offset=1])
    if other.rank != 0:
        raise Error(
            "scalar tensor expected to be 0 dim but it has ",
            other.rank,
            " dimensions and ",
            other.numel,
            " elements.",
        )
    if len(self_list) == 0:
        raise Error("Tensor list must have at least one tensor.")
    if (
        _qualifies1(self_list, False)
        and other.on_mojo()
        and other.device == self_list[0].device
        and other.dtype == self_list[0].dtype
        and other.contig
        and not _self_overlaps(self_list)
        and not _scalar_overlaps_any(self_list, other)
    ):
        _foreach_mul_tensor_launch(self_list, other)
        return
    for t in self_list:
        _seq_mul_tensor_(t, other)


# aten::_foreach_addcmul_.Scalar(Tensor(a!)[] self, Tensor[] tensor1,
#   Tensor[] tensor2, Scalar value=1) -> ()
def op_foreach_addcmul_scalar_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var self_list = v_tensor_list(args[unsafe_offset=0])
    var t1_list = v_tensor_list(args[unsafe_offset=1])
    var t2_list = v_tensor_list(args[unsafe_offset=2])
    if len(self_list) == 0:
        raise Error("Tensor list must have at least one tensor.")
    if len(t1_list) != len(self_list) or len(t2_list) != len(self_list):
        raise Error("Tensor lists must have the same number of tensors.")
    var value_v = args[unsafe_offset=3].copy()
    if (
        not v_scalar_is_bool(value_v)
        and _qualifies3(self_list, t1_list, t2_list, False)
        and not _self_overlaps(self_list)
        and not _overlaps_any(self_list, t1_list)
        and not _overlaps_any(self_list, t2_list)
    ):
        _foreach_addc_launch(self_list, t1_list, t2_list, v_f64(value_v))
        return
    for i in range(len(self_list)):
        _seq_addcmul_(self_list[i], t1_list[i], t2_list[i], value_v)


def _batch_copy_dtype(dt: DType) -> Bool:
    """Mirrors tmb/kernels/data_movement/entry.mojo's `COPY_BATCH_DTYPES`."""
    return (
        dt == DType.float64
        or dt == DType.float32
        or dt == DType.float16
        or dt == DType.bfloat16
        or dt == DType.int64
        or dt == DType.int32
        or dt == DType.int16
        or dt == DType.int8
        or dt == DType.uint64
        or dt == DType.uint32
        or dt == DType.uint16
        or dt == DType.uint8
        or dt == DType.bool
    )


def _bits_dtype(itemsize: Int) -> DType:
    """The unsigned dtype of this width: a same-dtype copy only moves bits, so
    every dtype of one width shares one compiled variant."""
    if itemsize == 1:
        return DType.uint8
    if itemsize == 2:
        return DType.uint16
    if itemsize == 4:
        return DType.uint32
    return DType.uint64


def _writes_overlap(dsts: List[T], srcs: List[T]) -> Bool:
    """Whether a destination's bytes meet another destination's or any
    source's, the pairs a one-launch copy cannot order like sequential copy_.

    O(n log n) on plain integer sorts. Half-open ranges are pairwise disjoint
    exactly when, with starts and ends sorted independently, every start is
    at or past the previous end; the sorted destinations then pair up in
    order, and a binary search finds the one each source could meet.
    """
    var begins = List[Int](capacity=len(dsts))
    var ends = List[Int](capacity=len(dsts))
    for t in dsts:
        if t.numel > 0:
            begins.append(t.ptr)
            ends.append(t.ptr + t.numel * t.itemsize)
    sort(begins)
    sort(ends)
    for k in range(1, len(begins)):
        if begins[k] < ends[k - 1]:
            return True
    for t in srcs:
        if t.numel == 0:
            continue
        var end = t.ptr + t.numel * t.itemsize
        # The last destination starting before this source ends.
        var lo = 0
        var hi = len(begins)
        while lo < hi:
            var mid = (lo + hi) // 2
            if begins[mid] < end:
                lo = mid + 1
            else:
                hi = mid
        if lo > 0 and ends[lo - 1] > t.ptr:
            return True
    return False


def _batched_copy_device(device: Int) raises -> Bool:
    """Where the batched rectangle copy runs (CopyBatched).

    TODO: enable on Metal. Tried on an M4 (macOS 26.6): 142 of the 173 copy
    tests in tests/native/test_foreach.py fail. float64 does not build
    (Apple GPUs have none; it must stay excluded), some variants hit "Failed
    to verify LLVM IR for Metal", and the rest return wrong elements. The
    descriptors are an array passed by value, and indexing a copied array of
    pointers is what miscompiles on Metal (see `_cat_pick` in
    tmb/kernels/data_movement/entry.mojo).
    """
    var api = dev(device)[].api
    return api == "cuda" or api == "hip"


def _copy_batch_qualifies(dsts: List[T], srcs: List[T]) raises -> Bool:
    """CUDA's `_foreach_copy_` fast-path rule (one device, contiguous, one
    dtype per list, index-aligned shapes) plus the cross-pair overlap check
    it lacks, so aliased lists keep sequential copy_ semantics."""
    var first = dsts[0].copy()
    if not first.on_mojo() or not _batched_copy_device(first.device):
        return False
    var src_dtype = srcs[0].dtype
    if not _batch_copy_dtype(first.dtype) or not _batch_copy_dtype(src_dtype):
        return False
    for i in range(len(dsts)):
        if not _tensor_qualifies(dsts[i], first.device, first.dtype):
            return False
        if not _tensor_qualifies(srcs[i], first.device, src_dtype):
            return False
        if not dsts[i].same_shape(srcs[i]):
            return False
    # Empty occurrences have no byte-range hazard. The caller still bumps
    # each occurrence, including repeated references to the same empty tensor.
    return not _writes_overlap(dsts, srcs)


def _copy_batch_launch(dsts: List[T], srcs: List[T]) raises:
    var src_dtype = srcs[0].dtype
    var dst_dtype = dsts[0].dtype
    if src_dtype == dst_dtype:
        src_dtype = _bits_dtype(dsts[0].itemsize)
        dst_dtype = src_dtype
    var ctx = ctx_for(dsts[0].device)
    var call = KernelCall("data_movement", "CopyBatched")
    call.arg_dtype(0, src_dtype)
    call.out_dtype(dst_dtype)
    # One-row rectangles: source, destination, rows, cols and both pitches.
    var metadata = List[Int](capacity=6 * len(dsts))
    for i in range(len(dsts)):
        metadata.append(srcs[i].ptr)
        metadata.append(dsts[i].ptr)
        metadata.append(1)
        metadata.append(dsts[i].numel)
        metadata.append(dsts[i].numel)
        metadata.append(dsts[i].numel)
    call.tuple(metadata)
    call.int(ctx_ptr(ctx))
    call.run()
    # This schema has no ADInplaceOrView wrapper, unlike scalar copy_.
    for t in dsts:
        t.bump_version()
    _ = ctx


def _copy_adjacent_views(dsts: List[T], srcs: List[T]) raises -> Bool:
    """Copy matching contiguous partitions of two buffers with one DMA.

    Adjacency is proved from current pointers, shapes and strides. Require
    one storage base on each side so the DMA never crosses allocations.
    Overlapping source/destination spans retain sequential copy_ semantics.
    """
    var first = dsts[0].copy()
    if not first.on_mojo() or dev(first.device)[].api != "cuda":
        return False
    var dst_begin = 0
    var src_begin = 0
    var dst_storage = 0
    var src_storage = 0
    var nbytes = 0
    for i in range(len(dsts)):
        var d = dsts[i].copy()
        var s = srcs[i].copy()
        if (
            not d.on_mojo()
            or not s.on_mojo()
            or d.device != first.device
            or s.device != first.device
            or d.stype != first.stype
            or s.stype != first.stype
            or not d.contig
            or not s.contig
            or not d.same_shape(s)
        ):
            return False
        if d.numel == 0:
            continue
        if nbytes == 0:
            dst_begin, src_begin = d.ptr, s.ptr
            dst_storage, src_storage = d.storage_ptr(), s.storage_ptr()
            if dst_storage == 0 or src_storage == 0:
                return False
        if (
            d.ptr != dst_begin + nbytes
            or s.ptr != src_begin + nbytes
            or d.storage_ptr() != dst_storage
            or s.storage_ptr() != src_storage
        ):
            return False
        nbytes += d.numel * d.itemsize
    if nbytes != 0 and dst_begin != src_begin:
        if dst_begin < src_begin + nbytes and src_begin < dst_begin + nbytes:
            return False
        copy_d2d(ctx_for(first.device), dst_begin, src_begin, nbytes)
    # _foreach_copy_ owns its version bumps (no ADInplaceOrView wrapper).
    # Repeated empty destinations still receive one bump per occurrence.
    for d in dsts:
        d.bump_version()
    return True


# aten::_foreach_copy_(Tensor(a!)[] self, Tensor[] src, bool non_blocking=False) -> ()
def op_foreach_copy_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var dsts = v_tensor_list(args[unsafe_offset=0])
    var srcs = v_tensor_list(args[unsafe_offset=1])
    if len(dsts) == 0:
        raise Error("Tensor list must have at least one tensor.")
    if len(srcs) != len(dsts):
        raise Error("Tensor lists must have the same number of tensors.")
    if _copy_adjacent_views(dsts, srcs):
        return
    if _copy_batch_qualifies(dsts, srcs):
        _copy_batch_launch(dsts, srcs)
        return
    for i in range(len(dsts)):
        var copy_args = List[Value]()
        copy_args.append(_tensor_value(dsts[i]))
        copy_args.append(_tensor_value(srcs[i]))
        copy_args.append(args[unsafe_offset=2].copy())
        _ = call_op("aten::copy_", "", copy_args^, 1)


# aten::_foreach_lerp_.Scalar(Tensor(a!)[] self, Tensor[] tensors1, Scalar weight) -> ()
def op_foreach_lerp_scalar_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var self_list = v_tensor_list(args[unsafe_offset=0])
    var end_list = v_tensor_list(args[unsafe_offset=1])
    if len(self_list) == 0:
        raise Error("Tensor list must have at least one tensor.")
    if len(end_list) != len(self_list):
        raise Error("Tensor lists must have the same number of tensors.")
    var weight_v = args[unsafe_offset=2].copy()
    if (
        not v_scalar_is_bool(weight_v)
        and _qualifies2(self_list, end_list, False)
        and not _self_overlaps(self_list)
        and not _overlaps_any(self_list, end_list)
    ):
        _foreach_lerp_launch(self_list, end_list, v_f64(weight_v))
        return
    for i in range(len(self_list)):
        _seq_lerp_scalar_(self_list[i], end_list[i], weight_v)


# aten::_foreach_sqrt(Tensor[] self) -> Tensor[]
def op_foreach_sqrt(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var self_list = v_tensor_list(args[unsafe_offset=0])
    if len(self_list) == 0:
        raise Error("Tensor list must have at least one tensor.")
    if _qualifies1(self_list, False):
        ret_tensor_list(rets, 0, _foreach_sqrt_launch(self_list))
        return
    var result = List[T]()
    for t in self_list:
        result.append(_seq_sqrt(t))
    ret_tensor_list(rets, 0, result)


# aten::_foreach_norm.Scalar(Tensor[] self, Scalar ord=2, ScalarType? dtype=None) -> Tensor[]
def op_foreach_norm_scalar(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var self_list = v_tensor_list(args[unsafe_offset=0])
    if len(self_list) == 0:
        raise Error("Tensor list must have at least one tensor.")
    var ord_v = args[unsafe_offset=1].copy()
    var dtype_v = args[unsafe_offset=2].copy()
    var ord_is_two = not v_scalar_is_bool(ord_v) and v_f64(ord_v) == 2.0
    var dtype_ok = (
        dtype_v.tag == TAG_NONE or v_dtype_or(dtype_v, ST_FLOAT32) == ST_FLOAT32
    )
    if ord_is_two and dtype_ok and _qualifies1(self_list, False):
        ret_tensor_list(rets, 0, _foreach_norm_launch(self_list))
        return
    var result = List[T]()
    for t in self_list:
        result.append(_seq_vector_norm(t, ord_v, dtype_v))
    ret_tensor_list(rets, 0, result)


def _scalar_f32_tensor(
    v: Value, name: StaticString, device: Int, dtype: DType = DType.float32
) raises -> Int:
    """A validated one-element device scalar (`grad_scale`/`found_inf`/...):
    its device pointer, or 0 for None."""
    if v.tag == TAG_NONE:
        return 0
    var t = v_tensor(v)
    if (
        t.device != device
        or not t.on_mojo()
        or t.dtype != dtype
        or t.numel != 1
        or not t.contig
    ):
        raise Error(
            name,
            " must be a contiguous one-element ",
            dtype,
            " tensor on the same mojo device as the other operands",
        )
    return t.ptr


# --- GradScaler (torch/amp/grad_scaler.py) ----------------------------------


def _unscale_dtype_ok(t: T) raises -> Bool:
    """The batched kernel's dtypes; Metal's fixed-arity arm is float32 only."""
    if dev(t.device)[].api == "metal":
        return t.dtype == DType.float32
    return (
        t.dtype == DType.float32
        or t.dtype == DType.float16
        or t.dtype == DType.bfloat16
    )


def _unscale_launch(tensors: List[T], inv_scale: T, found_inf: T) raises:
    """One batched launch over same-dtype tensors whose elements are dense:
    the math is elementwise, so any stride permutation of a dense tensor is
    the same `numel` elements from its data pointer."""
    var metadata = List[Int]()
    for t in tensors:
        metadata.append(t.ptr)
        metadata.append(t.numel)
    var aux = List[Int]()
    aux.append(inv_scale.ptr)
    aux.append(found_inf.ptr)
    var dtype = tensors[0].dtype
    var ctx = ctx_for(tensors[0].device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("optimizer", "ForeachNonFiniteUnscale")
    call.arg_dtype(0, dtype)
    call.out_dtype(dtype)
    call.tuple(metadata)
    call.tuple(List[Int]())
    call.tuple(aux)
    call.int(dtype_code(dtype))
    call.int(cp)
    call.run()
    for t in tensors:
        t.bump_version()
    _ = ctx


# aten::_amp_foreach_non_finite_check_and_unscale_(Tensor(a!)[] self,
#   Tensor(b!) found_inf, Tensor inv_scale) -> ()
def op_amp_foreach_non_finite_check_and_unscale_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """ATen's CUDA semantics: found_inf = 1 if any element is inf/NaN, and
    every element is multiplied by inv_scale unless it is exactly 1. One
    launch when the list shares a dtype and every tensor is dense, as
    GradScaler's per-(device, dtype) lists do; anything else goes tensor by
    tensor, through a contiguous temporary when the tensor is not dense."""
    var self_list = v_tensor_list(args[unsafe_offset=0])
    var found_inf = v_tensor(args[unsafe_offset=1])
    if len(self_list) == 0:
        return
    if not found_inf.on_mojo():
        unsupported("found_inf is not on a mojo device")
    var device = found_inf.device
    _ = _scalar_f32_tensor(args[unsafe_offset=1], "found_inf", device)
    _ = _scalar_f32_tensor(args[unsafe_offset=2], "inv_scale", device)
    var inv_scale = v_tensor(args[unsafe_offset=2])
    var batched = True
    for t in self_list:
        if not t.on_mojo() or t.device != device:
            raise Error(
                "_amp_foreach_non_finite_check_and_unscale_: every tensor"
                " must be on found_inf's device"
            )
        if not t.dtype.is_floating_point():
            raise Error(
                "_amp_foreach_non_finite_check_and_unscale_ only supports"
                " floating-point tensors"
            )
        if not _unscale_dtype_ok(t):
            unsupported(
                "_amp_foreach_non_finite_check_and_unscale_: dtype "
                + String(t.dtype)
                + " is not supported on this GPU"
            )
        if t.dtype != self_list[0].dtype or not is_dense(
            t.shape, t.strides, t.rank
        ):
            batched = False
    if batched:
        _unscale_launch(self_list, inv_scale, found_inf)
        found_inf.bump_version()
        return
    for t in self_list:
        if t.numel == 0:
            continue
        var one = List[T]()
        if is_dense(t.shape, t.strides, t.rank):
            one.append(t.copy())
            _unscale_launch(one, inv_scale, found_inf)
            continue
        var tmp = own(new_like(t))
        copy_strided_into(tmp.t, t)
        one.append(tmp.t.copy())
        _unscale_launch(one, inv_scale, found_inf)
        copy_strided_into(t, tmp.t)
        t.bump_version()
    found_inf.bump_version()


# aten::_amp_update_scale_(Tensor(a!) self, Tensor(b!) growth_tracker,
#   Tensor found_inf, float scale_growth_factor, float scale_backoff_factor,
#   int growth_interval) -> Tensor(a!)
def op_amp_update_scale_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var scale = v_tensor(args[unsafe_offset=0])
    if not scale.on_mojo():
        unsupported("_amp_update_scale_: the scale is not on a mojo device")
    var device = scale.device
    _ = _scalar_f32_tensor(args[unsafe_offset=0], "current_scale", device)
    var tracker_ptr = _scalar_f32_tensor(
        args[unsafe_offset=1], "growth_tracker", device, DType.int32
    )
    var found_inf_ptr = _scalar_f32_tensor(
        args[unsafe_offset=2], "found_inf", device
    )
    var factors = List[Int]()
    factors.append(Int(f64_bits(v_f64(args[unsafe_offset=3]))))
    factors.append(Int(f64_bits(v_f64(args[unsafe_offset=4]))))
    var ctx = ctx_for(device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("optimizer", "AmpUpdateScale")
    call.int(scale.ptr)
    call.int(tracker_ptr)
    call.int(found_inf_ptr)
    call.tuple(factors)
    call.int(v_int(args[unsafe_offset=5]))
    call.int(cp)
    call.run()
    v_tensor(args[unsafe_offset=1]).bump_version()
    ret_ref(rets, 0, scale)
    _ = ctx


# --- fused optimizers (torch.optim's fused=True) ----------------------------
#
# `_fused_{adam,adamw,sgd,adagrad}_` have no CompositeExplicitAutograd
# registration (native_functions.yaml), so there is no sequential fallback:
# what the kernel cannot take raises, with the message stock CUDA raises
# (aten/src/ATen/native/cuda/Fused*Kernel.cu). Their autogen functional and
# `.out` variants are CompositeExplicitAutograd over these in-place ones.


# `flags` bits, as tmb/kernels/optimizer/kernels.mojo reads them.
comptime _FO_AMSGRAD = 1
comptime _FO_MAXIMIZE = 2
comptime _FO_NESTEROV = 4
comptime _FO_FIRST_STEP = 8
comptime _FO_MOMENTUM = 16


def _fused_same_sizes_and_strides(p: T, t: T) -> Bool:
    """`_check_tensors_share_sizes_and_strides`: a size-1 dim's stride is
    free."""
    if p.rank != t.rank:
        return False
    for i in range(p.rank):
        if p.dim(i) != t.dim(i):
            return False
        if p.dim(i) != 1 and p.stride(i) != t.stride(i):
            return False
    return True


def _fused_fast_path_ok(
    lists: List[List[T]], skip_cross_list_dtype: Bool
) -> Bool:
    """ATen's `check_fast_path_restrictions` (ForeachUtils.h): one device,
    non-overlapping and dense, one dtype per list (and across lists unless
    `skip_cross_list_dtype`), and every list laid out like the first. Dense
    tensors sharing strides are walked in storage order, as flat arrays."""
    var first = lists[0][0].copy()
    for i in range(len(lists)):
        if len(lists[i]) == 0:
            continue
        var list_dtype = lists[i][0].dtype
        for j in range(len(lists[i])):
            var t = lists[i][j].copy()
            if (
                not t.on_mojo()
                or t.device != first.device
                or not is_dense(t.shape, t.strides, t.rank)
                or t.dtype != list_dtype
                or (not skip_cross_list_dtype and t.dtype != first.dtype)
            ):
                return False
            if i > 0 and not _fused_same_sizes_and_strides(lists[0][j], t):
                return False
    return True


def _fused_check_device(
    t: T, name: StaticString, device: Int, dtype: DType = DType.float32
) raises -> Int:
    """A device scalar the kernel dereferences (grad_scale, found_inf, a
    device lr): stock CUDA's same-device check, then its `data_ptr<float>`
    dtype check. Its first element's address."""
    if not t.on_mojo() or t.device != device:
        raise Error(name, " must be on the same GPU device as the params")
    if t.dtype != dtype:
        raise Error(
            "expected scalar type Float but found ",
            _scalar_type_name(t.dtype),
        )
    if t.numel < 1:
        raise Error(name, " must have at least one element")
    return t.ptr


def _fused_cpu_lr(lr: T) raises -> Float64:
    """`lr.item<double>()` of a CPU tensor lr."""
    if lr.numel != 1:
        raise Error(
            "a Tensor with ",
            lr.numel,
            " elements cannot be converted to Scalar",
        )
    if lr.dtype == DType.float32:
        return Float64(
            Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=lr.ptr)[]
        )
    if lr.dtype == DType.float64:
        return Pointer[Float64, MutUntrackedOrigin](
            unsafe_from_address=lr.ptr
        )[]
    unsupported(
        "fused optimizers: a CPU lr tensor of dtype " + String(lr.dtype)
    )
    return 0.0


def _fused_optimizer(
    op: String,
    kernel_name: String,
    mixed_label: String,
    layout_msg: String,
    lists: List[List[T]],
    steps: List[T],
    lr_v: Value,
    var hyper: List[Float64],
    flags: Int,
    grad_scale_v: Value,
    found_inf_v: Value,
    mutated: List[List[T]],
) raises:
    """Validate like stock CUDA, then one batched launch.

    `lists` is what the fast-path check covers: params, grads, then the
    state lists the kernel reads (their order is the kernel's state0..2).
    `hyper[0]` is overwritten with the float lr when `lr_v` is a tensor on
    the CPU. `mutated` is every `Tensor(x!)[]` argument, for the version
    counters.
    """
    var params = lists[0].copy()
    var count = len(params)
    for i in range(1, len(lists)):
        if len(lists[i]) != count:
            raise Error(
                "Tensor lists must have the same number of tensors, got ",
                count,
                " and ",
                len(lists[i]),
            )
    if len(steps) != 0 and len(steps) != count:
        raise Error(
            "Tensor lists must have the same number of tensors, got ",
            count,
            " and ",
            len(steps),
        )
    if count == 0:
        return
    var first = params[0].copy()
    var device = first.device

    # A CPU lr tensor is `lr.item<double>()`: the float-lr overload.
    var lr_ptr = 0
    var lr_is_tensor = lr_v.tag == TAG_TENSOR or lr_v.tag == TAG_TENSOR_REF
    var lr_t = Optional[T](None)
    if lr_is_tensor:
        var t = v_tensor(lr_v)
        if t.on_cpu():
            hyper[0] = _fused_cpu_lr(t)
        else:
            lr_t = t^
    var grad_scale_ptr = 0
    var found_inf_ptr = 0
    if grad_scale_v.tag != TAG_NONE:
        grad_scale_ptr = _fused_check_device(
            v_tensor(grad_scale_v), "grad_scale", device
        )
    if found_inf_v.tag != TAG_NONE:
        found_inf_ptr = _fused_check_device(
            v_tensor(found_inf_v), "found_inf", device
        )
    if lr_t:
        lr_ptr = _fused_check_device(lr_t.value(), "lr", device)

    var mixed = (
        mixed_label != ""
        and len(lists) > 2
        and first.dtype != lists[2][0].dtype
    )
    if not _fused_fast_path_ok(lists, mixed):
        raise Error(layout_msg)
    var state_dtype = first.dtype
    if mixed:
        # validate_mixed_precision_dtypes (fused_adam_utils.cuh): float32
        # params and grads, bfloat16 states.
        var what = List[String]()
        what.append("params")
        what.append("grads")
        what.append("optimizer states")
        what.append("optimizer states")
        what.append("max_exp_avg_sqs")
        for i in range(len(lists)):
            var want = DType.float32 if i < 2 else DType.bfloat16
            var got = lists[i][0].dtype
            if got != want:
                raise Error(
                    mixed_label,
                    " requires ",
                    "float32 " if i < 2 else "bfloat16 ",
                    what[i],
                    ", got ",
                    _scalar_type_name(got),
                )
        state_dtype = DType.bfloat16
    elif (
        first.dtype != DType.float32
        and first.dtype != DType.float16
        and first.dtype != DType.bfloat16
        and first.dtype != DType.float64
    ):
        raise Error(
            '"',
            kernel_name,
            "\" not implemented for '",
            _scalar_type_name(first.dtype),
            "'",
        )
    if first.dtype == DType.float64 and dev(device)[].api == "metal":
        unsupported("fused optimizers: Apple GPUs have no float64")

    for i in range(len(steps)):
        var step = steps[i].copy()
        if (
            not step.on_mojo()
            or step.device != device
            or step.dtype != DType.float32
            or step.numel < 1
        ):
            raise Error(
                (
                    "fused optimizer state_steps must be float32 tensors on the"
                    " params' device (invalid index "
                ),
                i,
                ")",
            )

    var metadata = List[Int](capacity=count * 7)
    for j in range(count):
        metadata.append(lists[0][j].ptr)
        metadata.append(lists[1][j].ptr)
        for s in range(3):
            metadata.append(lists[2 + s][j].ptr if 2 + s < len(lists) else 0)
        metadata.append(steps[j].ptr if len(steps) != 0 else 0)
        metadata.append(lists[0][j].numel)
    var scalars = List[Int](capacity=5)
    for i in range(5):
        scalars.append(Int(f64_bits(hyper[i] if i < len(hyper) else 0.0)))

    var ctx = ctx_for(device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("optimizer", op)
    call.arg_dtype(0, first.dtype)
    call.arg_dtype(1, state_dtype)
    call.tuple(metadata)
    call.tuple(scalars)
    call.int(flags)
    call.int(lr_ptr)
    call.int(grad_scale_ptr)
    call.int(found_inf_ptr)
    call.int(cp)
    call.run()
    _ = ctx
    for group in mutated:
        for t in group:
            t.bump_version()


def _fused_adam_family(args: Values, adamw: Bool) raises:
    var amsgrad = v_bool(args[unsafe_offset=11])
    var maximize = v_bool(args[unsafe_offset=12])
    var lists = List[List[T]]()
    for i in range(4):
        lists.append(v_tensor_list(args[unsafe_offset=i]))
    var max_exp_avg_sqs = v_tensor_list(args[unsafe_offset=4])
    if amsgrad:
        lists.append(max_exp_avg_sqs.copy())
    var hyper = List[Float64]()
    var lr_v = args[unsafe_offset=6].copy()
    hyper.append(
        0.0 if lr_v.tag == TAG_TENSOR
        or lr_v.tag == TAG_TENSOR_REF else v_f64(lr_v)
    )
    for i in range(7, 11):
        hyper.append(v_f64(args[unsafe_offset=i]))
    var mutated = lists.copy()
    if not amsgrad:
        mutated.append(max_exp_avg_sqs^)
    _fused_optimizer(
        "FusedAdamW" if adamw else "FusedAdam",
        ("fused_adamw_kernel_cuda" if adamw else "fused_adam_kernel_cuda"),
        (
            "Mixed-precision fused AdamW" if adamw else "Mixed-precision fused Adam"
        ),
        "params, grads, exp_avgs, exp_avg_sqs, and max_exp_avg_sqs must have same dtype, device, and layout" if amsgrad else (
            "params, grads, exp_avgs, and exp_avg_sqs must have same dtype,"
            " device, and layout"
        ),
        lists,
        v_tensor_list(args[unsafe_offset=5]),
        lr_v,
        hyper^,
        (_FO_AMSGRAD if amsgrad else 0) | (_FO_MAXIMIZE if maximize else 0),
        args[unsafe_offset=13],
        args[unsafe_offset=14],
        mutated,
    )


# aten::_fused_adam_(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs,
#   Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *,
#   float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad,
#   bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
# aten::_fused_adam_.tensor_lr(..., Tensor lr, ...) -> ()
def op_fused_adam_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _fused_adam_family(args, False)


# aten::_fused_adamw_(...same as _fused_adam_...) -> ()
# aten::_fused_adamw_.tensor_lr(..., Tensor lr, ...) -> ()
def op_fused_adamw_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _fused_adam_family(args, True)


# aten::_fused_sgd_(Tensor(a!)[] self, Tensor(b!)[] grads,
#   Tensor(c!)[] momentum_buffer_list, *, float weight_decay, float momentum,
#   float lr, float dampening, bool nesterov, bool maximize, bool is_first_step,
#   Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
# aten::_fused_sgd_.tensor_lr(..., Tensor lr, ...) -> ()
def op_fused_sgd_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var lists = List[List[T]]()
    lists.append(v_tensor_list(args[unsafe_offset=0]))
    lists.append(v_tensor_list(args[unsafe_offset=1]))
    var buffers = v_tensor_list(args[unsafe_offset=2])
    var weight_decay = v_f64(args[unsafe_offset=3])
    var momentum = v_f64(args[unsafe_offset=4])
    var lr_v = args[unsafe_offset=5].copy()
    var dampening = v_f64(args[unsafe_offset=6])
    var nesterov = v_bool(args[unsafe_offset=7])
    var maximize = v_bool(args[unsafe_offset=8])
    var is_first_step = v_bool(args[unsafe_offset=9])
    var has_momentum = len(buffers) != 0
    var layout_msg: String
    if has_momentum:
        if not momentum > 0:
            raise Error("Check failed: momentum > 0 (", momentum, " vs. 0). ")
        lists.append(buffers.copy())
        layout_msg = (
            "Expected at::native::check_fast_path_restrictions( {params,"
            " grads, momentum_buffer_list}) to be true, but got false."
        )
    else:
        if momentum != 0:
            raise Error("Check failed: momentum == 0 (", momentum, " vs. 0). ")
        layout_msg = (
            "Expected at::native::check_fast_path_restrictions({params,"
            " grads}) to be true, but got false."
        )
    var hyper = List[Float64]()
    hyper.append(
        0.0 if lr_v.tag == TAG_TENSOR
        or lr_v.tag == TAG_TENSOR_REF else v_f64(lr_v)
    )
    hyper.append(weight_decay)
    hyper.append(momentum)
    hyper.append(dampening)
    hyper.append(0.0)
    var flags = (
        (_FO_MAXIMIZE if maximize else 0)
        | (_FO_NESTEROV if nesterov else 0)
        | (_FO_FIRST_STEP if is_first_step and has_momentum else 0)
        | (_FO_MOMENTUM if has_momentum else 0)
    )
    var mutated = lists.copy()
    if not has_momentum:
        mutated.append(buffers^)
    _fused_optimizer(
        "FusedSgd",
        (
            "fused_sgd_with_momentum_kernel_cuda" if has_momentum else "fused_sgd_kernel_cuda"
        ),
        "",
        layout_msg,
        lists,
        List[T](),
        lr_v,
        hyper^,
        flags,
        args[unsafe_offset=10],
        args[unsafe_offset=11],
        mutated,
    )


# aten::_fused_adagrad_(Tensor(a!)[] self, Tensor(b!)[] grads,
#   Tensor(c!)[] state_sums, Tensor(d!)[] state_steps, *, float lr,
#   float lr_decay, float weight_decay, float eps, bool maximize,
#   Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
# aten::_fused_adagrad_.tensor_lr(..., Tensor lr, ...) -> ()
def op_fused_adagrad_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var lists = List[List[T]]()
    for i in range(3):
        lists.append(v_tensor_list(args[unsafe_offset=i]))
    var steps = v_tensor_list(args[unsafe_offset=3])
    var lr_v = args[unsafe_offset=4].copy()
    var hyper = List[Float64]()
    hyper.append(
        0.0 if lr_v.tag == TAG_TENSOR
        or lr_v.tag == TAG_TENSOR_REF else v_f64(lr_v)
    )
    for i in range(5, 8):
        hyper.append(v_f64(args[unsafe_offset=i]))
    hyper.append(0.0)
    var mutated = lists.copy()
    mutated.append(steps.copy())
    _fused_optimizer(
        "FusedAdagrad",
        "fused_adagrad_kernel_cuda",
        "",
        (
            "params, grads, and state_sums must have same dtype, device, and"
            " layout"
        ),
        lists,
        steps,
        lr_v,
        hyper^,
        _FO_MAXIMIZE if v_bool(args[unsafe_offset=8]) else 0,
        args[unsafe_offset=9],
        args[unsafe_offset=10],
        mutated,
    )


def register_foreach(site: Site) raises:
    impl[
        op_amp_foreach_non_finite_check_and_unscale_,
        "_amp_foreach_non_finite_check_and_unscale_",
    ](site)
    impl[op_amp_update_scale_, "_amp_update_scale_"](site)
    impl[op_foreach_copy_, "_foreach_copy_"](site)
    impl[op_foreach_add_scalar_, "_foreach_add_.Scalar"](site)
    impl[op_foreach_addcmul_scalar_, "_foreach_addcmul_.Scalar"](site)
    impl[op_foreach_lerp_scalar_, "_foreach_lerp_.Scalar"](site)
    impl[op_foreach_mul_scalar_, "_foreach_mul_.Scalar"](site)
    impl[op_foreach_mul_tensor_, "_foreach_mul_.Tensor"](site)
    impl[op_foreach_norm_scalar, "_foreach_norm.Scalar"](site)
    impl[op_foreach_sqrt, "_foreach_sqrt"](site)
    impl[op_fused_adam_, "_fused_adam_"](site)
    impl[op_fused_adam_, "_fused_adam_.tensor_lr"](site)
    impl[op_fused_adamw_, "_fused_adamw_"](site)
    impl[op_fused_adamw_, "_fused_adamw_.tensor_lr"](site)
    impl[op_fused_sgd_, "_fused_sgd_"](site)
    impl[op_fused_sgd_, "_fused_sgd_.tensor_lr"](site)
    impl[op_fused_adagrad_, "_fused_adagrad_"](site)
    impl[op_fused_adagrad_, "_fused_adagrad_.tensor_lr"](site)
    # _foreach_div_.ScalarList / _foreach_addcdiv_.ScalarList: intentionally
    # unregistered -- see the module docstring (Scalar[] cannot be marshalled
    # by the current C++ shim).
