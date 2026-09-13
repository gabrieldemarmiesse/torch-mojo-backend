"""aten ops: foreach group (see docs/native_backend.md).

Every `_foreach_*_` / `_fused_adamw_` op below tries ONE batched kernel
launch (the `optimizer_ops` family, ported host-side from
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

`_fused_adamw_` / `_fused_adamw_.tensor_lr` are the one exception: they have
no CompositeExplicitAutograd registration in ATen at all (see
native_functions.yaml), so there is no equivalent sequential fallback to
call into. Declining there really does mean NotImplementedError, matching
the old `_register_fast` (no-fallback) binding exactly.

`_foreach_div_.ScalarList` and `_foreach_addcdiv_.ScalarList` are NOT
registered here: there is no batched kernel for them, and with no
PrivateUse1 override ATen's own CompositeExplicitAutograd decomposition
(`foreach_tensor_div_scalarlist_kernel_slow_` /
`..._addcdiv_scalarlist_kernel_slow_`) already does exactly what the
sequential fallbacks below do -- one `div_.Scalar` / `addcdiv_` per tensor.
(Their `Scalar[]` argument does marshal: `to_record`'s `ListType` branch in
`native/csrc/shim_dispatch.cpp` has a `NumberType` arm.)
"""
from std.math import ceildiv
from std.utils import IndexList

from abi import (
    ST_FLOAT32,
    T,
    Value,
    Values,
    call_op,
    dtype_code,
    f64_bits,
    new_tensor,
    own,
    Owned,
    ret_tensor_list,
    unsupported,
    v_bool,
    v_dtype_or,
    v_f64,
    v_scalar_is_bool,
    v_tensor,
    v_tensor_list,
    TAG_NONE,
    TAG_SCALAR_INT,
    TAG_TENSOR,
    TAG_TENSOR_REF,
)
from device import ctx_for, ctx_ptr, dev
from foreach_clip_contract import FOREACH_CHUNK_ELEMENTS
from kernels import KernelCall
from op_utils import MAX_RANK
from registry import Site, impl, op_address_of


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
# every list mojo-resident, contiguous, on one shared (non-MAX-cpu) device,
# one shared dtype, with index-aligned tensors sharing a shape ------------


def _is_max_cpu(device: Int) raises -> Bool:
    return dev(device)[].is_cpu


def _tensor_qualifies(t: T, device: Int, dtype: DType) -> Bool:
    return t.on_mojo() and t.device == device and t.dtype == dtype and t.contig


def _qualifies1(a: List[T], allow_half: Bool) raises -> Bool:
    if len(a) == 0:
        return False
    var first = a[0].copy()
    if not first.on_mojo() or _is_max_cpu(first.device):
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


# --- batched launches (optimizer_ops family: ForeachMul/Add/MulTensor/Lerp/
# Addcmul/Sqrt/L2Norm, FusedAdamW -- see optimizer_ops.mojo's `tmb_call`
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
    var call = KernelCall("optimizer_ops", String(op_name))
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
    var call = KernelCall("optimizer_ops", "ForeachMulTensor")
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
    self_list: List[T], end_list: List[T], weight: Float64
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
    var call = KernelCall("optimizer_ops", "ForeachLerp")
    call.arg_dtype(0, dtype)
    call.arg_dtype(1, dtype)
    call.out_dtype(dtype)
    call.tuple(metadata)
    call.tuple(scalars)
    call.tuple(aux)
    call.int(dtype_code(dtype))
    call.int(cp)
    call.run()
    for t in self_list:
        t.bump_version()
    _ = ctx


def _foreach_addcmul_launch(
    self_list: List[T], t1_list: List[T], t2_list: List[T], value: Float64
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
    var call = KernelCall("optimizer_ops", "ForeachAddcmul")
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
    var call = KernelCall("optimizer_ops", "ForeachSqrt")
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
    var call = KernelCall("optimizer_ops", "ForeachL2Norm")
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
        _foreach_addcmul_launch(self_list, t1_list, t2_list, v_f64(value_v))
        return
    for i in range(len(self_list)):
        _seq_addcmul_(self_list[i], t1_list[i], t2_list[i], value_v)


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


def _adamw_scalar_tensor(
    v: Value, name: StaticString, device: Int
) raises -> Int:
    """A validated read-only scalar (`grad_scale`/`found_inf`): its device
    pointer, or 0 for None."""
    if v.tag == TAG_NONE:
        return 0
    var t = v_tensor(v)
    if (
        t.device != device
        or not t.on_mojo()
        or t.dtype != DType.float32
        or t.numel != 1
        or not t.contig
    ):
        raise Error(
            name,
            (
                " must be a contiguous scalar float32 tensor on the same mojo"
                " device as the parameters"
            ),
        )
    return t.ptr


def _fused_adamw_impl(args: Values, n_args: Int) raises:
    var parameters = v_tensor_list(args[unsafe_offset=0])
    var grads = v_tensor_list(args[unsafe_offset=1])
    var exp_avgs = v_tensor_list(args[unsafe_offset=2])
    var exp_avg_sqs = v_tensor_list(args[unsafe_offset=3])
    var max_exp_avg_sqs = v_tensor_list(args[unsafe_offset=4])
    var state_steps = v_tensor_list(args[unsafe_offset=5])
    var amsgrad = v_bool(args[unsafe_offset=11])
    var maximize = v_bool(args[unsafe_offset=12])

    var tensor_count = len(parameters)
    if (
        len(grads) != tensor_count
        or len(exp_avgs) != tensor_count
        or len(exp_avg_sqs) != tensor_count
        or len(state_steps) != tensor_count
    ):
        raise Error(
            "fused AdamW tensor lists must have the same length as parameters"
        )
    if amsgrad:
        if len(max_exp_avg_sqs) != tensor_count:
            raise Error(
                "max_exp_avg_sqs must have the same length as parameters"
            )
    elif len(max_exp_avg_sqs) != 0:
        raise Error("max_exp_avg_sqs must be empty when amsgrad is False")
    if tensor_count == 0:
        return

    var first = parameters[0].copy()
    if not first.on_mojo():
        unsupported("fused AdamW: parameters are not on a mojo device")
    var device = first.device

    var metadata = List[Int]()
    for i in range(tensor_count):
        var p = parameters[i].copy()
        if (
            not p.on_mojo()
            or p.device != device
            or p.dtype != DType.float32
            or not p.contig
        ):
            raise Error(
                (
                    "fused AdamW tensors must have the same dtype, device,"
                    " shape, and numel; contiguous float32 is required (invalid"
                    " tensor index "
                ),
                i,
                ")",
            )
        var g = grads[i].copy()
        var ea = exp_avgs[i].copy()
        var eas = exp_avg_sqs[i].copy()
        if (
            not g.on_mojo()
            or g.device != device
            or g.dtype != DType.float32
            or not g.contig
            or not g.same_shape(p)
            or not ea.on_mojo()
            or ea.device != device
            or ea.dtype != DType.float32
            or not ea.contig
            or not ea.same_shape(p)
            or not eas.on_mojo()
            or eas.device != device
            or eas.dtype != DType.float32
            or not eas.contig
            or not eas.same_shape(p)
        ):
            raise Error(
                (
                    "fused AdamW tensors must have the same dtype, device,"
                    " shape, and numel; contiguous float32 is required (invalid"
                    " tensor index "
                ),
                i,
                ")",
            )
        var max_eas_ptr = 0
        if amsgrad:
            var meas = max_exp_avg_sqs[i].copy()
            if (
                not meas.on_mojo()
                or meas.device != device
                or meas.dtype != DType.float32
                or not meas.contig
                or not meas.same_shape(p)
            ):
                raise Error(
                    (
                        "fused AdamW tensors must have the same dtype, device,"
                        " shape, and numel; contiguous float32 is required"
                        " (invalid tensor index "
                    ),
                    i,
                    ")",
                )
            max_eas_ptr = meas.ptr
        var step = state_steps[i].copy()
        if (
            not step.on_mojo()
            or step.device != device
            or step.dtype != DType.float32
            or step.numel != 1
            or not step.contig
        ):
            raise Error(
                (
                    "fused AdamW state_steps must be contiguous scalar float32"
                    " tensors on the parameter device (invalid index "
                ),
                i,
                ")",
            )
        metadata.append(p.ptr)
        metadata.append(g.ptr)
        metadata.append(ea.ptr)
        metadata.append(eas.ptr)
        metadata.append(max_eas_ptr)
        metadata.append(step.ptr)
        metadata.append(p.numel)

    var lr_v = args[unsafe_offset=6].copy()
    var lr_scalar = 0.0
    var lr_ptr = 0
    if lr_v.tag == TAG_TENSOR or lr_v.tag == TAG_TENSOR_REF:
        var lr_t = v_tensor(lr_v)
        if lr_t.on_mojo():
            lr_ptr = _adamw_scalar_tensor(lr_v, "lr", device)
        else:
            if lr_t.numel != 1:
                raise Error("tensor lr must be a scalar CPU or mojo tensor")
            lr_scalar = Float64(
                Pointer[Float32, MutUntrackedOrigin](
                    unsafe_from_address=lr_t.ptr
                )[]
            )
    else:
        lr_scalar = v_f64(lr_v)

    var beta1 = v_f64(args[unsafe_offset=7])
    var beta2 = v_f64(args[unsafe_offset=8])
    var weight_decay = v_f64(args[unsafe_offset=9])
    var eps = v_f64(args[unsafe_offset=10])
    var grad_scale_ptr = _adamw_scalar_tensor(
        args[unsafe_offset=13], "grad_scale", device
    )
    var found_inf_ptr = _adamw_scalar_tensor(
        args[unsafe_offset=14], "found_inf", device
    )

    var flags_int = (1 if amsgrad else 0) | ((1 if maximize else 0) << 1)
    var scalars = List[Int]()
    scalars.append(Int(f64_bits(lr_scalar)))
    scalars.append(Int(f64_bits(beta1)))
    scalars.append(Int(f64_bits(beta2)))
    scalars.append(Int(f64_bits(weight_decay)))
    scalars.append(Int(f64_bits(eps)))

    var ctx = ctx_for(device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("optimizer_ops", "FusedAdamW")
    call.arg_dtype(0, DType.float32)
    call.arg_dtype(1, DType.float32)
    call.arg_dtype(2, DType.float32)
    call.arg_dtype(3, DType.float32)
    call.arg_dtype(4, DType.float32)
    call.out_dtype(DType.float32)
    call.flag("AMSGRAD", 1 if amsgrad else 0)
    call.flag("MAXIMIZE", 1 if maximize else 0)
    call.flag("TENSOR_LR", 1 if lr_ptr != 0 else 0)
    call.flag("GRAD_SCALE", 1 if grad_scale_ptr != 0 else 0)
    call.flag("FOUND_INF", 1 if found_inf_ptr != 0 else 0)
    call.tuple(metadata)
    call.tuple(scalars)
    call.int(0)
    call.int(flags_int)
    call.int(lr_ptr)
    call.int(grad_scale_ptr)
    call.int(found_inf_ptr)
    call.int(cp)
    call.run()
    # Every list the kernel writes: `grads` too, which it overwrites with
    # the unscaled gradient when grad_scale is given.
    for t in parameters:
        t.bump_version()
    for t in grads:
        t.bump_version()
    for t in exp_avgs:
        t.bump_version()
    for t in exp_avg_sqs:
        t.bump_version()
    for t in max_exp_avg_sqs:
        t.bump_version()
    _ = ctx


# aten::_fused_adamw_(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs,
#   Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *,
#   float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad,
#   bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
# aten::_fused_adamw_.tensor_lr(..., Tensor lr, ...) -> ()
def op_fused_adamw_(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _fused_adamw_impl(args, n_args)


def register_foreach(site: Site) raises:
    impl[op_foreach_add_scalar_, "_foreach_add_.Scalar"](site)
    impl[op_foreach_addcmul_scalar_, "_foreach_addcmul_.Scalar"](site)
    impl[op_foreach_lerp_scalar_, "_foreach_lerp_.Scalar"](site)
    impl[op_foreach_mul_scalar_, "_foreach_mul_.Scalar"](site)
    impl[op_foreach_mul_tensor_, "_foreach_mul_.Tensor"](site)
    impl[op_foreach_norm_scalar, "_foreach_norm.Scalar"](site)
    impl[op_foreach_sqrt, "_foreach_sqrt"](site)
    impl[op_fused_adamw_, "_fused_adamw_"](site)
    impl[op_fused_adamw_, "_fused_adamw_.tensor_lr"](site)
    # _foreach_div_.ScalarList / _foreach_addcdiv_.ScalarList: intentionally
    # unregistered -- see the module docstring (Scalar[] cannot be marshalled
    # by the current C++ shim).


@export
def tmb_op_address() abi("C") -> Int:
    """Entry of this file's one-op extension: the address of the op the
    TMB_OP define selected (registry.mojo)."""
    return op_address_of[register_foreach]()
