"""ATen ops: factories group (see docs/native_backend.md).

Most of this group's aten surface -- full/zeros/ones/new_*/scalar_tensor/
empty_like/*_like -- is CompositeExplicitAutograd upstream (verified against
the installed torch build): it decomposes into `empty.memory_format` +
`fill_.Scalar`, both already registered by ops_core.mojo, so a fused
registration here would run the identical two calls and buy nothing. See the
final report for the exact list skipped on that basis.

What actually needs a kernel here (no PrivateUse1 fallback exists upstream):
`arange.start_out` (arange/arange.start/arange.start_step are themselves
CompositeExplicitAutograd and call this through `empty` + `arange_out`, so
registering only the `out` overload is enough). The random factories
(uniform_, normal_, random_, ...) and native_dropout are ops_random.mojo.
"""
from std.math import ceil
from std.utils import IndexList

from abi import (
    T,
    Value,
    Values,
    cpu_empty,
    dtype_code,
    new_like,
    own,
    ret_ref,
    unsupported,
    v_f64,
    v_int,
    v_scalar_is_integral,
    v_tensor,
    TAG_TENSOR,
)
from device import copy_from_host, ctx_for, ctx_ptr, dev
from kernels import KernelCall
from op_utils import MAX_RANK
from ops_common import call_op, copy_strided_into, resize_out
from registry import Site, impl


# ---------------------------------------------------------------------------
# arange.start_out
#
# arange / arange.start / arange.start_step are CompositeExplicitAutograd
# (aten/src/ATen/native/TensorFactories.cpp: `result = at::empty({0},
# options); return at::arange_out(result, start, end, step)`), so torch
# itself resolves dtype/device and hands this kernel a fresh, empty `out` to
# grow. No separate registration of those three earns anything beyond what
# this one buys them for free.
# ---------------------------------------------------------------------------


# The dtypes the Arange kernel (elementwise_ops family) is built for -- no
# bool, no 16/32/64-bit unsigned (aten_fast.py `_ARANGE_DTYPES`).
comptime _ARANGE_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.int64,
    DType.int32,
    DType.int16,
    DType.int8,
    DType.uint8,
]


def _is_finite(x: Float64) -> Bool:
    """`x - x == 0`: true for every finite float, false for +-inf and NaN --
    avoids pulling in std.utils.numerics.isfinite's SIMD-mask return just for
    one scalar check."""
    return (x - x) == 0.0


def _is_arange_kernel_dtype(dt: DType) -> Bool:
    comptime for d in _ARANGE_DTYPES:
        if dt == d:
            return True
    return False


# Values cross into the kernel as Float64 (`call.f64`); one beyond this
# magnitude cannot round-trip exactly (aten_ops/factories.py's
# `_MAX_EXACT_F64_INT`), so it forces the host fallback regardless of dtype.
comptime _MAX_EXACT_F64_INT: Float64 = 9007199254740992.0  # 2**53


def _arange_needs_host_fallback(
    start_v: Value, end_v: Value, step_v: Value, dt: DType, device: Int
) raises -> Bool:
    if not _is_arange_kernel_dtype(dt):
        return True
    if dt == DType.float64 and dev(device)[].api == "metal":
        return True
    if (
        abs(v_f64(start_v)) > _MAX_EXACT_F64_INT
        or abs(v_f64(end_v)) > _MAX_EXACT_F64_INT
        or abs(v_f64(step_v)) > _MAX_EXACT_F64_INT
    ):
        return True
    return False


def _arange_numel(start_v: Value, end_v: Value, step_v: Value) raises -> Int:
    """`aten::arange`'s output length, mirroring
    `at::native::compute_arange_size` (RangeUtils.h): bounds/sign checks in
    double precision, then an exact integer ceiling division when every
    scalar is integral (`-(-(end-start)//step)`, the old host path's
    formula and Mojo's `//` floors like Python's), else `ceil` in double."""
    var dstart = v_f64(start_v)
    var dend = v_f64(end_v)
    var dstep = v_f64(step_v)
    if dstep == 0.0:
        raise Error("step must be nonzero")
    if not (_is_finite(dstart) and _is_finite(dend)):
        raise Error("unsupported range: ", dstart, " -> ", dend)
    if not (
        (dstep > 0.0 and dend >= dstart) or (dstep < 0.0 and dend <= dstart)
    ):
        raise Error("upper bound and lower bound inconsistent with step sign")
    if (
        v_scalar_is_integral(start_v)
        and v_scalar_is_integral(end_v)
        and v_scalar_is_integral(step_v)
    ):
        var size = -(-(v_int(end_v) - v_int(start_v)) // v_int(step_v))
        return max(size, 0)
    var size_d = ceil((dend - dstart) / dstep)
    if size_d < 0.0:
        return 0
    return Int(size_d)


def _arange_fill(out_t: T, start: Float64, step: Float64) raises:
    """Writes `out_t[i] = start + i*step` for a contiguous, kernel-supported
    `out_t` (elementwise_ops Arange; aten_fast.py `fast_arange`)."""
    var ctx = ctx_for(out_t.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("elementwise_ops", "Arange")
    call.out_dtype(out_t.dtype)
    call.int(out_t.ptr)
    call.f64(start)
    call.f64(step)
    call.int(out_t.numel)
    call.int(dtype_code(out_t.dtype))
    call.int(cp)
    call.run()
    _ = ctx


def _host_arange_start_out(
    start_v: Value, end_v: Value, step_v: Value, out_t: T
) raises:
    """Host fallback for a dtype the device kernel doesn't cover, or a
    magnitude/precision the device accumulator can't (`_host_arange_tensor`
    in aten_ops/factories.py): builds the range on a CPU tensor of `out`'s
    already-resolved dtype/shape through the real `arange.start_out` CPU
    kernel, then copies it onto the mojo device."""
    var cpu = own(cpu_empty(out_t.shape, out_t.rank, out_t.stype))
    var call_args = List[Value](capacity=4)
    call_args.append(start_v.copy())
    call_args.append(end_v.copy())
    call_args.append(step_v.copy())
    call_args.append(Value(TAG_TENSOR, 0, Int64(cpu.t.h), 0))
    # `Results` releases tmb_call_op's own fresh wrapper handle.
    _ = call_op("aten::arange", "start_out", call_args^, 1)
    _copy_cpu_into(out_t, cpu.t)
    _ = cpu^  # alive past the launch


def _copy_cpu_into(dst: T, src: T) raises:
    """`dst` (mojo device) = `src` (a contiguous CPU tensor, same
    dtype/shape): the host->device half of ops_core.op_copy_from, inlined
    because normal_ and the arange host fallback are the only factories
    callers of exactly this shape."""
    if dst.numel == 0:
        return
    var nbytes = dst.numel * dst.itemsize
    if dst.contig:
        copy_from_host(
            dst.device, ctx_for(dst.device), dst.ptr, src.ptr, nbytes
        )
    else:
        var tmp = own(new_like(dst))
        copy_from_host(
            dst.device, ctx_for(dst.device), tmp.t.ptr, src.ptr, nbytes
        )
        copy_strided_into(dst, tmp.t)
        _ = tmp^  # alive past the launch


# aten::arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)
def op_arange_start_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var start_v = args[unsafe_offset=0].copy()
    var end_v = args[unsafe_offset=1].copy()
    var step_v = args[unsafe_offset=2].copy()
    var out_t = v_tensor(args[unsafe_offset=3])
    var numel = _arange_numel(start_v, end_v, step_v)
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 1] = numel
    resize_out(out_t, shape, 1)  # a 1-d `out` of this length is left alone
    if numel == 0:
        ret_ref(rets, 0, out_t)
        return
    if _arange_needs_host_fallback(
        start_v, end_v, step_v, out_t.dtype, out_t.device
    ):
        _host_arange_start_out(start_v, end_v, step_v, out_t)
        ret_ref(rets, 0, out_t)
        return
    var start = v_f64(start_v)
    var step = v_f64(step_v)
    if out_t.contig:
        _arange_fill(out_t, start, step)
    else:
        var tmp = own(new_like(out_t))
        _arange_fill(tmp.t, start, step)
        copy_strided_into(out_t, tmp.t)
        _ = tmp^  # alive past the launch
    ret_ref(rets, 0, out_t)


def register_factories(site: Site) raises:
    impl[op_arange_start_out, "arange.start_out"](site)
