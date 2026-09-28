"""ATen ops: factories group (see agents_docs/native_backend.md).

Most of this group's aten surface -- full/zeros/ones/new_*/scalar_tensor/
empty_like/*_like -- is CompositeExplicitAutograd upstream (verified against
the installed torch build): it decomposes into `empty.memory_format` +
`fill_.Scalar`, both already registered by tmb/ops/core.mojo, so a fused
registration here would run the identical two calls and buy nothing. See the
final report for the exact list skipped on that basis.

What actually needs a kernel here (no PrivateUse1 fallback exists upstream):
`arange.start_out` (arange/arange.start/arange.start_step are themselves
CompositeExplicitAutograd and call this through `empty` + `arange_out`, so
registering only the `out` overload is enough), and for the same reason
`eye.out` / `eye.m_out`, `linspace.out` / `logspace.out` (their functional
and Tensor-argument overloads are composites that end there), plus
`tril_indices` / `triu_indices`, which have per-backend kernels upstream. The random factories
(uniform_, normal_, random_, ...) and native_dropout are tmb/ops/random.mojo.
"""
from std.math import ceil, pow
from std.utils import IndexList

from tmb.backend.abi import (
    ST_INT32,
    ST_INT64,
    T,
    Value,
    Values,
    cpu_empty,
    dtype_code,
    new_like,
    new_tensor,
    own,
    ret_owned,
    ret_ref,
    unsupported,
    v_dtype_or,
    v_f64,
    v_int,
    v_is_none,
    v_scalar_is_bool,
    v_scalar_is_integral,
    v_tensor,
    TAG_TENSOR,
)
from tmb.backend.device import copy_from_host, ctx_for, ctx_ptr, dev
from tmb.backend.kernel_call import KernelCall
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.common import (
    assert_no_internal_overlap,
    call_op,
    copy_strided_into,
    fill_value,
    resize_out,
)
from tmb.ops.data_movement import _resolve_device, _scalar_type_name
from tmb.backend.registry import Site, impl


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


# The dtypes the Arange kernel (elementwise family) is built for -- no
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
    `out_t` (elementwise Arange; aten_fast.py `fast_arange`)."""
    var ctx = ctx_for(out_t.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("elementwise", "Arange")
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


# ---------------------------------------------------------------------------
# eye.out / eye.m_out -- ATen's `eye_out_cuda` (native/cuda/TensorFactories.cu):
# resize to (n, m), zero it, then fill the diagonal view
# `as_strided({min(n, m)}, {stride(0) + stride(1)})` with one.
# ---------------------------------------------------------------------------


def _diagonal_raw(t: T) -> T:
    """The main diagonal of the 2-D `t` as a kernel-only view (same handle,
    so never hand it to the dispatcher)."""
    var d = t.copy()
    var n = min(t.dim(0), t.dim(1))
    d.rank = 1
    d.shape = IndexList[MAX_RANK](1)
    d.strides = IndexList[MAX_RANK](0)
    d.shape[MAX_RANK - 1] = n
    d.strides[MAX_RANK - 1] = t.stride(0) + t.stride(1)
    d.numel = n
    d.contig = n <= 1 or d.strides[MAX_RANK - 1] == 1
    return d^


def _eye_into(mut out_t: T, n: Int, m: Int) raises:
    if n < 0:
        raise Error("n must be greater or equal to 0, got ", n)
    if m < 0:
        raise Error("m must be greater or equal to 0, got ", m)
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 2] = n
    shape[MAX_RANK - 1] = m
    resize_out(out_t, shape, 2)
    assert_no_internal_overlap(out_t)
    if out_t.numel == 0:
        return
    fill_value(out_t, 0.0)
    fill_value(_diagonal_raw(out_t), 1.0)


# aten::eye.out(SymInt n, *, Tensor(a!) out) -> Tensor(a!)
def op_eye_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var n = v_int(args[unsafe_offset=0])
    var out_t = v_tensor(args[unsafe_offset=1])
    _eye_into(out_t, n, n)
    ret_ref(rets, 0, out_t)


# aten::eye.m_out(SymInt n, SymInt m, *, Tensor(a!) out) -> Tensor(a!)
def op_eye_m_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var n = v_int(args[unsafe_offset=0])
    var m = v_int(args[unsafe_offset=1])
    var out_t = v_tensor(args[unsafe_offset=2])
    _eye_into(out_t, n, m)
    ret_ref(rets, 0, out_t)


# ---------------------------------------------------------------------------
# linspace.out / logspace.out -- ATen's `linspace_cuda_out` /
# `logspace_cuda_out` (native/cuda/RangeFactories.cu). The host rounds the
# endpoints and the step the way those functions do (in the output dtype
# for floats, c10::Half / c10::BFloat16 arithmetic for the half types, a
# float32 step for integers); the elementwise `Linspace` kernel does the
# per-element half of the arithmetic.
# ---------------------------------------------------------------------------


@always_inline
def _round_to[dt: DType](x: Float32) -> Float64:
    """`x` rounded to the half type `dt` (c10 converts through float)."""
    return x.cast[dt]().cast[DType.float64]()


def _scalar_to_int(v: Value, dt: DType) raises -> Int:
    """`Scalar::to<scalar_t>()` for an integer `dt`, after
    `_check_convertible`: an integral Scalar converts modulo 2^bits (a
    negative one into uint8 wraps, -1 -> 255), a floating one truncates
    toward zero."""
    if not v_scalar_is_integral(v):
        return Int(v_f64(v))
    var x = v_int(v)
    if dt == DType.uint8:
        return x & 0xFF
    return x


def _linspace_params(
    dt: DType, start_v: Value, end_v: Value, steps: Int, is_log: Bool
) raises -> Tuple[Float64, Float64, Float64]:
    """(start, end, step), each exactly representable in the type the
    kernel computes in, rounded as ATen's CUDA host code rounds them."""
    var den = steps - 1
    if dt == DType.float64:
        var s = v_f64(start_v)
        var e = v_f64(end_v)
        return (s, e, (e - s) / Float64(den))
    if dt == DType.float32:
        var s = v_f64(start_v).cast[DType.float32]()
        var e = v_f64(end_v).cast[DType.float32]()
        var st = (e - s) / Float32(den)
        return (
            s.cast[DType.float64](),
            e.cast[DType.float64](),
            st.cast[DType.float64](),
        )
    if dt == DType.float16 or dt == DType.bfloat16:
        # c10::Half / c10::BFloat16: `start.to<scalar_t>()` goes double ->
        # float -> half, and every operator computes in float and rounds.
        var s: Float64
        var e: Float64
        var diff: Float64
        var d: Float64
        var st: Float64
        if dt == DType.float16:
            s = _round_to[DType.float16](v_f64(start_v).cast[DType.float32]())
            e = _round_to[DType.float16](v_f64(end_v).cast[DType.float32]())
            diff = _round_to[DType.float16](
                e.cast[DType.float32]() - s.cast[DType.float32]()
            )
            d = _round_to[DType.float16](Float32(den))
            st = _round_to[DType.float16](
                diff.cast[DType.float32]() / d.cast[DType.float32]()
            )
        else:
            s = _round_to[DType.bfloat16](v_f64(start_v).cast[DType.float32]())
            e = _round_to[DType.bfloat16](v_f64(end_v).cast[DType.float32]())
            diff = _round_to[DType.bfloat16](
                e.cast[DType.float32]() - s.cast[DType.float32]()
            )
            d = _round_to[DType.bfloat16](Float32(den))
            st = _round_to[DType.bfloat16](
                diff.cast[DType.float32]() / d.cast[DType.float32]()
            )
        return (s, e, st)
    # Integers: `scalar_t` endpoints, a float32 step. linspace subtracts in
    # float, logspace in the integer type (the two CUDA kernels differ).
    var si = _scalar_to_int(start_v, dt)
    var ei = _scalar_to_int(end_v, dt)
    var step: Float32
    if is_log:
        step = Float32(ei - si) / Float32(den)
    else:
        step = (Float32(ei) - Float32(si)) / Float32(den)
    return (
        Float32(si).cast[DType.float64](),
        Float32(ei).cast[DType.float64](),
        step.cast[DType.float64](),
    )


def _c10_type_name(dt: DType) -> String:
    """How `c10::checked_convert` names a type in its overflow error."""
    if dt == DType.float16:
        return "c10::Half"
    if dt == DType.bfloat16:
        return "c10::BFloat16"
    if dt == DType.float32:
        return "float"
    if dt == DType.float64:
        return "double"
    if dt == DType.int64:
        return "int64_t"
    if dt == DType.int32:
        return "int"
    if dt == DType.int16:
        return "int16_t"
    if dt == DType.int8:
        return "int8_t"
    return "uint8_t"


def _overflow_error(dt: DType) -> Error:
    return Error(
        "value cannot be converted to type ",
        _c10_type_name(dt),
        " without overflow",
    )


def _float_limits(dt: DType) -> Float64:
    """The largest finite value of the floating `dt`."""
    if dt == DType.float16:
        return 65504.0
    if dt == DType.bfloat16:
        return 3.3895313892515355e38
    if dt == DType.float32:
        return 3.4028234663852886e38
    return 1.7976931348623157e308


def _int_limits(dt: DType) -> Tuple[Int, Int]:
    if dt == DType.int8:
        return (-128, 127)
    if dt == DType.uint8:
        return (0, 255)
    if dt == DType.int16:
        return (-32768, 32767)
    if dt == DType.int32:
        return (-2147483648, 2147483647)
    return (-9223372036854775808, 9223372036854775807)


def _check_float_convertible(x: Float64, dt: DType) raises:
    """`c10::overflows<scalar_t, double>`: into a floating type, infinities
    and NaN pass and a finite value past the largest one fails; into an
    integer type, NaN, infinities and anything outside [lowest, max + 1)
    fail."""
    if dt == DType.bool:
        return
    if dt.is_floating_point():
        if x != x or not _is_finite(x):
            return
        if abs(x) > _float_limits(dt):
            raise _overflow_error(dt)
        return
    if x != x or not _is_finite(x):
        raise _overflow_error(dt)
    var lim = _int_limits(dt)
    if x < Float64(lim[0]) or x >= Float64(lim[1]) + 1.0:
        raise _overflow_error(dt)


def _check_convertible(v: Value, dt: DType) raises:
    """`Scalar::to<scalar_t>()`'s overflow check (c10::checked_convert and
    `c10::overflows`). A bool Scalar never overflows; an integral one into
    an unsigned type may be negative down to -max (two's complement wrap:
    -1 -> 255), into any other type it must lie in the type's range; a
    floating one follows `_check_float_convertible`."""
    if dt == DType.bool or v_scalar_is_bool(v):
        return
    if not v_scalar_is_integral(v):
        _check_float_convertible(v_f64(v), dt)
        return
    var x = v_int(v)
    if dt.is_floating_point():
        if abs(Float64(x)) > _float_limits(dt):
            raise _overflow_error(dt)
        return
    var lim = _int_limits(dt)
    if dt == DType.uint8:
        if x > lim[1] or x < -lim[1]:
            raise _overflow_error(dt)
        return
    if x < lim[0] or x > lim[1]:
        raise _overflow_error(dt)


def _is_linspace_dtype(dt: DType) -> Bool:
    return _is_arange_kernel_dtype(dt)


def _linspace_launch(
    target: T,
    params: Tuple[Float64, Float64, Float64],
    base: Float64,
    is_log: Bool,
) raises:
    var ctx = ctx_for(target.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("elementwise", "Linspace")
    call.out_dtype(target.dtype)
    call.int(target.ptr)
    call.f64(params[0])
    call.f64(params[1])
    call.f64(params[2])
    call.int(target.numel)
    call.f64(base)
    call.int(1 if is_log else 0)
    call.int(dtype_code(target.dtype))
    call.int(cp)
    call.run()
    _ = ctx


def _range_out(
    start_v: Value,
    end_v: Value,
    steps: Int,
    base: Float64,
    is_log: Bool,
    mut out_t: T,
) raises:
    var name = String("logspace") if is_log else String("linspace")
    if steps < 0:
        raise Error("number of steps must be non-negative")
    if out_t.numel != steps:
        var shape = IndexList[MAX_RANK](1)
        shape[MAX_RANK - 1] = steps
        resize_out(out_t, shape, 1)
    assert_no_internal_overlap(out_t)
    if steps == 0:
        return
    if out_t.dtype == DType.float64 and dev(out_t.device)[].api == "metal":
        unsupported(name + ".out: float64 is not supported on Apple GPU")
    if steps == 1:
        # `r.fill_(start)` / `r.fill_(pow(base, start))`: before any dtype
        # dispatch, so every dtype (bool included) takes a one-step range,
        # and fill's own conversion check applies to the value it fills.
        if is_log:
            var value = pow(base, v_f64(start_v))
            _check_float_convertible(value, out_t.dtype)
            fill_value(out_t, value)
        else:
            _check_convertible(start_v, out_t.dtype)
            fill_value(out_t, start_v.copy())
        return
    if not _is_linspace_dtype(out_t.dtype):
        if out_t.dtype == DType.bool:
            raise Error(
                '"',
                name,
                "_cuda\" not implemented for '",
                _scalar_type_name(out_t.dtype),
                "'",
            )
        unsupported(name + ".out of dtype " + String(out_t.dtype))
    _check_convertible(start_v, out_t.dtype)
    _check_convertible(end_v, out_t.dtype)
    if out_t.contig:
        _range_fill(out_t, start_v, end_v, steps, base, is_log)
    else:
        var tmp = own(new_like(out_t))
        _range_fill(tmp.t, start_v, end_v, steps, base, is_log)
        copy_strided_into(out_t, tmp.t)
        _ = tmp^  # alive past the launch


def _range_fill(
    target: T,
    start_v: Value,
    end_v: Value,
    steps: Int,
    base: Float64,
    is_log: Bool,
) raises:
    """`target` (contiguous, `steps` >= 2 elements) = the range."""
    var b = base
    if target.dtype == DType.float32 or target.dtype.is_integral():
        b = base.cast[DType.float32]().cast[DType.float64]()
    elif target.dtype == DType.float16:
        b = _round_to[DType.float16](base.cast[DType.float32]())
    elif target.dtype == DType.bfloat16:
        b = _round_to[DType.bfloat16](base.cast[DType.float32]())
    _linspace_launch(
        target,
        _linspace_params(target.dtype, start_v, end_v, steps, is_log),
        b,
        is_log,
    )


# aten::linspace.out(Scalar start, Scalar end, int steps, *, Tensor(a!) out) -> Tensor(a!)
def op_linspace_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var out_t = v_tensor(args[unsafe_offset=3])
    _range_out(
        args[unsafe_offset=0].copy(),
        args[unsafe_offset=1].copy(),
        v_int(args[unsafe_offset=2]),
        10.0,
        False,
        out_t,
    )
    ret_ref(rets, 0, out_t)


# aten::logspace.out(Scalar start, Scalar end, int steps, float base=10.0, *, Tensor(a!) out) -> Tensor(a!)
def op_logspace_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var out_t = v_tensor(args[unsafe_offset=4])
    _range_out(
        args[unsafe_offset=0].copy(),
        args[unsafe_offset=1].copy(),
        v_int(args[unsafe_offset=2]),
        v_f64(args[unsafe_offset=3]),
        True,
        out_t,
    )
    ret_ref(rets, 0, out_t)


# ---------------------------------------------------------------------------
# tril_indices / triu_indices -- ATen's `tril_indices_cuda` /
# `triu_indices_cuda` produce the (row, col) coordinates of the lower /
# upper triangle in row-major order as a (2, N) tensor. N is known from the
# arguments alone; the coordinates are written by a host loop into pinned
# staging and uploaded once (an O(N) copy of what is usually a small
# tensor, and no device work to launch).
# ---------------------------------------------------------------------------


def _tri_indices(args: Values, rets: Values, upper: Bool) raises:
    var row = v_int(args[unsafe_offset=0])
    var col = v_int(args[unsafe_offset=1])
    var offset = v_int(args[unsafe_offset=2])
    var stype = v_dtype_or(args[unsafe_offset=3], ST_INT64)
    var device = _resolve_device(args[unsafe_offset=5])
    if row < 0:
        raise Error("row must be non-negative, got", row)
    if col < 0:
        raise Error("col must be non-negative, got", col)
    if stype != ST_INT64 and stype != ST_INT32:
        unsupported(
            ("triu" if upper else "tril")
            + "_indices: dtype must be int32 or int64"
        )
    # Row-major walk: row i holds columns [lo(i), hi(i)).
    var total = 0
    for i in range(row):
        var lo = max(0, i + offset) if upper else 0
        var hi = col if upper else min(col, i + offset + 1)
        total += max(hi - lo, 0)
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 2] = 2
    shape[MAX_RANK - 1] = total
    var out = own(new_tensor(shape, 2, stype, device))
    if total > 0:
        var host = own(cpu_empty(shape, 2, stype))
        var k = 0
        for i in range(row):
            var lo = max(0, i + offset) if upper else 0
            var hi = col if upper else min(col, i + offset + 1)
            for j in range(lo, hi):
                if stype == ST_INT64:
                    var p = Pointer[Int64, MutUntrackedOrigin](
                        unsafe_from_address=host.t.ptr
                    )
                    p[unsafe_offset=k] = Int64(i)
                    p[unsafe_offset=total + k] = Int64(j)
                else:
                    var p = Pointer[Int32, MutUntrackedOrigin](
                        unsafe_from_address=host.t.ptr
                    )
                    p[unsafe_offset=k] = Int32(i)
                    p[unsafe_offset=total + k] = Int32(j)
                k += 1
        _copy_cpu_into(out.t, host.t)
        _ = host^  # alive past the upload
    ret_owned(rets, 0, out)


# aten::tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
def op_tril_indices(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _tri_indices(args, rets, False)


# aten::triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
def op_triu_indices(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _tri_indices(args, rets, True)


def register_factories(site: Site) raises:
    impl[op_arange_start_out, "arange.start_out"](site)
    impl[op_eye_out, "eye.out"](site)
    impl[op_eye_m_out, "eye.m_out"](site)
    impl[op_linspace_out, "linspace.out"](site)
    impl[op_logspace_out, "logspace.out"](site)
    impl[op_tril_indices, "tril_indices"](site)
    impl[op_triu_indices, "triu_indices"](site)
