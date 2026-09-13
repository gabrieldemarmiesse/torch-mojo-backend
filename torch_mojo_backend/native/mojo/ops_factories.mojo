"""aten ops: factories group (see docs/native_backend.md).

Most of this group's aten surface -- full/zeros/ones/new_*/scalar_tensor/
empty_like/*_like -- is CompositeExplicitAutograd upstream (verified against
the installed torch build): it decomposes into `empty.memory_format` +
`fill_.Scalar`, both already registered by ops_core.mojo, so a fused
registration here would run the identical two calls and buy nothing. See the
final report for the exact list skipped on that basis.

What actually needs a kernel here (no PrivateUse1 fallback exists upstream):
`arange.start_out` (arange/arange.start/arange.start_step are themselves
CompositeExplicitAutograd and call this through `empty` + `arange_out`, so
registering only the `out` overload is enough), `uniform_` and `normal_`
(RNG), and `native_dropout(_backward)` (float32 GPU only, as in the old
eager path -- `_on_gpu`/dtype gate ported verbatim).
"""
from std.math import ceil
from std.utils import IndexList

from abi import (
    Owned,
    T,
    Value,
    Values,
    ST_BOOL,
    contiguous_strides,
    cpu_empty,
    dtype_code,
    f64_bits,
    new_like,
    new_tensor,
    own,
    release,
    ret_owned,
    ret_ref,
    set_sizes_strides,
    unsupported,
    v_bool_or,
    v_f64,
    v_f64_or,
    v_generator,
    v_int,
    v_scalar_is_integral,
    v_tensor,
    TAG_DOUBLE,
    TAG_NONE,
    TAG_TENSOR,
    ST_FLOAT32,
    retain,
)
from device import copy_d2d, copy_from_host, ctx_for, ctx_ptr, dev
from kernels import KernelCall
from op_utils import MAX_RANK
from ops_common import (
    call_op,
    cast_into,
    cast_to,
    contiguous,
    copy_strided_into,
    fill_value,
    philox_reserve,
    resize_out,
)
from registry import Site, impl, op_address_of


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
    ret_ref(rets, 0, out_t)


# ---------------------------------------------------------------------------
# uniform_ -- on-device Philox draw (aten_fast.py fast_aten_uniform_).
# ---------------------------------------------------------------------------


def _uniform_fill(
    t: T, from_: Float64, to: Float64, seed: UInt64, offset: UInt64
) raises:
    var ctx = ctx_for(t.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("random_ops", "UniformFill")
    call.out_dtype(t.dtype)
    call.int(t.ptr)
    call.f64(from_)
    call.f64(to)
    call.int(t.numel)
    call.int(dtype_code(t.dtype))
    call.int(Int(seed & 0xFFFFFFFF))
    call.int(Int((seed >> 32) & 0xFFFFFFFF))
    call.int(Int(offset & 0xFFFFFFFF))
    call.int(Int((offset >> 32) & 0xFFFFFFFF))
    call.int(cp)
    call.run()
    _ = ctx


# aten::uniform_(Tensor(a!) self, float from=0., float to=1., *, Generator? generator=None) -> Tensor(a!)
def op_uniform_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var from_ = v_f64_or(args[unsafe_offset=1], 0.0)
    var to = v_f64_or(args[unsafe_offset=2], 1.0)
    # ATen dispatches uniform_impl_ over the floating types plus the two
    # 16-bit floats (aten_fast.py `_UNIFORM_DTYPES`).
    if (
        t.dtype != DType.float32
        and t.dtype != DType.float16
        and t.dtype != DType.bfloat16
        and t.dtype != DType.float64
    ):
        unsupported("uniform_ of dtype " + String(t.dtype))
    if t.dtype == DType.float64 and dev(t.device)[].api == "metal":
        unsupported("uniform_ of dtype float64 on Apple GPU")
    var lowest: Float64
    var highest: Float64
    if t.dtype == DType.float16:
        lowest = -65504.0
        highest = 65504.0
    elif t.dtype == DType.bfloat16:
        lowest = -3.3895313892515355e38
        highest = 3.3895313892515355e38
    elif t.dtype == DType.float64:
        lowest = -1.7976931348623157e308
        highest = 1.7976931348623157e308
    else:
        lowest = -3.4028234663852886e38
        highest = 3.4028234663852886e38
    # ATen's own checks (check_uniform_bounds), in ATen's order.
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
    # Unlike the old eager path (whose docstring explains that
    # `torch.Generator(device="mojo")` could not even be constructed from
    # Python), the native backend's PrivateUse1HooksInterface implements
    # `getNewGenerator`, so an explicit generator is a real, independently
    # seeded `MojoGeneratorImpl` -- `tmb_philox_reserve` accepts its handle
    # directly (0 still means "this device's default generator") and the
    # shim raises a clean error if it belongs to a non-mojo device.
    var generator = v_generator(args[unsafe_offset=3])
    if t.numel == 0:
        ret_ref(rets, 0, t)
        return
    # float64 needs two of Philox's four 32-bit words per element; every
    # other dtype takes one (uniform_kernels.mojo).
    var words = 2 if t.dtype == DType.float64 else 1
    var increment = (t.numel * words + 3) // 4
    var seed_offset = philox_reserve(generator, t.device, increment)
    if t.contig:
        _uniform_fill(t, from_, to, seed_offset[0], seed_offset[1])
    else:
        var tmp = own(new_like(t))
        _uniform_fill(tmp.t, from_, to, seed_offset[0], seed_offset[1])
        copy_strided_into(t, tmp.t)
    ret_ref(rets, 0, t)


# ---------------------------------------------------------------------------
# normal_ -- host draw through the real CPU generator (aten_ops/rng.py):
# the old path never used the device's Philox stream for this op, only
# torch's own default CPU generator, and always declined an explicit one.
# ---------------------------------------------------------------------------


# aten::normal_(Tensor(a!) self, float mean=0., float std=1., *, Generator? generator=None) -> Tensor(a!)
def op_normal_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var t = v_tensor(args[unsafe_offset=0])
    var mean = v_f64_or(args[unsafe_offset=1], 0.0)
    var std = v_f64_or(args[unsafe_offset=2], 1.0)
    if v_generator(args[unsafe_offset=3]) != 0:
        # A mojo generator's state (MojoGeneratorImpl, real as of this
        # backend) is Philox seed/offset, not the CPU Mersenne-Twister state
        # `aten::normal_`'s host kernel below expects -- it cannot be
        # forwarded, so only the host's own default CPU generator is used
        # (matches the old aten_ops/rng.py path exactly).
        unsupported(
            "aten::normal_ on the mojo device draws from the host's default"
            " CPU generator; an explicit generator= is not supported"
        )
    var cpu = own(cpu_empty(t.shape, t.rank, t.stype))
    var call_args = List[Value](capacity=4)
    call_args.append(Value(TAG_TENSOR, 0, Int64(cpu.t.h), 0))
    call_args.append(Value(TAG_DOUBLE, 0, f64_bits(mean), 0))
    call_args.append(Value(TAG_DOUBLE, 0, f64_bits(std), 0))
    call_args.append(Value(TAG_NONE, 0, 0, 0))
    _ = call_op(
        "aten::normal_", "", call_args^, 1
    )  # Results releases its handle
    _copy_cpu_into(t, cpu.t)
    ret_ref(rets, 0, t)


# aten::random_.from(Tensor(a!) self, int from, int? to, *, Generator? generator=None) -> Tensor(a!)
# aten::random_.to(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)
# aten::random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)
def _random_host(
    overload: StaticString, args: Values, rets: Values, n_scalars: Int
) raises:
    """torch.randint / random_ on the device: drawn by the host kernel on a
    CPU scratch tensor (the host's default CPU generator, like normal_),
    then copied over; `n_scalars` integer arguments follow `self`."""
    var t = v_tensor(args[unsafe_offset=0])
    if v_generator(args[unsafe_offset=1 + n_scalars]) != 0:
        unsupported(
            "aten::random_ on the mojo device draws from the host's default"
            " CPU generator; an explicit generator= is not supported"
        )
    var cpu = own(cpu_empty(t.shape, t.rank, t.stype))
    var call_args = List[Value](capacity=2 + n_scalars)
    call_args.append(Value(TAG_TENSOR, 0, Int64(cpu.t.h), 0))
    for i in range(n_scalars):
        call_args.append(args[unsafe_offset=1 + i].copy())
    call_args.append(Value(TAG_NONE, 0, 0, 0))
    # `Results` releases tmb_call_op's own fresh wrapper handle.
    _ = call_op("aten::random_", String(overload), call_args^, 1)
    _copy_cpu_into(t, cpu.t)
    ret_ref(rets, 0, t)


def op_random_from(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _random_host("from", args, rets, 2)


def op_random_to(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _random_host("to", args, rets, 1)


def op_random_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _random_host("", args, rets, 0)


# ---------------------------------------------------------------------------
# native_dropout / native_dropout_backward -- float32 GPU only, matching the
# old eager path exactly (dropout_ops family: NativeDropoutF32,
# NativeDropoutBackwardF32).
# ---------------------------------------------------------------------------


def _bool_mask(a: T, value: Bool) raises -> Owned:
    var m = own(new_tensor(a.shape, a.rank, ST_BOOL, a.device))
    fill_value(m.t, 1.0 if value else 0.0)
    return m^


def _native_dropout_fill(
    output: T, mask: T, input: T, p: Float64, seed: UInt64, offset: UInt64
) raises:
    var ctx = ctx_for(output.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("dropout_ops", "NativeDropoutF32")
    call.arg_dtype(0, input.dtype)
    call.out_dtype_i(0, output.dtype)
    call.out_dtype_i(1, mask.dtype)
    call.int(output.ptr)
    call.int(mask.ptr)
    call.int(input.ptr)
    call.int(input.numel)
    call.f64(p)
    call.int(Int(seed & 0xFFFFFFFF))
    call.int(Int((seed >> 32) & 0xFFFFFFFF))
    call.int(Int(offset & 0xFFFFFFFF))
    call.int(Int((offset >> 32) & 0xFFFFFFFF))
    call.int(cp)
    call.run()
    _ = ctx


def _dropout_dtype_ok(dtype: DType) -> Bool:
    return (
        dtype == DType.float32
        or dtype == DType.float16
        or dtype == DType.bfloat16
    )


# aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)
def op_native_dropout(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    if not _dropout_dtype_ok(a.dtype) or dev(a.device)[].is_cpu:
        unsupported(
            "native_dropout is only implemented for float32/float16/bfloat16"
            " tensors on the mojo GPU device"
        )
    # `train=None` behaves like `train=True` (matches the old eager path);
    # only an explicit False takes the inference shortcut.
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
        # ATen's GPU endpoint shortcut: no division, no generator state
        # consumed.
        var output = own(new_like(a))
        fill_value(output.t, 0.0)
        var mask = _bool_mask(a, False)
        ret_owned(rets, 0, output)
        ret_owned(rets, 1, mask)
        return
    # Validate and allocate before touching generator state: nothing after
    # the Philox reservation may read the host or synchronize the device.
    # Half-precision inputs run the float32 kernel on a float32 copy and
    # cast the output back (the mask is dtype-free).
    var ac = contiguous(a)
    var a32 = cast_to(ac, ST_FLOAT32)
    var output = own(new_like(a))
    var out32 = own(new_like(a32)) if a.dtype != DType.float32 else own(
        T(retain(output.t))
    )
    var mask = own(new_tensor(a.shape, a.rank, ST_BOOL, a.device))
    var increment = (a.numel + 3) // 4
    var seed_offset = philox_reserve(0, a.device, increment)
    _native_dropout_fill(
        out32.t, mask.t, a32, p, seed_offset[0], seed_offset[1]
    )
    if a32.h != ac.h:
        release(a32.h)
    if ac.h != a.h:
        release(ac.h)
    if out32.t.h != output.t.h:
        cast_into(output.t, out32.t)
    _ = out32^
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
        or dev(grad.device)[].is_cpu
        or keep.device != grad.device
        or not grad.same_shape(keep)
    ):
        unsupported(
            "native_dropout_backward is only implemented for a"
            " float32/float16/bfloat16 grad_output and a bool mask on the"
            " same mojo GPU device"
        )
    var grad_input = own(new_like(grad))
    if grad.numel > 0:
        var gc = contiguous(grad)
        var g32 = cast_to(gc, ST_FLOAT32)
        var kc = contiguous(keep)
        var gi32 = own(new_like(g32)) if grad.dtype != DType.float32 else own(
            T(retain(grad_input.t))
        )
        _native_dropout_backward_fill(gi32.t, g32, kc, scale)
        if g32.h != gc.h:
            release(g32.h)
        if gc.h != grad.h:
            release(gc.h)
        if kc.h != keep.h:
            release(kc.h)
        if gi32.t.h != grad_input.t.h:
            cast_into(grad_input.t, gi32.t)
        _ = gi32^
    ret_owned(rets, 0, grad_input)


def _native_dropout_backward_fill(
    grad_input: T, grad: T, mask: T, scale: Float64
) raises:
    var ctx = ctx_for(grad_input.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("dropout_ops", "NativeDropoutBackwardF32")
    call.arg_dtype(0, grad.dtype)
    call.arg_dtype(1, mask.dtype)
    call.out_dtype(grad_input.dtype)
    call.int(grad_input.ptr)
    call.int(grad.ptr)
    call.int(mask.ptr)
    call.int(grad.numel)
    call.f64(scale)
    call.int(cp)
    call.run()
    _ = ctx


def register_factories(site: Site) raises:
    impl[op_random_from, "random_.from"](site)
    impl[op_random_to, "random_.to"](site)
    impl[op_random_, "random_"](site)
    impl[op_arange_start_out, "arange.start_out"](site)
    impl[op_uniform_, "uniform_"](site)
    impl[op_normal_, "normal_"](site)
    impl[op_native_dropout, "native_dropout"](site)
    impl[op_native_dropout_backward, "native_dropout_backward"](site)


@export
def tmb_op_address() abi("C") -> Int:
    """Entry of this file's one-op extension: the address of the op the
    TMB_OP define selected (registry.mojo)."""
    return op_address_of[register_factories]()
