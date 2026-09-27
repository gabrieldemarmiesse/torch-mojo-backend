# ===----------------------------------------------------------------------=== #
# Pointwise kernels with up to three operands and four scalar parameters:
# the binary/ternary math ops and the parameterized activations and their
# backwards of tmb/ops/pointwise.mojo. The math of every kind is
# `tmb.kernels.common.pointwise_math.pointwise`, which the torch.compile
# custom ops share.
#
# The ops layer hands this family flat operands only: each is either a dense
# buffer of the output's extent ("full"), one element on the device (a 0-d
# tensor, read once per call and splatted) or a host scalar (its value in a
# slot, converted to the operand dtype here on the host). Broadcasting beyond
# a single element and strided layouts are materialized there first, so one
# body serves every shape and the launch is the shared gpu_elementwise
# launcher (16-byte vectors when every full operand is aligned).
#
# Slots: out, a, b, c (addresses, 0 for an unused/immediate operand), modes
# (2 bits per operand: 0 full, 1 device element, 2 host value), the three
# host values (f64 bits), numel, p0..p3 (f64 bits), DeviceContext.
# Defines: OP (the kind), DTYPE_ARG_0 (the operands' dtype), DTYPE_OUT.
# ===----------------------------------------------------------------------=== #

from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.sys.info import (
    has_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    size_of,
)
from std.utils.coord import Coord

from tmb.kernels.common.gpu_elementwise import elementwise
from tmb.kernels.common.op_utils import (
    Argv,
    _make_ptr,
    _raw_ctx,
    _raw_f64,
    _raw_int,
)
from tmb.kernels.common.pointwise_math import (
    is_polynomial,
    metal_rounds_params,
    param_dtype,
    pointwise,
)
from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_abi_on,
    _dtype_out_on,
    _op_on,
    _tmb_entry_error,
)

# NVIDIA-only launch choices (the only GPU they were measured on): the
# element-aligned vector route for misaligned operands and `_wide_f32`.
comptime _ua_ok = has_nvidia_gpu_accelerator()

comptime MODE_FULL = 0
comptime MODE_DEVICE = 1
comptime MODE_HOST = 2

comptime OPERAND_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
]
comptime OUT_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.bool,
]

# Every kind this family builds, with its operand count.
comptime KINDS_1 = [
    "elu",
    "frexp_exponent",
    "frexp_mantissa",
    "hardshrink",
    "hardsigmoid",
    "hardswish",
    "hardtanh",
    "leaky_relu",
    "log_sigmoid",
    "mish",
    "pow_scalar_base",
    "pow_tensor_scalar",
    "rsub_alpha_scalar",
    "scale",
    "softplus",
    "softshrink",
    "threshold",
]
comptime KINDS_2 = [
    "atan2",
    "chebyshev_polynomial_t",
    "chebyshev_polynomial_u",
    "chebyshev_polynomial_v",
    "chebyshev_polynomial_w",
    "copysign",
    "elu_backward",
    "fmax",
    "fmin",
    "fmod",
    "gcd",
    "gelu_backward_none",
    "gelu_backward_tanh",
    "hardsigmoid_backward",
    "hardswish_backward",
    "hardtanh_backward",
    "heaviside",
    "hermite_polynomial_h",
    "hermite_polynomial_he",
    "hypot",
    "igamma",
    "igammac",
    "ipow",
    "laguerre_polynomial_l",
    "lcm",
    "ldexp",
    "ldexp_pow2",
    "leaky_relu_backward",
    "legendre_polynomial_p",
    "lerp_scalar",
    "log_sigmoid_backward",
    "logaddexp",
    "logaddexp2",
    "logit_backward",
    "lshift",
    "huber",
    "maximum",
    "minimum",
    "mish_backward",
    "mse",
    "mul_scale",
    "nextafter",
    "rrelu_noise",
    "rrelu_train",
    "rshift",
    "rsub_alpha",
    "shifted_chebyshev_polynomial_t",
    "shifted_chebyshev_polynomial_u",
    "shifted_chebyshev_polynomial_v",
    "shifted_chebyshev_polynomial_w",
    "shrink_backward",
    "silu_backward",
    "smooth_l1",
    "softplus_backward",
    "xlog1py",
    "xlogy",
    "zeta",
]
comptime KINDS_3 = [
    "bce",
    "bce_backward",
    "bce_logits",
    "clamp",
    "huber_backward",
    "lerp",
    "mse_backward",
    "smooth_l1_backward",
]


@always_inline
def _heavy_kind[kind: StaticString]() -> Bool:
    """Bodies expensive enough for gpu_elementwise's small-grid block choice
    (`_policy_block`): transcendental, dividing or looping math. The cheap
    selects and scalings (hardshrink, leaky_relu, deg2rad, copysign, ...)
    measured slower with it."""
    return (
        kind == "elu"
        or kind == "softplus"
        or kind == "mish"
        or kind == "log_sigmoid"
        or kind == "elu_backward"
        or kind == "mish_backward"
        or kind == "silu_backward"
        or kind == "softplus_backward"
        or kind == "log_sigmoid_backward"
        or kind == "gelu_backward_none"
        or kind == "gelu_backward_tanh"
        or kind == "logit_backward"
        or kind == "atan2"
        or kind == "hypot"
        or kind == "logaddexp"
        or kind == "logaddexp2"
        or kind == "xlogy"
        or kind == "xlog1py"
        or kind == "igamma"
        or kind == "igammac"
        or kind == "zeta"
        or kind == "bce"
        or kind == "bce_backward"
        or kind == "bce_logits"
        or kind == "fmod"
        or kind == "gcd"
        or kind == "lcm"
        or kind == "pow_scalar_base"
        or kind == "pow_tensor_scalar"
        or kind == "ldexp_pow2"
        or kind == "nextafter"
        or kind == "fmax"
        or kind == "fmin"
        or kind == "lerp"
        or kind == "lerp_scalar"
        or is_polynomial[kind]()
    )


@always_inline
def _lanes[kind: StaticString, dt: DType]() -> Int:
    """Lanes per thread of the aligned launch: 16 bytes, except igamma /
    igammac, whose per-element loops (series, continued fraction) run one
    element per lane on NVIDIA: H100 PCIe, ncu base clocks, 16M float32,
    coherent (sorted) inputs igamma 898 -> 848 us (torch 762), igammac 922
    -> 863 (744); random inputs 2.20 ms (2.27), 2.71 (2.57); 281673 sorted
    elements igammac 23 us (55)."""
    comptime if _ua_ok and (kind == "igamma" or kind == "igammac"):
        return 1
    else:
        return 16 // size_of[dt]()


# Below this many elements the 4-lane launch wins even for `_wide_f32`
# kinds (twice the threads on a GPU the grid does not fill: mish f32 281673
# elements 6.4 us at 4 lanes, 7.3 at 8, torch 6.5). H100 PCIe, ncu base
# clocks.
comptime _WIDE_MIN_NUMEL = 1 << 21


@always_inline
def _wide_f32[kind: StaticString]() -> Bool:
    """float32 kinds launched 8 lanes per thread (two 16-byte vectors)
    rather than 4: bodies heavy enough that two independent chains per
    thread beat twice the threads. Measured with ncu on H100 PCIe at base
    clocks, 16M elements, against stock torch's vec4 kernel: mish 85 -> 72
    us (torch 70), softplus 83 -> 75 (72), log_sigmoid 74 -> 70 (69),
    mish_backward 116 -> 103 (103), xlog1py 73 -> 70 (68). elu, logaddexp,
    xlogy and the HBM-bound backwards measured the same or slower at 8."""
    return (
        kind == "mish"
        or kind == "softplus"
        or kind == "log_sigmoid"
        or kind == "mish_backward"
        or kind == "xlog1py"
    )


@always_inline
def _host_value[dt: DType](v: Float64) -> Scalar[dt]:
    """A host scalar in the operand dtype. An integer or bool operand's slot
    holds the int64's bits (the ops layer's `_pw_flat`; a bool travels as
    the byte type here), narrowed with two's complement wrap as torch
    narrows a wrapped number."""
    comptime if dt.is_floating_point():
        return v.cast[dt]()
    elif dt.is_integral():
        return bitcast[DType.int64](v).cast[dt]()
    else:
        return Scalar[dt](Int(v))


@always_inline
def _launch[
    kind: StaticString,
    dt: DType,
    odt: DType,
    arity: Int,
    full_a: Bool,
    full_b: Bool,
    full_c: Bool,
    width: Int,
    ua: Bool = False,
](
    out_addr: Int,
    a_addr: Int,
    b_addr: Int,
    c_addr: Int,
    dev_a: Bool,
    dev_b: Bool,
    dev_c: Bool,
    va: Scalar[dt],
    vb: Scalar[dt],
    vc: Scalar[dt],
    numel: Int,
    params: SIMD[param_dtype[dt](), 4],
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[odt](out_addr)
    var a_ptr = _make_ptr[dt](a_addr)
    var b_ptr = _make_ptr[dt](b_addr)
    var c_ptr = _make_ptr[dt](c_addr)
    # The parameters cross the launch as four scalars: a by-value vector
    # kernel argument that the body reads makes Metal's pipeline compiler
    # fail ("Internal compiler error"); scalars travel in constant buffers.
    var p0 = params[0]
    var p1 = params[1]
    var p2 = params[2]
    var p3 = params[3]

    @always_inline
    @__parameter
    @__copy_capture(
        out_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        dev_a,
        dev_b,
        dev_c,
        va,
        vb,
        vc,
        p0,
        p1,
        p2,
        p3,
    )
    def func[w: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        # `ua`: full operands off the vector grid, loaded element-aligned.
        comptime in_align = size_of[dt]() if ua else min(16, w * size_of[dt]())
        comptime out_align = min(16, w * size_of[odt]())
        var a: SIMD[dt, w]
        comptime if full_a:
            a = a_ptr.unsafe_load[width=w, alignment=in_align](i)
        else:
            a = SIMD[dt, w](a_ptr[unsafe_offset=0] if dev_a else va)
        var b = SIMD[dt, w](0)
        comptime if arity >= 2:
            comptime if full_b:
                b = b_ptr.unsafe_load[width=w, alignment=in_align](i)
            else:
                b = SIMD[dt, w](b_ptr[unsafe_offset=0] if dev_b else vb)
        var c = SIMD[dt, w](0)
        comptime if arity >= 3:
            comptime if full_c:
                c = c_ptr.unsafe_load[width=w, alignment=in_align](i)
            else:
                c = SIMD[dt, w](c_ptr[unsafe_offset=0] if dev_c else vc)
        out_ptr.unsafe_store[width=w, alignment=out_align](
            i,
            pointwise[kind, dt, odt, w](
                a, b, c, SIMD[param_dtype[dt](), 4](p0, p1, p2, p3)
            ),
        )

    elementwise[
        func,
        simd_width=width,
        target="gpu",
        _trace_description=kind,
        _heavy=_heavy_kind[kind](),
    ](Coord(numel), ctx)


@always_inline
def _by_mask[
    kind: StaticString,
    dt: DType,
    odt: DType,
    arity: Int,
    width: Int,
    ua: Bool = False,
](
    mask: Int,
    out_addr: Int,
    a_addr: Int,
    b_addr: Int,
    c_addr: Int,
    dev_a: Bool,
    dev_b: Bool,
    dev_c: Bool,
    va: Scalar[dt],
    vb: Scalar[dt],
    vc: Scalar[dt],
    numel: Int,
    params: SIMD[param_dtype[dt](), 4],
    ctx: DeviceContext,
) raises:
    """Instantiate the body for the runtime pattern of full operands (bit i
    = operand i is full): each pattern is its own straight-line kernel."""
    comptime for m in range(1, 1 << arity):
        if mask == m:
            _launch[
                kind,
                dt,
                odt,
                arity,
                (m & 1) != 0,
                (m & 2) != 0,
                (m & 4) != 0,
                width,
                ua,
            ](
                out_addr,
                a_addr,
                b_addr,
                c_addr,
                dev_a,
                dev_b,
                dev_c,
                va,
                vb,
                vc,
                numel,
                params,
                ctx,
            )
            return
    raise Error("pointwise: no full operand (mask ", mask, ")")


def _pointwise_go[kind: StaticString, arity: Int](argv: Argv, argc: Int) raises:
    if argc != 14:
        raise Error("pointwise expects 14 arguments, got ", argc)
    var out_addr = _raw_int(argv[unsafe_offset=0])
    var a_addr = _raw_int(argv[unsafe_offset=1])
    var b_addr = _raw_int(argv[unsafe_offset=2])
    var c_addr = _raw_int(argv[unsafe_offset=3])
    var modes = _raw_int(argv[unsafe_offset=4])
    var ha = _raw_f64(argv[unsafe_offset=5])
    var hb = _raw_f64(argv[unsafe_offset=6])
    var hc = _raw_f64(argv[unsafe_offset=7])
    var numel = _raw_int(argv[unsafe_offset=8])
    var p0 = _raw_f64(argv[unsafe_offset=9])
    var p1 = _raw_f64(argv[unsafe_offset=10])
    var p2 = _raw_f64(argv[unsafe_offset=11])
    var p3 = _raw_f64(argv[unsafe_offset=12])
    var ctx = _raw_ctx(argv[unsafe_offset=13])
    if numel <= 0:
        return
    var mode_a = modes & 3
    var mode_b = (modes >> 2) & 3
    var mode_c = (modes >> 4) & 3
    var mask = (
        (1 if mode_a == MODE_FULL else 0)
        | ((2 if mode_b == MODE_FULL else 0) if arity >= 2 else 0)
        | ((4 if mode_c == MODE_FULL else 0) if arity >= 3 else 0)
    )

    var handled = False
    comptime for dt in OPERAND_DTYPES:
        comptime if _dtype_arg_abi_on[0, dt]():
            comptime for odt0 in OUT_DTYPES:
                comptime if _dtype_out_on[0, odt0]():
                    # torch's bool storage is one 0/1 byte: bool results are
                    # computed and stored as uint8.
                    comptime odt = DType.uint8 if odt0 == DType.bool else odt0
                    comptime pdt = param_dtype[dt]()
                    comptime if (
                        (dt == DType.float64 or odt == DType.float64)
                        and has_apple_gpu_accelerator()
                    ):
                        raise Error("float64 is not supported on Apple GPU")
                    elif not has_accelerator():
                        raise Error("no GPU accelerator available")
                    else:
                        var pd = SIMD[DType.float64, 4](p0, p1, p2, p3)
                        comptime if (
                            has_apple_gpu_accelerator()
                            and metal_rounds_params[kind]()
                            and dt.is_floating_point()
                        ):
                            # MPS's kernels take these as scalar_t.
                            pd = pd.cast[dt]().cast[DType.float64]()
                        var params = pd.cast[pdt]()
                        comptime if dt.is_integral():
                            # Integer parameters travel as the int64's bits
                            # (`_int_exact_param`): exact past 2**53.
                            params = rebind[SIMD[pdt, 4]](
                                bitcast[DType.int64, 4](pd)
                            )
                        var va = _host_value[dt](ha)
                        var vb = _host_value[dt](hb)
                        var vc = _host_value[dt](hc)
                        comptime vw = _lanes[kind, dt]()
                        comptime wide = (
                            _ua_ok and dt == DType.float32 and _wide_f32[kind]()
                        )
                        comptime out_align = min(16, vw * size_of[odt]())
                        var aligned = out_addr % out_align == 0
                        if mask & 1:
                            aligned = aligned and a_addr % 16 == 0
                        if mask & 2:
                            aligned = aligned and b_addr % 16 == 0
                        if mask & 4:
                            aligned = aligned and c_addr % 16 == 0
                        if aligned:
                            comptime if wide:
                                if numel >= _WIDE_MIN_NUMEL:
                                    _by_mask[kind, dt, odt, arity, 8](
                                        mask,
                                        out_addr,
                                        a_addr,
                                        b_addr,
                                        c_addr,
                                        mode_a == MODE_DEVICE,
                                        mode_b == MODE_DEVICE,
                                        mode_c == MODE_DEVICE,
                                        va,
                                        vb,
                                        vc,
                                        numel,
                                        params,
                                        ctx,
                                    )
                                else:
                                    _by_mask[kind, dt, odt, arity, vw](
                                        mask,
                                        out_addr,
                                        a_addr,
                                        b_addr,
                                        c_addr,
                                        mode_a == MODE_DEVICE,
                                        mode_b == MODE_DEVICE,
                                        mode_c == MODE_DEVICE,
                                        va,
                                        vb,
                                        vc,
                                        numel,
                                        params,
                                        ctx,
                                    )
                            else:
                                _by_mask[kind, dt, odt, arity, vw](
                                    mask,
                                    out_addr,
                                    a_addr,
                                    b_addr,
                                    c_addr,
                                    mode_a == MODE_DEVICE,
                                    mode_b == MODE_DEVICE,
                                    mode_c == MODE_DEVICE,
                                    va,
                                    vb,
                                    vc,
                                    numel,
                                    params,
                                    ctx,
                                )
                        else:
                            # A full operand off the vector grid (a view at
                            # an odd element): element-aligned loads, still
                            # 16-byte lanes and aligned stores, on NVIDIA
                            # (measured on H100: offset_1 deg2rad f16 16M
                            # ran 1.34x torch at width 1).
                            comptime if _ua_ok:
                                if out_addr % out_align == 0:
                                    _by_mask[kind, dt, odt, arity, vw, True](
                                        mask,
                                        out_addr,
                                        a_addr,
                                        b_addr,
                                        c_addr,
                                        mode_a == MODE_DEVICE,
                                        mode_b == MODE_DEVICE,
                                        mode_c == MODE_DEVICE,
                                        va,
                                        vb,
                                        vc,
                                        numel,
                                        params,
                                        ctx,
                                    )
                                else:
                                    _by_mask[kind, dt, odt, arity, 1](
                                        mask,
                                        out_addr,
                                        a_addr,
                                        b_addr,
                                        c_addr,
                                        mode_a == MODE_DEVICE,
                                        mode_b == MODE_DEVICE,
                                        mode_c == MODE_DEVICE,
                                        va,
                                        vb,
                                        vc,
                                        numel,
                                        params,
                                        ctx,
                                    )
                            else:
                                _by_mask[kind, dt, odt, arity, 1](
                                    mask,
                                    out_addr,
                                    a_addr,
                                    b_addr,
                                    c_addr,
                                    mode_a == MODE_DEVICE,
                                    mode_b == MODE_DEVICE,
                                    mode_c == MODE_DEVICE,
                                    va,
                                    vb,
                                    vc,
                                    numel,
                                    params,
                                    ctx,
                                )
                        handled = True
    if not handled:
        raise Error("pointwise ", kind, ": dtype pair not compiled in")


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kind and one dtype pair per build.
    Slots are described in the header; errors come back as (rc=1, message).
    """
    try:
        comptime for kind in KINDS_1:
            comptime if _op_on[kind]():
                _pointwise_go[kind, 1](argv, argc)
                return 0
        comptime for kind in KINDS_2:
            comptime if _op_on[kind]():
                _pointwise_go[kind, 2](argv, argc)
                return 0
        comptime for kind in KINDS_3:
            comptime if _op_on[kind]():
                _pointwise_go[kind, 3](argv, argc)
                return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
