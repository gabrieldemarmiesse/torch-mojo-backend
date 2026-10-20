# ===----------------------------------------------------------------------=== #
# Complex kernels of tmb/ops/pointwise.mojo's ternary ops: lerp (Scalar and
# Tensor weight), addcmul, addcdiv. Stock CUDA jiterates them (Lerp.cu,
# PointwiseOpsKernel.cu): libc++'s <complex> (llvm_complex.cpp) under NVRTC,
# so the products are `cmul_jit`, the quotients `cdiv_jit`.
#
# Operands as in the complex family (tmb/kernels/complex): each full (a dense
# buffer of the output's extent), one device element or a host value, the
# interleaved component storage of a complex64 / complex128 tensor.
#
# Slots: out, a, b, c (addresses), modes (2 bits per operand: 0 full,
# 1 device element, 2 host value), a_re, a_im, b_re, b_im, c_re, c_im (f64
# bits), numel (complex elements), p_re, p_im (f64 bits: the weight or the
# value), DeviceContext.
# Defines: OP (the kind), DTYPE_ARG_0 (the component dtype).
# ===----------------------------------------------------------------------=== #

from max.gpu.host import DeviceContext
from std.sys._assembly import inlined_assembly
from std.sys.info import (
    has_accelerator,
    has_apple_gpu_accelerator,
    is_nvidia_gpu,
)
from std.utils.numerics import isnan
from std.utils.coord import Coord

from tmb.kernels.common.complex_math import Cx, cdiv_jit, chypot, cmul_jit
from tmb.kernels.common.gpu_elementwise import elementwise
from tmb.kernels.common.host_float import f32_bits_of
from std.memory import bitcast
from tmb.kernels.common.op_utils import (
    Argv,
    _make_ptr,
    _raw_ctx,
    _raw_f64,
    _raw_int,
)
from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)

comptime MODE_FULL = 0
comptime MODE_DEVICE = 1
comptime MODE_HOST = 2

comptime COMPONENT_DTYPES = [DType.float32, DType.float64]
comptime KINDS = ["cx_lerp_scalar", "cx_lerp", "cx_addcmul", "cx_addcdiv"]


@always_inline
def _hval[dt: DType](v: Float64) -> Scalar[dt]:
    """A host component in dt: float32 by c10's conversion (no flush)."""
    comptime if dt == DType.float32:
        return rebind[Scalar[dt]](bitcast[DType.float32](f32_bits_of(v)))
    else:
        return v.cast[dt]()


@always_inline
def _sub_rn[dt: DType](x: Scalar[dt], y: Scalar[dt]) -> Scalar[dt]:
    """x - y with no algebra (fast-math would fold 0 - y into -y, whose zero
    has the other sign)."""
    comptime if is_nvidia_gpu():
        comptime asm = "sub.rn.f32 $0, $1, $2;" if dt == DType.float32 else (
            "sub.rn.f64 $0, $1, $2;"
        )
        comptime cons = "=f,f,f" if dt == DType.float32 else "=d,d,d"
        return inlined_assembly[
            asm, Scalar[dt], constraints=cons, has_side_effect=False
        ](x, y)
    else:
        return x - y


@always_inline
def _add[dt: DType](x: Cx[dt], y: Cx[dt]) -> Cx[dt]:
    return Cx[dt](x.re + y.re, x.im + y.im)


@always_inline
def _sub[dt: DType](x: Cx[dt], y: Cx[dt]) -> Cx[dt]:
    return Cx[dt](x.re - y.re, x.im - y.im)


@always_inline
def _lerp[dt: DType](s: Cx[dt], e: Cx[dt], w: Cx[dt]) -> Cx[dt]:
    """Lerp.cu's jiterated lerp: abs(w) < 0.5 ? s + w (e - s) :
    e - (e - s)(1 - w), abs being libc++'s hypot(re, im); 1 - w's
    imaginary part is 0 - w.im (NVRTC's FADD, +0 for a zero)."""
    var diff = _sub(e, s)
    var aw = chypot(w.re, w.im)
    if not isnan(aw) and aw < Scalar[dt](0.5):
        return _add(s, cmul_jit(w, diff))
    var omw = Cx[dt](
        _sub_rn(Scalar[dt](1), w.re), _sub_rn(Scalar[dt](0), w.im)
    )
    return _sub(e, cmul_jit(diff, omw))


@always_inline
def _ternary[
    kind: StaticString, dt: DType
](a: Cx[dt], b: Cx[dt], c: Cx[dt], p: Cx[dt]) -> Cx[dt]:
    comptime if kind == "cx_lerp_scalar":
        return _lerp(a, b, p)
    elif kind == "cx_lerp":
        return _lerp(a, b, c)
    elif kind == "cx_addcmul":
        # a + alpha * (b * c)
        return _add(a, cmul_jit(p, cmul_jit(b, c)))
    else:
        comptime assert kind == "cx_addcdiv", "unknown complex ternary kind"
        # a + alpha * (b / c)
        return _add(a, cmul_jit(p, cdiv_jit(b, c)))


@always_inline
def _launch[
    kind: StaticString, dt: DType
](
    out_addr: Int,
    a_addr: Int,
    b_addr: Int,
    c_addr: Int,
    modes: Int,
    h: SIMD[DType.float64, 8],
    numel: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dt](out_addr)
    var a_ptr = _make_ptr[dt](a_addr)
    var b_ptr = _make_ptr[dt](b_addr)
    var c_ptr = _make_ptr[dt](c_addr)
    var mode_a = modes & 3
    var mode_b = (modes >> 2) & 3
    var mode_c = (modes >> 4) & 3
    var ha_re = _hval[dt](h[0])
    var ha_im = _hval[dt](h[1])
    var hb_re = _hval[dt](h[2])
    var hb_im = _hval[dt](h[3])
    var hc_re = _hval[dt](h[4])
    var hc_im = _hval[dt](h[5])
    var hp_re = _hval[dt](h[6])
    var hp_im = _hval[dt](h[7])

    @always_inline
    @__parameter
    @__copy_capture(
        out_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        mode_a,
        mode_b,
        mode_c,
        ha_re,
        ha_im,
        hb_re,
        hb_im,
        hc_re,
        hc_im,
        hp_re,
        hp_im,
    )
    def func[w: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var a = Cx[dt](ha_re, ha_im)
        if mode_a != MODE_HOST:
            var k = 2 * i if mode_a == MODE_FULL else 0
            a = Cx[dt](a_ptr[unsafe_offset=k], a_ptr[unsafe_offset=k + 1])
        var b = Cx[dt](hb_re, hb_im)
        if mode_b != MODE_HOST:
            var k = 2 * i if mode_b == MODE_FULL else 0
            b = Cx[dt](b_ptr[unsafe_offset=k], b_ptr[unsafe_offset=k + 1])
        var c = Cx[dt](hc_re, hc_im)
        if mode_c != MODE_HOST:
            var k = 2 * i if mode_c == MODE_FULL else 0
            c = Cx[dt](c_ptr[unsafe_offset=k], c_ptr[unsafe_offset=k + 1])
        var r = _ternary[kind, dt](a, b, c, Cx[dt](hp_re, hp_im))
        out_ptr[unsafe_offset=2 * i] = r.re
        out_ptr[unsafe_offset=2 * i + 1] = r.im

    elementwise[
        func,
        simd_width=1,
        target="gpu",
        _trace_description=kind,
        _heavy=True,
    ](Coord(numel), ctx)


def _go[kind: StaticString](argv: Argv, argc: Int) raises:
    if argc != 15:
        raise Error("complex_pointwise expects 15 arguments, got ", argc)
    var out_addr = _raw_int(argv[unsafe_offset=0])
    var a_addr = _raw_int(argv[unsafe_offset=1])
    var b_addr = _raw_int(argv[unsafe_offset=2])
    var c_addr = _raw_int(argv[unsafe_offset=3])
    var modes = _raw_int(argv[unsafe_offset=4])
    var h = SIMD[DType.float64, 8](0)
    for k in range(6):
        h[k] = _raw_f64(argv[unsafe_offset=5 + k])
    var numel = _raw_int(argv[unsafe_offset=11])
    h[6] = _raw_f64(argv[unsafe_offset=12])
    h[7] = _raw_f64(argv[unsafe_offset=13])
    var ctx = _raw_ctx(argv[unsafe_offset=14])
    if numel <= 0:
        return
    var handled = False
    comptime for dt in COMPONENT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            comptime if dt == DType.float64 and has_apple_gpu_accelerator():
                raise Error("float64 is not supported on Apple GPU")
            elif not has_accelerator():
                raise Error("no GPU accelerator available")
            else:
                _launch[kind, dt](
                    out_addr, a_addr, b_addr, c_addr, modes, h, numel, ctx
                )
                handled = True
    if not handled:
        raise Error("complex_pointwise ", kind, ": dtype not compiled in")


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kind and one component dtype per build.
    Slots are described in the header; errors come back as (rc=1, message).
    """
    try:
        comptime for kind in KINDS:
            comptime if _op_on[kind]():
                _go[kind](argv, argc)
                return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
