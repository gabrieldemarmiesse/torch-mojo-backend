# ===----------------------------------------------------------------------=== #
# Elementwise kernels of the complex dtypes (tmb/ops/complex.mojo).
#
# A complex64 / complex128 tensor is, here, its interleaved float32 / float64
# storage: element i is the pair at 2i, 2i + 1.
#
# The ops layer hands this family flat operands only, as the pointwise family
# does: each is a dense buffer of the output's extent ("full"), one element
# on the device (a 0-d tensor, read once per element and splatted) or a host
# scalar (its two components in slots). Strided and broadcast inputs are
# materialized there first. One complex element per lane.
#
# Kinds, by operand and result type ("cx" = complex pairs, "re" = the
# component dtype, "b" = bool bytes):
#   cx -> cx : conj neg
#   cx -> re : real imag
#   cx -> b  : nonzero
#   cx, cx -> b  : eq ne
#
# Slots: out, a, b (addresses, 0 for an unused/immediate operand), modes (2
# bits per operand: 0 full, 1 device element, 2 host value), a_re, a_im,
# b_re, b_im (f64 bits), numel (complex elements), p_re, p_im (f64 bits: a
# kind's scalar parameter), DeviceContext.
# Defines: OP (the kind), DTYPE_ARG_0 (the component dtype, float32/float64).
# ===----------------------------------------------------------------------=== #

from max.gpu.host import DeviceContext
from std.sys.info import has_accelerator, has_apple_gpu_accelerator, size_of
from std.utils.coord import Coord
from std.utils.numerics import isnan

from tmb.kernels.common.complex_math import Cx
from tmb.kernels.common.gpu_elementwise import elementwise
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

comptime KINDS_CX_CX = ["conj", "neg"]
comptime KINDS_CX_RE = ["real", "imag"]
comptime KINDS_CX_B = ["nonzero"]
comptime KINDS_CXCX_B = ["eq", "ne"]


@always_inline
def _unary_cx[kind: StaticString, dt: DType](z: Cx[dt]) -> Cx[dt]:
    comptime if kind == "conj":
        return Cx[dt](z.re, -z.im)
    else:
        comptime assert kind == "neg", "unknown complex unary kind"
        return Cx[dt](-z.re, -z.im)


@always_inline
def _unary_re[kind: StaticString, dt: DType](z: Cx[dt]) -> Scalar[dt]:
    comptime if kind == "real":
        return z.re
    else:
        comptime assert kind == "imag", "unknown complex-to-real kind"
        return z.im


@always_inline
def _unary_b[kind: StaticString, dt: DType](z: Cx[dt]) -> Bool:
    comptime assert kind == "nonzero", "unknown complex predicate"
    return z.re != 0 or z.im != 0 or isnan(z.re) or isnan(z.im)


@always_inline
def _binary_b[kind: StaticString, dt: DType](a: Cx[dt], b: Cx[dt]) -> Bool:
    # A NaN component compares unequal; tested on the bits (the build
    # assumes no NaN in comparisons).
    var nan = isnan(a.re) or isnan(a.im) or isnan(b.re) or isnan(b.im)
    var same = not nan and a.re == b.re and a.im == b.im
    comptime if kind == "eq":
        return same
    else:
        comptime assert kind == "ne", "unknown complex comparison"
        return not same


@always_inline
def _to_re[kind: StaticString]() -> Bool:
    return kind == "real" or kind == "imag"


@always_inline
def _to_b[kind: StaticString]() -> Bool:
    return kind == "nonzero" or kind == "eq" or kind == "ne"


@always_inline
def _arity[kind: StaticString]() -> Int:
    comptime if kind == "eq" or kind == "ne":
        return 2
    else:
        return 1


@always_inline
def _launch[
    kind: StaticString, dt: DType
](
    out_addr: Int,
    a_addr: Int,
    b_addr: Int,
    mode_a: Int,
    mode_b: Int,
    a_re: Float64,
    a_im: Float64,
    b_re: Float64,
    b_im: Float64,
    p_re: Float64,
    p_im: Float64,
    numel: Int,
    ctx: DeviceContext,
) raises:
    comptime arity = _arity[kind]()
    var out_ptr = _make_ptr[dt](out_addr)
    var out_b = _make_ptr[DType.uint8](out_addr)
    var a_ptr = _make_ptr[dt](a_addr)
    var b_ptr = _make_ptr[dt](b_addr)
    var ha_re = a_re.cast[dt]()
    var ha_im = a_im.cast[dt]()
    var hb_re = b_re.cast[dt]()
    var hb_im = b_im.cast[dt]()
    var hp_re = p_re.cast[dt]()
    var hp_im = p_im.cast[dt]()

    @always_inline
    @__parameter
    @__copy_capture(
        out_ptr,
        out_b,
        a_ptr,
        b_ptr,
        mode_a,
        mode_b,
        ha_re,
        ha_im,
        hb_re,
        hb_im,
        hp_re,
        hp_im,
    )
    def func[w: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var ai = 2 * i if mode_a == MODE_FULL else 0
        var a = Cx[dt](ha_re, ha_im)
        if mode_a != MODE_HOST:
            a = Cx[dt](a_ptr[unsafe_offset=ai], a_ptr[unsafe_offset=ai + 1])
        comptime if arity == 1:
            comptime if not _to_re[kind]() and not _to_b[kind]():
                var r = _unary_cx[kind, dt](a)
                out_ptr[unsafe_offset=2 * i] = r.re
                out_ptr[unsafe_offset=2 * i + 1] = r.im
            elif _to_re[kind]():
                out_ptr[unsafe_offset=i] = _unary_re[kind, dt](a)
            else:
                out_b[unsafe_offset=i] = UInt8(1) if _unary_b[kind, dt](
                    a
                ) else UInt8(0)
        else:
            var bi = 2 * i if mode_b == MODE_FULL else 0
            var b = Cx[dt](hb_re, hb_im)
            if mode_b != MODE_HOST:
                b = Cx[dt](b_ptr[unsafe_offset=bi], b_ptr[unsafe_offset=bi + 1])
            comptime assert _to_b[kind](), "unknown complex binary kind"
            out_b[unsafe_offset=i] = UInt8(1) if _binary_b[kind, dt](
                a, b
            ) else UInt8(0)

    elementwise[
        func,
        simd_width=1,
        target="gpu",
        _trace_description=kind,
        _heavy=True,
    ](Coord(numel), ctx)


def _complex_go[kind: StaticString](argv: Argv, argc: Int) raises:
    if argc != 12:
        raise Error("complex expects 12 arguments, got ", argc)
    var out_addr = _raw_int(argv[unsafe_offset=0])
    var a_addr = _raw_int(argv[unsafe_offset=1])
    var b_addr = _raw_int(argv[unsafe_offset=2])
    var modes = _raw_int(argv[unsafe_offset=3])
    var a_re = _raw_f64(argv[unsafe_offset=4])
    var a_im = _raw_f64(argv[unsafe_offset=5])
    var b_re = _raw_f64(argv[unsafe_offset=6])
    var b_im = _raw_f64(argv[unsafe_offset=7])
    var numel = _raw_int(argv[unsafe_offset=8])
    var p_re = _raw_f64(argv[unsafe_offset=9])
    var p_im = _raw_f64(argv[unsafe_offset=10])
    var ctx = _raw_ctx(argv[unsafe_offset=11])
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
                    out_addr,
                    a_addr,
                    b_addr,
                    modes & 3,
                    (modes >> 2) & 3,
                    a_re,
                    a_im,
                    b_re,
                    b_im,
                    p_re,
                    p_im,
                    numel,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error("complex ", kind, ": dtype not compiled in")


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kind and one component dtype per build.
    Slots are described in the header; errors come back as (rc=1, message).
    """
    try:
        comptime for kind in KINDS_CX_CX:
            comptime if _op_on[kind]():
                _complex_go[kind](argv, argc)
                return 0
        comptime for kind in KINDS_CX_RE:
            comptime if _op_on[kind]():
                _complex_go[kind](argv, argc)
                return 0
        comptime for kind in KINDS_CX_B:
            comptime if _op_on[kind]():
                _complex_go[kind](argv, argc)
                return 0
        comptime for kind in KINDS_CXCX_B:
            comptime if _op_on[kind]():
                _complex_go[kind](argv, argc)
                return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
