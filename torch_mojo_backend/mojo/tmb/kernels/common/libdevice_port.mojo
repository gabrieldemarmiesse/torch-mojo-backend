# ===----------------------------------------------------------------------=== #
# Bit-exact Mojo ports of the CUDA libdevice routines curand's Philox
# transforms and ATen's math kernels call.
#
# Each body is a literal translation of the matching `__nv_*` function in
# libdevice.10.bc (nvidia-cuda-nvcc wheel, nvvm/libdevice; disassemble with
# llvmlite.binding.parse_bitcode to follow along).  Every
# `__nvvm_reflect("__CUDA_FTZ")` branch takes the value-0 arm: PyTorch is built
# without -ftz, so these are the non-ftz variants.  `llvm.nvvm.fma.rn.*`
# becomes `math.fma`, `mul.rn`/`add.rn` become plain operators (except where
# Mojo would contract the pair into an fma -- see `_add_rn_d`), and the
# PTX-only instructions libdevice itself uses
# (rcp.approx.ftz, ex2.approx.ftz, sin/cos.approx, add.rz, fma.rm, cvt.sat,
# cvt.rni.s32) are emitted verbatim as inline PTX.
#
# Portability: the routines whose libdevice body is pure IEEE arithmetic
# (nv_logf, nv_exp, nv_log1pf, nv_sincospi) run the ported body on every target
# and are bit-exact everywhere.  The routines that depend on an NVIDIA hardware
# approximation instruction (nv_expf, nv_log, nv_log1p, nv_tanf, nv_tan,
# nv_fast_sincosf) fall back to std.math off NVIDIA and are NOT bit-exact
# there.
# ===----------------------------------------------------------------------=== #

from std.bit import count_leading_zeros
from std.collections import Array
from std.math import cos as _std_cos
from std.math import exp as _std_exp
from std.math import fma
from std.math import log as _std_log
from std.math import log1p as _std_log1p
from std.math import sin as _std_sin
from std.math import tan as _std_tan
from std.sys import llvm_intrinsic
from std.sys._assembly import inlined_assembly
from std.sys.info import is_gpu, is_nvidia_gpu


# --------------------------------------------------------------------------- #
# small conversions
# --------------------------------------------------------------------------- #


@always_inline
def _u32(x: Int32) -> UInt32:
    return x.cast[DType.uint32]()


@always_inline
def _i32(x: UInt32) -> Int32:
    return x.cast[DType.int32]()


@always_inline
def _z64(x: UInt32) -> UInt64:
    """Zero-extend to 64 bits. The mask is load-bearing: Mojo folds a chained
    Int32 -> UInt32 -> UInt64 cast into one SIGNED widening."""
    return x.cast[DType.uint64]() & UInt64(0xFFFFFFFF)


@always_inline
def _z128(x: UInt64) -> UInt128:
    """Zero-extend to 128 bits; see _z64 for why the mask is there."""
    return x.cast[DType.uint128]() & UInt128(0xFFFFFFFFFFFFFFFF)


@always_inline
def _hi(x: Float64) -> Int32:
    """llvm.nvvm.d2i.hi."""
    return (x.to_bits[DType.uint64]() >> UInt64(32)).cast[DType.int32]()


@always_inline
def _lo(x: Float64) -> Int32:
    """llvm.nvvm.d2i.lo."""
    return x.to_bits[DType.uint64]().cast[DType.int32]()


@always_inline
def _i2d(lo: Int32, hi: Int32) -> Float64:
    """llvm.nvvm.lohi.i2d."""
    return Float64(from_bits=(_z64(_u32(hi)) << UInt64(32)) | _z64(_u32(lo)))


@always_inline
def _fb(x: Float32) -> Int32:
    return x.to_bits[DType.uint32]().cast[DType.int32]()


@always_inline
def _fr(b: Int32) -> Float32:
    return Float32(from_bits=_u32(b))


comptime _F32_INF = Float32(from_bits=UInt32(0x7F800000))
comptime _F64_INF = Float64(from_bits=UInt64(0x7FF0000000000000))


# --------------------------------------------------------------------------- #
# PTX primitives
# --------------------------------------------------------------------------- #


@always_inline
def _rcp_approx_ftz_d(x: Float64) -> Float64:
    return inlined_assembly[
        "rcp.approx.ftz.f64 $0, $1;",
        Float64,
        constraints="=d,d",
        has_side_effect=False,
    ](x)


@always_inline
def _rcp_approx_ftz_f(x: Float32) -> Float32:
    return inlined_assembly[
        "rcp.approx.ftz.f32 $0, $1;",
        Float32,
        constraints="=f,f",
        has_side_effect=False,
    ](x)


@always_inline
def _ex2_approx_ftz_f(x: Float32) -> Float32:
    return inlined_assembly[
        "ex2.approx.ftz.f32 $0, $1;",
        Float32,
        constraints="=f,f",
        has_side_effect=False,
    ](x)


@always_inline
def _sin_approx_f(x: Float32) -> Float32:
    return inlined_assembly[
        "sin.approx.f32 $0, $1;",
        Float32,
        constraints="=f,f",
        has_side_effect=False,
    ](x)


@always_inline
def _cos_approx_f(x: Float32) -> Float32:
    return inlined_assembly[
        "cos.approx.f32 $0, $1;",
        Float32,
        constraints="=f,f",
        has_side_effect=False,
    ](x)


@always_inline
def _sat_f(x: Float32) -> Float32:
    """llvm.nvvm.saturate.f: clamp to [0, 1], NaN -> 0."""
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "cvt.sat.f32.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x)
    else:
        if not (x > Float32(0.0)):
            return Float32(0.0)
        return Float32(1.0) if x > Float32(1.0) else x


@always_inline
def _fma_rm_f(a: Float32, b: Float32, c: Float32) -> Float32:
    """fma.rm.f32: fused multiply-add rounded toward -infinity."""
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "fma.rm.f32 $0, $1, $2, $3;",
            Float32,
            constraints="=f,f,f,f",
            has_side_effect=False,
        ](a, b, c)
    else:
        # Only reached from nv_expf with a in [0,1], b = 252, c ~ 2**23, so
        # a*b + c is an integer that Float64 holds exactly and the nearest
        # Float32 is at most one integer above it.
        var r = fma(a, b, c)
        var exact = (
            a.cast[DType.float64]() * b.cast[DType.float64]()
            + c.cast[DType.float64]()
        )
        return r - Float32(1.0) if exact < r.cast[DType.float64]() else r


@always_inline
def _add_rz_f(a: Float32, b: Float32) -> Float32:
    """add.rz.f32: round toward zero."""
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "add.rz.f32 $0, $1, $2;",
            Float32,
            constraints="=f,f,f",
            has_side_effect=False,
        ](a, b)
    else:
        # TwoSum is exact for a finite, non-overflowing sum, so the sign of the
        # residual says whether round-to-nearest overshot away from zero.
        var s = a + b
        if not (s - s == Float32(0.0)):  # inf or nan: rounding mode is moot
            return s
        var bb = s - a
        var err = (a - (s - bb)) + (b - bb)
        if err == Float32(0.0):
            return s
        if (err < Float32(0.0)) == (s < Float32(0.0)):
            return s
        return Float32(from_bits=s.to_bits[DType.uint32]() - UInt32(1))


@always_inline
def _add_rn_d(a: Float64, b: Float64) -> Float64:
    """add.rn.f64 spelled out. Mojo contracts `x*y + z` into an fma; libdevice
    rounds the product first, so the fusion must be blocked here."""
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "add.rn.f64 $0, $1, $2;",
            Float64,
            constraints="=d,d,d",
            has_side_effect=False,
        ](a, b)
    else:
        return a + b


@always_inline
def _rint_d(x: Float64) -> Float64:
    return llvm_intrinsic["llvm.rint", Float64, has_side_effect=False](x)


@always_inline
def _trunc_d(x: Float64) -> Float64:
    return llvm_intrinsic["llvm.trunc", Float64, has_side_effect=False](x)


@always_inline
def _fabs_d(x: Float64) -> Float64:
    return llvm_intrinsic["llvm.fabs", Float64, has_side_effect=False](x)


@always_inline
def _fabs_f(x: Float32) -> Float32:
    return llvm_intrinsic["llvm.fabs", Float32, has_side_effect=False](x)


@always_inline
def _d2i_rn(x: Float64) -> Int32:
    """cvt.rni.s32.f64; every caller keeps the value in range."""
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "cvt.rni.s32.f64 $0, $1;",
            Int32,
            constraints="=r,d",
            has_side_effect=False,
        ](x)
    else:
        return _rint_d(x).cast[DType.int32]()


@always_inline
def _f2i_rn(x: Float32) -> Int32:
    """cvt.rni.s32.f32; every caller keeps the value in range."""
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "cvt.rni.s32.f32 $0, $1;",
            Int32,
            constraints="=r,f",
            has_side_effect=False,
        ](x)
    else:
        return llvm_intrinsic["llvm.rint", Float32, has_side_effect=False](
            x
        ).cast[DType.int32]()


# --------------------------------------------------------------------------- #
# __nv_logf
# --------------------------------------------------------------------------- #


@always_inline
def nv_logf(a: Float32) -> Float32:
    """`__nv_logf` (libdevice_ir/__nv_logf.ll). Pure IEEE: exact everywhere."""
    var m = a
    var i = Float32(0.0)
    if a < Float32(from_bits=UInt32(0x00800000)):  # 2**-126
        m = a * Float32(from_bits=UInt32(0x4B000000))  # 2**23
        i = Float32(-23.0)

    var e = (_fb(m) - Int32(1059760811)) & Int32(-8388608)  # bits(2/3f)
    var f = _fr(_fb(m) - e)
    i = fma(e.cast[DType.float32](), Float32(from_bits=UInt32(0x34000000)), i)

    var t = f - Float32(1.0)
    var p = fma(
        Float32(from_bits=UInt32(0xBE055027)),
        t,
        Float32(from_bits=UInt32(0x3E1039F6)),
    )
    p = fma(p, t, Float32(from_bits=UInt32(0xBDF8CDCC)))
    p = fma(p, t, Float32(from_bits=UInt32(0x3E0F2955)))
    p = fma(p, t, Float32(from_bits=UInt32(0xBE2AD8B9)))
    p = fma(p, t, Float32(from_bits=UInt32(0x3E4CED0B)))
    p = fma(p, t, Float32(from_bits=UInt32(0xBE7FFF22)))
    p = fma(p, t, Float32(from_bits=UInt32(0x3EAAAA78)))
    p = fma(p, t, Float32(-0.5))
    p = p * t
    p = fma(p, t, t)
    var r = fma(i, Float32(from_bits=UInt32(0x3F317218)), p)  # ln(2)

    if m.to_bits[DType.uint32]() >= UInt32(0x7F800000):
        r = fma(m, _F32_INF, _F32_INF)
    if m == Float32(0.0):
        r = -_F32_INF
    return r


# --------------------------------------------------------------------------- #
# __nv_expf
# --------------------------------------------------------------------------- #


@always_inline
def nv_expf(a: Float32) -> Float32:
    """`__nv_expf` (libdevice_ir/__nv_expf.ll). NVIDIA only: ex2.approx.ftz."""
    comptime if not is_nvidia_gpu():
        return _std_exp(a)

    comptime l2e = Float32(from_bits=UInt32(0x3FB8AA3B))  # log2(e) high
    comptime l2e_lo = Float32(from_bits=UInt32(0x32A57060))  # log2(e) low
    comptime magic = Float32(from_bits=UInt32(0x4B400000))  # 1.5 * 2**23
    comptime jscale = Float32(from_bits=UInt32(0x3BBB989D))  # fl32(l2e / 252)

    var t = fma(a, jscale, Float32(0.5))
    var s = _sat_f(t)
    var f = _fma_rm_f(
        s, Float32(252.0), (Float32(-126.0) + magic) + Float32(127.0)
    )
    var d = -(f - (magic + Float32(127.0)))
    var z = fma(a, l2e, d)
    z = fma(a, l2e_lo, z)
    # f is an integer in [1.5*2**23 + 1, +252]; its low 9 bits shifted into the
    # exponent field build 2**(f - 1.5*2**23).
    var scale = Float32(from_bits=f.to_bits[DType.uint32]() << UInt32(23))
    return _ex2_approx_ftz_f(z) * scale


# --------------------------------------------------------------------------- #
# __nv_log1pf
# --------------------------------------------------------------------------- #


@always_inline
def nv_log1pf(a: Float32) -> Float32:
    """`__nv_log1pf` (libdevice_ir/__nv_log1pf.ll). Pure IEEE plus add.rz."""
    var t = _add_rz_f(a, Float32(1.0))
    var e = (_fb(t) - Int32(1061158912)) & Int32(-8388608)  # bits(0.75f)
    var m = _fr(_fb(a) - e)
    var u = _fr(Int32(1082130432) - e)  # bits(4.0f)
    var x = m + fma(Float32(0.25), u, Float32(-1.0))
    var i = e.cast[DType.float32]() * Float32(from_bits=UInt32(0x34000000))

    var p = fma(
        Float32(from_bits=UInt32(0xBD39BF78)),
        x,
        Float32(from_bits=UInt32(0x3DD80012)),
    )
    p = fma(p, x, Float32(from_bits=UInt32(0xBE0778E0)))
    p = fma(p, x, Float32(from_bits=UInt32(0x3E146475)))
    p = fma(p, x, Float32(from_bits=UInt32(0xBE2A68DD)))
    p = fma(p, x, Float32(from_bits=UInt32(0x3E4CAF9E)))
    p = fma(p, x, Float32(from_bits=UInt32(0xBE800042)))
    p = fma(p, x, Float32(from_bits=UInt32(0x3EAAAAE6)))
    p = fma(p, x, Float32(-0.5))
    p = p * x
    p = fma(p, x, x)
    var r = fma(i, Float32(from_bits=UInt32(0x3F317218)), p)  # ln(2)

    if a.to_bits[DType.uint32]() >= UInt32(0x7F800000):
        if _fb(a) > Int32(-1082130432):  # signed bits(-1.0f)
            r = fma(a, _F32_INF, _F32_INF)
        if a == Float32(0.0):
            r = Float32(-0.0)
    return r


# --------------------------------------------------------------------------- #
# __nv_log / __nv_log1p
# --------------------------------------------------------------------------- #


@always_inline
def _log_core(a: Float64) -> Float64:
    """The body of `__nv_log` (libdevice_ir/__nv_log.ll)."""
    var x = a
    var ihi = _hi(a)
    var ilo = _lo(a)
    var e = Int32(-1023)
    if ihi < Int32(1048576):
        x = a * Float64(from_bits=UInt64(0x4350000000000000))  # 2**54
        e = Int32(-1023) - Int32(54)
        ihi = _hi(x)
        ilo = _lo(x)

    if ihi > Int32(0) and ihi < Int32(2146435072):
        e = e + _i32(_u32(ihi) >> UInt32(20))
        var mhi = (ihi & Int32(-2146435073)) | Int32(1072693248)
        var m = _i2d(ilo, mhi)
        if mhi > Int32(1073127582):  # m > sqrt(2)
            m = _i2d(_lo(m), Int32(-1048576) + _hi(m))
            e = e + Int32(1)

        var u = m - Float64(1.0)
        var v = m + Float64(1.0)
        var r0 = _rcp_approx_ftz_d(v)
        var nr = fma(-v, r0, Float64(1.0))
        nr = fma(nr, nr, nr)
        var rcp = fma(nr, r0, r0)

        var uq = u * rcp
        var w = _add_rn_d(uq, uq)
        var w2 = w * w
        var p = fma(
            Float64(from_bits=UInt64(0x3EB1380B3AE80F1E)),
            w2,
            Float64(from_bits=UInt64(0x3ED0EE258B7A8B04)),
        )
        p = fma(p, w2, Float64(from_bits=UInt64(0x3EF3B2669F02676F)))
        p = fma(p, w2, Float64(from_bits=UInt64(0x3F1745CBA9AB0956)))
        p = fma(p, w2, Float64(from_bits=UInt64(0x3F3C71C72D1B5154)))
        p = fma(p, w2, Float64(from_bits=UInt64(0x3F624924923BE72D)))
        p = fma(p, w2, Float64(from_bits=UInt64(0x3F8999999999A3C4)))
        p = fma(p, w2, Float64(from_bits=UInt64(0x3FB5555555555554)))

        var dd = Float64(2.0) * (u - w)
        var cc = fma(-w, u, dd)
        var lo2 = rcp * cc
        var hi2 = p * w2
        var s = fma(hi2, w, lo2)

        var ef = _i2d(Int32(-2147483648) ^ e, Int32(1127219200)) - _i2d(
            Int32(-2147483648), Int32(1127219200)
        )
        var z = fma(ef, Float64(from_bits=UInt64(0x3FE62E42FEFA39EF)), w)
        var zz = fma(-ef, Float64(from_bits=UInt64(0x3FE62E42FEFA39EF)), z)
        var k = s - (zz - w)
        k = fma(ef, Float64(from_bits=UInt64(0x3C7ABC9E3B39803F)), k)
        return z + k

    var q = fma(x, _F64_INF, _F64_INF)
    if _fr(_hi(x)) == Float32(0.0):
        q = -_F64_INF
    return q


@always_inline
def nv_log(a: Float64) -> Float64:
    """`__nv_log` (libdevice_ir/__nv_log.ll). NVIDIA only: rcp.approx.ftz.f64.
    """
    comptime if not is_nvidia_gpu():
        return _std_log(a)
    return _log_core(a)


@always_inline
def nv_log1p(a: Float64) -> Float64:
    """`__nv_log1p` (libdevice_ir/__nv_log1p.ll). NVIDIA only (uses _log_core).
    """
    comptime if not is_nvidia_gpu():
        return _std_log1p(a)

    var ahi = _fr(_hi(a))
    if ahi < Float32(from_bits=UInt32(0x3FE55555)) and ahi > Float32(
        from_bits=UInt32(0xBFD99999)
    ):
        var u = a / (a + Float64(2.0))
        var c = (-a) * u
        var t = _add_rn_d(a, c)
        var t2 = t * t
        var p = fma(
            Float64(from_bits=UInt64(0x3EB372FB2FBE14B5)),
            t2,
            Float64(from_bits=UInt64(0x3ED087FFCEB2DC44)),
        )
        p = fma(p, t2, Float64(from_bits=UInt64(0x3EF3B9FF890F468C)))
        p = fma(p, t2, Float64(from_bits=UInt64(0x3F17457EFD51BAF8)))
        p = fma(p, t2, Float64(from_bits=UInt64(0x3F3C71C8DE3CE825)))
        p = fma(p, t2, Float64(from_bits=UInt64(0x3F6249248FA4661F)))
        p = fma(p, t2, Float64(from_bits=UInt64(0x3F899999999D70C4)))
        p = fma(p, t2, Float64(from_bits=UInt64(0x3FB5555555555462)))
        p = p * t2
        return fma(p, t, c) + a

    return _log_core(a + Float64(1.0))


# --------------------------------------------------------------------------- #
# __nv_exp
# --------------------------------------------------------------------------- #


@always_inline
def nv_exp(a: Float64) -> Float64:
    """`__nv_exp` (libdevice_ir/__nv_exp.ll). Pure IEEE: exact everywhere."""
    comptime l2e = Float64(from_bits=UInt64(0x3FF71547652B82FE))
    comptime magic = Float64(from_bits=UInt64(0x4338000000000000))  # 1.5*2**52

    var t = fma(a, l2e, magic)
    var j = _lo(t)
    var f = t + Float64(from_bits=UInt64(0xC338000000000000))
    var r = fma(f, Float64(from_bits=UInt64(0xBFE62E42FEFA39EF)), a)
    r = fma(f, Float64(from_bits=UInt64(0xBC7ABC9E3B39803F)), r)

    var p = fma(
        Float64(from_bits=UInt64(0x3E5ADE1569CE2BDF)),
        r,
        Float64(from_bits=UInt64(0x3E928AF3FCA213EA)),
    )
    p = fma(p, r, Float64(from_bits=UInt64(0x3EC71DEE62401315)))
    p = fma(p, r, Float64(from_bits=UInt64(0x3EFA01997C89EB71)))
    p = fma(p, r, Float64(from_bits=UInt64(0x3F2A01A014761F65)))
    p = fma(p, r, Float64(from_bits=UInt64(0x3F56C16C1852B7AF)))
    p = fma(p, r, Float64(from_bits=UInt64(0x3F81111111122322)))
    p = fma(p, r, Float64(from_bits=UInt64(0x3FA55555555502A1)))
    p = fma(p, r, Float64(from_bits=UInt64(0x3FC5555555555511)))
    p = fma(p, r, Float64(from_bits=UInt64(0x3FE000000000000B)))
    p = fma(p, r, Float64(1.0))
    p = fma(p, r, Float64(1.0))

    var z = _i2d(_lo(p), (j << Int32(20)) + _hi(p))

    if _fabs_f(_fr(_hi(a))) < Float32(from_bits=UInt32(0x4086232B)):
        return z

    var zz = Float64(0.0)
    if not (a < Float64(0.0)):
        zz = a + _F64_INF
    if _fabs_f(_fr(_hi(a))) < Float32(from_bits=UInt32(0x40874800)):
        # Split the scaling in two so the intermediate stays normal.
        var jh = (j + ((j >> Int32(31)) & Int32(1))) >> Int32(1)  # sdiv 2
        var s1 = _i2d(_lo(p), _hi(p) + (jh << Int32(20)))
        var s2 = _i2d(Int32(0), Int32(1072693248) + ((j - jh) << Int32(20)))
        zz = s1 * s2
    return zz


# --------------------------------------------------------------------------- #
# __nv_sincospi
# --------------------------------------------------------------------------- #


@always_inline
def nv_sincospi(a: Float64) -> Tuple[Float64, Float64]:
    """`__nv_sincospi` (libdevice_ir/__nv_sincospi.ll). Pure IEEE: exact
    everywhere. Returns (sin(pi*a), cos(pi*a))."""
    var x = a
    var h = _hi(a)
    if _u32(h + h) > UInt32(0x86800000):
        x = a * Float64(0.0)

    var two_x = _i2d(_lo(x), Int32(1048576) + _hi(x))  # exact doubling
    var rn = _rint_d(two_x)
    var i = rn.cast[DType.int64]().cast[DType.int32]()
    var t = fma(-rn, Float64(0.5), x)

    var lo = t * Float64(from_bits=UInt64(0x3CA1A62633145C07))
    var r = fma(t, Float64(from_bits=UInt64(0x400921FB54442D18)), lo)

    var r2 = r * r
    var c = fma(
        Float64(from_bits=UInt64(0xBDA8FF8320FD8164)),
        r2,
        Float64(from_bits=UInt64(0x3E21EEA7C1EF8528)),
    )
    c = fma(c, r2, Float64(from_bits=UInt64(0xBE927E4F8E06E6D9)))
    c = fma(c, r2, Float64(from_bits=UInt64(0x3EFA01A019DDBCE9)))
    c = fma(c, r2, Float64(from_bits=UInt64(0xBF56C16C16C15D47)))
    c = fma(c, r2, Float64(from_bits=UInt64(0x3FA5555555555551)))
    c = fma(c, r2, Float64(-0.5))
    c = fma(c, r2, Float64(1.0))

    var q2 = r * r
    var s = fma(
        Float64(from_bits=UInt64(0x3DE5DB65F9785EBA)),
        q2,
        Float64(from_bits=UInt64(0xBE5AE5F12CB0D246)),
    )
    s = fma(s, q2, Float64(from_bits=UInt64(0x3EC71DE369ACE392)))
    s = fma(s, q2, Float64(from_bits=UInt64(0xBF2A01A019DB62A1)))
    s = fma(s, q2, Float64(from_bits=UInt64(0x3F81111111110818)))
    s = fma(s, q2, Float64(from_bits=UInt64(0xBFC5555555555554)))
    s = fma(s, q2, Float64(0.0))
    s = fma(s, r, r)

    var neg_s = _i2d(_lo(s), _hi(s) ^ Int32(-2147483648))

    var sv = s
    var cv = c
    if (i & Int32(1)) != Int32(0):
        sv = c
        cv = neg_s
    if (i & Int32(2)) != Int32(0):
        sv = _i2d(_lo(sv), _hi(sv) ^ Int32(-2147483648))
        cv = _i2d(_lo(cv), _hi(cv) ^ Int32(-2147483648))

    var cout = fma(cv, Float64(1.0), Float64(0.0))
    if x == _trunc_d(x):
        sv = x * Float64(0.0)
    return (sv, cout)


# --------------------------------------------------------------------------- #
# __nv_tanf
# --------------------------------------------------------------------------- #


@always_inline
def _trig_reduction_slowpath_f(a: Float32) -> Tuple[Float32, Int32]:
    """`__internal_trig_reduction_slowpath` from __nv_tanf.ll (Payne-Hanek)."""
    # __cudart_i2opi_f: 2/pi, most significant word last.
    var i2opi = Array[UInt32, 6](fill=UInt32(0))
    i2opi[0] = UInt32(0x3C439041)
    i2opi[1] = UInt32(0xDB629599)
    i2opi[2] = UInt32(0xF534DDC0)
    i2opi[3] = UInt32(0xFC2757D1)
    i2opi[4] = UInt32(0x4E441529)
    i2opi[5] = UInt32(0xA2F9836E)

    var ia = _fb(a)
    var sign = ia & Int32(-2147483648)
    var e = ((ia >> Int32(23)) & Int32(255)) - Int32(128)
    var hi_in = _u32(ia << Int32(8)) | UInt32(0x80000000)
    var idx = Int32(4) - _i32(_u32(e) >> UInt32(5))

    var result = Array[UInt32, 7](fill=UInt32(0))
    var carry_hi = UInt32(0)
    for q in range(6):
        var prod = _z64(i2opi[q]) * _z64(hi_in) + _z64(carry_hi)
        result[q] = prod.cast[DType.uint32]()
        carry_hi = (prod >> UInt64(32)).cast[DType.uint32]()
    result[6] = carry_hi

    var sh = _u32(e & Int32(31))
    var hw = result[Int(idx) + 2]
    var lw = result[Int(idx) + 1]
    if sh != UInt32(0):
        var s2 = UInt32(32) - sh
        hw = (result[Int(idx) + 2] << sh) + (result[Int(idx) + 1] >> s2)
        lw = (result[Int(idx) + 1] << sh) + (result[Int(idx)] >> s2)

    var top = hw >> UInt32(30)
    var hi2 = (hw << UInt32(2)) + (lw >> UInt32(30))
    var lo2 = lw << UInt32(2)
    var carry = hi2 >> UInt32(31)
    var quad = _i32(top + carry)
    if sign != Int32(0):
        quad = -quad

    var s = sign
    if carry != UInt32(0):
        hi2 = hi2 ^ UInt32(0xFFFFFFFF)
        lo2 = lo2 ^ UInt32(0xFFFFFFFF)
        s = sign ^ Int32(-2147483648)

    var p = ((_z64(hi2) << UInt64(32)) | _z64(lo2)).cast[DType.int64]()
    var r = (
        p.cast[DType.float64]() * Float64(from_bits=UInt64(0x3BF921FB54442D19))
    ).cast[DType.float32]()
    if s != Int32(0):
        r = -r
    return (r, quad)


@always_inline
def nv_tanf(a: Float32) -> Float32:
    """`__nv_tanf` (libdevice_ir/__nv_tanf.ll). NVIDIA only: rcp.approx.ftz."""
    comptime if not is_nvidia_gpu():
        # std.math.tan is libm, i.e. CPU-only.
        comptime if is_gpu():
            return _std_sin(a) / _std_cos(a)
        else:
            return _std_tan(a)

    var j = _f2i_rn(a * Float32(from_bits=UInt32(0x3F22F983)))  # 2/pi
    var jf = j.cast[DType.float32]()
    var t = fma(jf, Float32(from_bits=UInt32(0xBFC90FDA)), a)
    t = fma(jf, Float32(from_bits=UInt32(0xB3A22168)), t)
    t = fma(jf, Float32(from_bits=UInt32(0xA7C234C5)), t)
    var i = j

    if _fabs_f(a) >= Float32(105615.0):
        if _fabs_f(a) == _F32_INF:
            t = a * Float32(0.0)
            i = Int32(0)
        else:
            var rq = _trig_reduction_slowpath_f(a)
            t = rq[0]
            i = rq[1]

    var x2 = t * t
    var p = fma(
        Float32(from_bits=UInt32(0x3C190000)),
        x2,
        Float32(from_bits=UInt32(0x3B560000)),
    )
    p = fma(p, x2, Float32(from_bits=UInt32(0x3CC70000)))
    p = fma(p, x2, Float32(from_bits=UInt32(0x3D5B0000)))
    p = fma(p, x2, Float32(from_bits=UInt32(0x3E089438)))
    p = fma(p, x2, Float32(from_bits=UInt32(0x3EAAAA88)))
    var u = x2 * t
    var z = fma(p, u, t)
    if _fabs_f(t) == Float32(from_bits=UInt32(0x3A00B43C)):
        z = t

    if (i & Int32(1)) != Int32(0):
        z = _rcp_approx_ftz_f(-z)
    return z


# --------------------------------------------------------------------------- #
# __nv_tan
# --------------------------------------------------------------------------- #


@always_inline
def _i2opi_d(k: Int32) -> UInt64:
    """`__cudart_i2opi_d[k]`: 2/pi, most significant word last."""
    var t = Array[UInt64, 18](fill=UInt64(0))
    t[0] = UInt64(0x6BFB5FB11F8D5D08)
    t[1] = UInt64(0x3D0739F78A5292EA)
    t[2] = UInt64(0x7527BAC7EBE5F17B)
    t[3] = UInt64(0x4F463F669E5FEA2D)
    t[4] = UInt64(0x6D367ECF27CB09B7)
    t[5] = UInt64(0xEF2F118B5A0A6D1F)
    t[6] = UInt64(0x1FF897FFDE05980F)
    t[7] = UInt64(0x9C845F8BBDF9283B)
    t[8] = UInt64(0x3991D639835339F4)
    t[9] = UInt64(0xE99C7026B45F7E41)
    t[10] = UInt64(0xE88235F52EBB4484)
    t[11] = UInt64(0xFE1DEB1CB129A73E)
    t[12] = UInt64(0x06492EEA09D1921C)
    t[13] = UInt64(0xB7246E3A424DD2E0)
    t[14] = UInt64(0xFE5163ABDEBBC561)
    t[15] = UInt64(0xDB6295993C439041)
    t[16] = UInt64(0xFC2757D1F534DDC0)
    t[17] = UInt64(0xA2F9836E4E441529)
    return t[Int(k)]


@always_inline
def _trig_reduction_slowpath_d(
    a: Float64, q_in: Int32
) -> Tuple[Float64, Int32]:
    """`__internal_trig_reduction_slowpathd` from libdevice (Payne-Hanek).

    inf/nan leaves the caller's quadrant untouched, as the original does."""
    var ihi = _hi(a)
    var sign = ihi & Int32(-2147483648)
    var expo = _i32((_u32(ihi) >> UInt32(20)) & UInt32(2047))
    if expo == Int32(2047):
        return (a, q_in)

    var e = expo - Int32(1024)
    var b = (a.to_bits[DType.uint64]() << UInt64(11)) | UInt64(
        0x8000000000000000
    )
    var base = Int32(16) - _i32(_u32(e) >> UInt32(6))

    var result = Array[UInt64, 5](fill=UInt64(0))
    var p = UInt64(0)
    var q = base - Int32(1)
    while True:
        var limit = Int32(18) if Int32(18) < base + Int32(3) else base + Int32(
            3
        )
        if q >= limit:
            break
        var prod = _z128(_i2opi_d(q)) * _z128(b) + _z128(p)
        result[Int(q - (base - Int32(1)))] = prod.cast[DType.uint64]()
        p = (prod >> UInt128(64)).cast[DType.uint64]()
        q = q + Int32(1)
    result[Int(q - (base - Int32(1)))] = p

    var sh = _z64(_u32(e & Int32(63)))
    var hi = result[3]
    var lo = result[2]
    if sh != UInt64(0):
        var s2 = UInt64(64) - sh
        hi = (result[3] << sh) | (result[2] >> s2)
        lo = (result[2] << sh) | (result[1] >> s2)

    var top = (hi >> UInt64(62)).cast[DType.uint32]()
    var hi2 = (hi << UInt64(2)) | (lo >> UInt64(62))
    var lo2 = lo << UInt64(2)
    var carry = (hi2 >> UInt64(63)).cast[DType.uint32]()
    var quad = _i32(top) + _i32(carry)
    if sign != Int32(0):
        quad = -quad

    var s = sign
    if carry != UInt32(0):
        var neg = UInt128(0) - ((_z128(hi2) << UInt128(64)) | _z128(lo2))
        lo2 = neg.cast[DType.uint64]()
        hi2 = (neg >> UInt128(64)).cast[DType.uint64]()
        s = sign ^ Int32(-2147483648)

    var nz = count_leading_zeros(hi2).cast[DType.uint32]()
    var hi3 = hi2
    if nz != UInt32(0) and nz != UInt32(64):
        hi3 = (hi2 << _z64(nz)) | (lo2 >> _z64(UInt32(64) - nz))

    var prod2 = _z128(hi3) * _z128(UInt64(0xC90FDAA22168C235))
    var phi = (prod2 >> UInt128(64)).cast[DType.uint64]()
    var plo = prod2.cast[DType.uint64]()
    var ee = nz
    if phi.cast[DType.int64]() > Int64(0):
        var dbl = (_z128(phi) << UInt128(64)) | _z128(plo)
        dbl = dbl + dbl
        phi = (dbl >> UInt128(64)).cast[DType.uint64]()
        ee = nz + UInt32(1)

    var expfield = _z64(UInt32(1022) - ee)
    var mant = (((phi + UInt64(1)) >> UInt64(10)) + UInt64(1)) >> UInt64(1)
    var bits = (_z64(_u32(s)) << UInt64(32)) | ((expfield << UInt64(52)) + mant)
    return (Float64(from_bits=bits), quad)


@always_inline
def nv_tan(a: Float64) -> Float64:
    """`__nv_tan` (libdevice_ir/__nv_tan.ll).

    AMD GPUs run the same double algorithm with an IEEE division for the one
    rcp.approx.ftz.f64 (then refined by the same Newton steps): LLVM has no
    AMDGPU lowering for a float64 `llvm.sin` / `llvm.cos` ("Cannot select:
    f64 = fsin"), and ROCm's own tan (ocml `__ocml_tan_f64`, what the
    hipified jiterator strings call) is a double reduction plus polynomial of
    the same accuracy. Apple GPUs have no float64 at all.
    """
    comptime if not is_gpu():
        return _std_tan(a)

    var z: Float64
    var i = Int32(0)
    if (_hi(a) & Int32(2147483647)) == Int32(2146435072) and _lo(a) == Int32(0):
        z = a * Float64(0.0)
    else:
        var q = _d2i_rn(a * Float64(from_bits=UInt64(0x3FE45F306DC9C883)))
        var qf = q.cast[DType.float64]()
        var t = fma(-qf, Float64(from_bits=UInt64(0x3FF921FB54442D18)), a)
        t = fma(-qf, Float64(from_bits=UInt64(0x3C91A62633145C00)), t)
        t = fma(-qf, Float64(from_bits=UInt64(0x397B839A252049C0)), t)
        i = q
        if _fabs_d(a) >= Float64(from_bits=UInt64(0x41E0000000000000)):
            var rq = _trig_reduction_slowpath_d(a, q)
            t = rq[0]
            i = rq[1]
        z = t

    var x2 = z * z
    var p = fma(
        Float64(from_bits=UInt64(0x3EE48DAC2799BCB9)),
        x2,
        Float64(from_bits=UInt64(0xBEF9757C5B27EBB1)),
    )
    p = fma(p, x2, Float64(from_bits=UInt64(0x3F0980E90FD91E04)))
    p = fma(p, x2, Float64(from_bits=UInt64(0xBEFAE2B0417D7E1D)))
    p = fma(p, x2, Float64(from_bits=UInt64(0x3F119F5341BFBA57)))
    p = fma(p, x2, Float64(from_bits=UInt64(0x3F15E791A00F6919)))
    p = fma(p, x2, Float64(from_bits=UInt64(0x3F2FF2E7FADEC73A)))
    p = fma(p, x2, Float64(from_bits=UInt64(0x3F434BC1B206DA62)))
    p = fma(p, x2, Float64(from_bits=UInt64(0x3F57DB18EF2F83F9)))
    p = fma(p, x2, Float64(from_bits=UInt64(0x3F6D6D2E7AE49FBC)))
    p = fma(p, x2, Float64(from_bits=UInt64(0x3F8226E3A816A776)))
    p = fma(p, x2, Float64(from_bits=UInt64(0x3F9664F485D25660)))
    p = fma(p, x2, Float64(from_bits=UInt64(0x3FABA1BA1BABF31D)))
    p = fma(p, x2, Float64(from_bits=UInt64(0x3FC11111111105D2)))
    p = fma(p, x2, Float64(from_bits=UInt64(0x3FD555555555555E)))
    var u = p * x2
    var r = fma(u, z, z)

    if (i & Int32(1)) != Int32(0):
        var d = fma(u, z, -(r - z))
        var r0: Float64
        comptime if is_nvidia_gpu():
            r0 = _rcp_approx_ftz_d(r)
        else:
            r0 = Float64(1.0) / r
        var nr = fma(-r, r0, Float64(1.0))
        nr = fma(nr, nr, nr)
        var rr = -fma(nr, r0, r0)
        var w = fma(r, rr, Float64(1.0))
        w = fma(rr, d, w)
        r = fma(w, rr, rr)
    return r


# --------------------------------------------------------------------------- #
# __nv_fast_sincosf  (== CUDA's __sincosf)
# --------------------------------------------------------------------------- #


@always_inline
def nv_fast_sincosf(a: Float32) -> Tuple[Float32, Float32]:
    """`__nv_fast_sincosf` (libdevice_ir/__nv_fast_sincosf.ll): sin.approx.f32
    and cos.approx.f32, non-ftz. NVIDIA only."""
    comptime if not is_nvidia_gpu():
        return (_std_sin(a), _std_cos(a))
    return (_sin_approx_f(a), _cos_approx_f(a))


# --------------------------------------------------------------------------- #
# The CUDA fast-math intrinsics ATen's float transforms call on device
# (ATen/NumericUtils.h: at::log/exp/tan/log1p use __logf/__expf/__tanf).
# --------------------------------------------------------------------------- #


@always_inline
def _lg2_approx_f(x: Float32) -> Float32:
    return inlined_assembly[
        "lg2.approx.f32 $0, $1;",
        Float32,
        constraints="=f,f",
        has_side_effect=False,
    ](x)


@always_inline
def _ex2_approx_f(x: Float32) -> Float32:
    return inlined_assembly[
        "ex2.approx.f32 $0, $1;",
        Float32,
        constraints="=f,f",
        has_side_effect=False,
    ](x)


@always_inline
def _div_approx_f(x: Float32, y: Float32) -> Float32:
    return inlined_assembly[
        "div.approx.f32 $0, $1, $2;",
        Float32,
        constraints="=f,f,f",
        has_side_effect=False,
    ](x, y)


@always_inline
def nv_fast_logf(a: Float32) -> Float32:
    """`__nv_fast_logf` (== CUDA `__logf`): lg2.approx.f32 * ln 2. NVIDIA only.
    """
    comptime if not is_nvidia_gpu():
        return _std_log(a)
    return Float32(0.6931471824645996) * _lg2_approx_f(a)


@always_inline
def nv_fast_expf(a: Float32) -> Float32:
    """`__nv_fast_expf` (== CUDA `__expf`): ex2.approx.f32(a * log2 e). NVIDIA only.
    """
    comptime if not is_nvidia_gpu():
        return _std_exp(a)
    return _ex2_approx_f(a * Float32(1.4426950216293335))


@always_inline
def nv_fast_tanf(a: Float32) -> Float32:
    """`__nv_fast_tanf` (== CUDA `__tanf`): div.approx(sin.approx, cos.approx). NVIDIA only.
    """
    comptime if not is_nvidia_gpu():
        return _std_sin(a) / _std_cos(a)
    return _div_approx_f(_sin_approx_f(a), _cos_approx_f(a))
