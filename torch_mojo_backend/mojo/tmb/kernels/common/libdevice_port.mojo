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
# (nv_logf, nv_log2f, nv_exp, nv_log1pf, nv_sincospi, nv_sincos) run the ported
# body on every target and are bit-exact everywhere.  The float64 nv_log /
# nv_log1p / nv_tan run the ported body on every GPU too (std.math on the
# host): their one approximation instruction, rcp.approx.ftz.f64, only seeds a
# reciprocal that libdevice refines cubically, so an IEEE `1 / x` seed off
# NVIDIA gives the same reciprocal (see `_rcp_approx_ftz_d`).  nv_log /
# nv_log1p must not fall back to std.math on a GPU: its float64 `log` is the
# Cephes *single-precision* polynomial evaluated in double (3e-9 relative off
# on MI300A; log1p, which calls it, 1e-6), where this port is within an ulp.
# The routines that depend on an NVIDIA hardware approximation for their result
# (nv_expf, nv_tanf, nv_fast_sincosf) fall back to std.math off NVIDIA and
# are NOT bit-exact there; nv_asinhf / nv_asinh are NVIDIA or host (libm) only.
# ===----------------------------------------------------------------------=== #

from std.bit import count_leading_zeros
from std.collections import Array
from std.math import asinh as _std_asinh
from std.math import cos as _std_cos
from std.math import exp as _std_exp
from std.math import fma
from std.math import log as _std_log
from std.math import log1p as _std_log1p
from std.math import sin as _std_sin
from std.math import sqrt as _std_sqrt
from std.math import tan as _std_tan
from std.sys import llvm_intrinsic
from std.sys._assembly import inlined_assembly
from std.sys.info import is_amd_gpu, is_gpu, is_nvidia_gpu


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
    """rcp.approx.ftz.f64 on NVIDIA; an IEEE `1 / x` elsewhere. Each caller
    (`_log_core`, `nv_tan`) only seeds a refinement r = r0 + r0 (e + e^2),
    e = 1 - x r0, that takes the seed's error e to e^3: from the hardware
    approximation or from the rounded quotient, the refined value is the
    correctly rounded reciprocal either way."""
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "rcp.approx.ftz.f64 $0, $1;",
            Float64,
            constraints="=d,d",
            has_side_effect=False,
        ](x)
    else:
        return Float64(1.0) / x


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
    rounds the product first, so the fusion must be blocked here (v_add_f64
    on AMD, as pow_math's `_add_rn`)."""
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "add.rn.f64 $0, $1, $2;",
            Float64,
            constraints="=d,d,d",
            has_side_effect=False,
        ](a, b)
    elif is_amd_gpu():
        return inlined_assembly[
            "v_add_f64 $0, $1, $2",
            Float64,
            constraints="=v,v,v",
            has_side_effect=False,
        ](a, b)
    else:
        return a + b


@always_inline
def _sub_rn_d(a: Float64, b: Float64) -> Float64:
    """sub.rn.f64, unfusable (see `_add_rn_d`)."""
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "sub.rn.f64 $0, $1, $2;",
            Float64,
            constraints="=d,d,d",
            has_side_effect=False,
        ](a, b)
    elif is_amd_gpu():
        return inlined_assembly[
            "v_add_f64 $0, $1, -$2",
            Float64,
            constraints="=v,v,v",
            has_side_effect=False,
        ](a, b)
    else:
        return a - b


@always_inline
def _mul_rn_d(a: Float64, b: Float64) -> Float64:
    """mul.rn.f64: a product neither LLVM nor ptxas may fuse into the add
    that consumes it (Mojo's GPU code is contractible down to ptxas)."""
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "mul.rn.f64 $0, $1, $2;",
            Float64,
            constraints="=d,d,d",
            has_side_effect=False,
        ](a, b)
    elif is_amd_gpu():
        return inlined_assembly[
            "v_mul_f64 $0, $1, $2",
            Float64,
            constraints="=v,v,v",
            has_side_effect=False,
        ](a, b)
    else:
        return a * b


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
    if a < 1.1754944e-38:  # 2**-126
        m = a * 8388608.0  # 2**23
        i = Float32(-23.0)

    var e = (_fb(m) - Int32(1059760811)) & Int32(-8388608)  # bits(2/3f)
    var f = _fr(_fb(m) - e)
    i = fma(e.cast[DType.float32](), Float32(1.1920929e-7), i)  # 2**-23

    var t = f - Float32(1.0)
    var p = fma(Float32(-0.13018857), t, Float32(0.1408461))
    p = fma(p, t, Float32(-0.12148628))
    p = fma(p, t, Float32(0.1398061))
    p = fma(p, t, Float32(-0.16684236))
    p = fma(p, t, Float32(0.200123))
    p = fma(p, t, Float32(-0.24999669))
    p = fma(p, t, Float32(0.33333182))
    p = fma(p, t, Float32(-0.5))
    p = p * t
    p = fma(p, t, t)
    var r = fma(i, Float32(0.6931472), p)  # ln(2)

    if m.to_bits[DType.uint32]() >= UInt32(0x7F800000):
        r = fma(m, _F32_INF, _F32_INF)
    if m == Float32(0.0):
        r = -_F32_INF
    return r


# --------------------------------------------------------------------------- #
# __nv_log2f
# --------------------------------------------------------------------------- #


@always_inline
def nv_log2f(a: Float32) -> Float32:
    """`__nv_log2f` (libdevice_ir/__nv_log2f.ll). Pure IEEE: exact everywhere.
    """
    var m = a
    var i = Float32(0.0)
    if a < 1.1754944e-38:  # 2**-126
        m = a * 8388608.0  # 2**23
        i = Float32(-23.0)

    var e = (_fb(m) - Int32(1060439283)) & Int32(-8388608)  # bits(sqrt(.5f))
    var f = _fr(_fb(m) - e)
    i = fma(e.cast[DType.float32](), Float32(1.1920929e-7), i)  # 2**-23

    var t = f - Float32(1.0)
    var p = fma(Float32(0.09702014), t, Float32(-0.16845393))
    p = fma(p, t, Float32(0.1716887))
    p = fma(p, t, Float32(-0.17900948))
    p = fma(p, t, Float32(0.20512372))
    p = fma(p, t, Float32(-0.24046533))
    p = fma(p, t, Float32(0.28857115))
    p = fma(p, t, Float32(-0.36067417))
    p = fma(p, t, Float32(0.4808982))
    p = fma(p, t, Float32(-0.7213475))
    p = p * t * t
    var r = fma(t, Float32(1.442695), p) + i  # log2(e)

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
def nv_log1pf[rare_special: Bool = False](a: Float32) -> Float32:
    """`__nv_log1pf` (libdevice_ir/__nv_log1pf.ll). Pure IEEE plus add.rz.

    `rare_special`: for callers whose inputs are almost never negative, inf
    or NaN (asinh): those cases become a branch around a call instead of
    selects every element pays for, as in CUDA's SASS. A caller that takes
    them often (log_ndtr, -erfc / 2 < 0) runs slower that way.
    """
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
        comptime if rare_special:
            r = _log1pf_special_call(a, r)
        else:
            r = _log1pf_special(a, r)
    return r


@always_inline
def _log1pf_special(a: Float32, r: Float32) -> Float32:
    """nv_log1pf's negative, inf and NaN inputs."""
    var s = r
    if _fb(a) > Int32(-1082130432):  # signed bits(-1.0f)
        s = fma(a, _F32_INF, _F32_INF)
    if a == Float32(0.0):
        s = Float32(-0.0)
    return s


@no_inline
def _log1pf_special_call(a: Float32, r: Float32) -> Float32:
    """A call LLVM cannot speculate, so the branch to it stays a branch."""
    return _log1pf_special(a, r)


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
    """`__nv_log` (libdevice_ir/__nv_log.ll) on every GPU (see the header:
    std.math's float64 log is float32-accurate); std.math on the host."""
    comptime if not is_gpu():
        return _std_log(a)
    return _log_core(a)


@always_inline
def nv_log1p(a: Float64) -> Float64:
    """`__nv_log1p` (libdevice_ir/__nv_log1p.ll) on every GPU; std.math on
    the host."""
    comptime if not is_gpu():
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
    comptime if is_nvidia_gpu():
        if nz == UInt32(64):  # PTX clamps the shift: hi2 << 64 | lo2 >> 0
            hi3 = lo2
        elif nz != UInt32(0):
            hi3 = (hi2 << _z64(nz)) | (lo2 >> _z64(UInt32(64) - nz))
    else:
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
        var r0 = _rcp_approx_ftz_d(r)
        var nr = fma(-r, r0, Float64(1.0))
        nr = fma(nr, nr, nr)
        var rr = -fma(nr, r0, r0)
        var w = fma(r, rr, Float64(1.0))
        w = fma(rr, d, w)
        r = fma(w, rr, rr)
    return r


# --------------------------------------------------------------------------- #
# __nv_sincos
# --------------------------------------------------------------------------- #


@always_inline
def nv_sincos(a: Float64) -> Tuple[Float64, Float64]:
    """`__nv_sincos` (libdevice.10.bc): (sin(a), cos(a)) from `__nv_tan`'s
    Cody-Waite / Payne-Hanek reduction and `__nv_sincospi`'s polynomials.

    Pure IEEE apart from cvt.rni (`_d2i_rn` has a portable fallback): exact
    everywhere.
    """
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
    var c = fma(
        Float64(from_bits=UInt64(0xBDA8FF8320FD8164)),
        x2,
        Float64(from_bits=UInt64(0x3E21EEA7C1EF8528)),
    )
    c = fma(c, x2, Float64(from_bits=UInt64(0xBE927E4F8E06E6D9)))
    c = fma(c, x2, Float64(from_bits=UInt64(0x3EFA01A019DDBCE9)))
    c = fma(c, x2, Float64(from_bits=UInt64(0xBF56C16C16C15D47)))
    c = fma(c, x2, Float64(from_bits=UInt64(0x3FA5555555555551)))
    c = fma(c, x2, Float64(-0.5))
    c = fma(c, x2, Float64(1.0))

    var sp = fma(
        Float64(from_bits=UInt64(0x3DE5DB65F9785EBA)),
        x2,
        Float64(from_bits=UInt64(0xBE5AE5F12CB0D246)),
    )
    sp = fma(sp, x2, Float64(from_bits=UInt64(0x3EC71DE369ACE392)))
    sp = fma(sp, x2, Float64(from_bits=UInt64(0xBF2A01A019DB62A1)))
    sp = fma(sp, x2, Float64(from_bits=UInt64(0x3F81111111110818)))
    sp = fma(sp, x2, Float64(from_bits=UInt64(0xBFC5555555555554)))
    sp = fma(sp, x2, Float64(0.0))
    var s = fma(sp, z, z)

    var sv = s
    var cv = c
    if (i & Int32(1)) != Int32(0):
        sv = c
        cv = _i2d(_lo(s), _hi(s) ^ Int32(-2147483648))
    if (i & Int32(2)) != Int32(0):
        sv = _i2d(_lo(sv), _hi(sv) ^ Int32(-2147483648))
        cv = _i2d(_lo(cv), _hi(cv) ^ Int32(-2147483648))
    return (sv, cv)


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


# --------------------------------------------------------------------------- #
# __nv_asinhf / __nv_asinh
# --------------------------------------------------------------------------- #


@always_inline
def _rsqrt_approx_ftz_f(x: Float32) -> Float32:
    return inlined_assembly[
        "rsqrt.approx.ftz.f32 $0, $1;",
        Float32,
        constraints="=f,f",
        has_side_effect=False,
    ](x)


@always_inline
def _rsqrt_approx_ftz_d(x: Float64) -> Float64:
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "rsqrt.approx.ftz.f64 $0, $1;",
            Float64,
            constraints="=d,d",
            has_side_effect=False,
        ](x)
    else:
        return Float64(1.0) / _std_sqrt(x)


@always_inline
def nv_asinhf(a: Float32) -> Float32:
    """`__nv_asinhf` of CUDA 13's libdevice (the nvidia-nvvm 13.0 wheel; the
    CUDA 12 one is another algorithm), what torch's cu13 wheels run. NVIDIA
    (rsqrt.approx, rcp.approx) or host (libm) only."""
    comptime assert is_nvidia_gpu() or not is_gpu(), "NVIDIA or host only"
    comptime if not is_nvidia_gpu():
        return _std_asinh(a)
    var x = _fabs_f(a)
    var s = fma(a, a, Float32(1.0))
    var d = fma(s, _rsqrt_approx_ftz_f(s), Float32(1.0))  # 1 + sqrt(1 + a^2)
    var t = fma(x, x * _rcp_approx_ftz_f(d), x)
    if x > Float32(8388608.0):  # 2**23: asinh(x) = log1p(x) + ln(2)
        t = x
    var r = nv_log1pf[rare_special=True](t)
    if x > Float32(8388608.0):
        r = r + Float32(0.6931472)
    if x <= _F32_INF:  # not NaN
        r = _fr((_fb(r) & Int32(0x7FFFFFFF)) | (_fb(a) & Int32(-2147483648)))
    return r


@always_inline
def nv_asinh(a: Float64) -> Float64:
    """`__nv_asinh` of CUDA 13's libdevice (see nv_asinhf), with its inlined
    __nv_log1p. NVIDIA (rsqrt.approx, rcp.approx) or host (libm) only."""
    comptime assert is_nvidia_gpu() or not is_gpu(), "NVIDIA or host only"
    comptime if not is_nvidia_gpu():
        return _std_asinh(a)
    var ahi = _hi(a) & Int32(0x7FFFFFFF)
    var x = _i2d(_lo(a), ahi)
    var s = fma(a, a, Float64(1.0))
    # rsqrt(s), refined once
    var y = _rsqrt_approx_ftz_d(s)
    var e = fma(s, -(y * y), Float64(1.0))
    y = fma(fma(Float64(0.375), e, Float64(0.5)), e * y, y)
    # 1 / (1 + sqrt(s)), refined
    var d = fma(s, y, Float64(1.0))
    var r0 = _rcp_approx_ftz_d(d)
    var nr = fma(-d, r0, Float64(1.0))
    var rc = fma(fma(nr, nr, nr), r0, r0)
    var t = fma(x, x * rc, x)
    var big = _u32(ahi) >= UInt32(1138753536)  # 2**63, inf and NaN too
    if big:
        t = x
    var r = nv_log1p(t)
    if big:
        r = _add_rn_d(r, Float64(0.6931471805599453))  # + ln(2)
    return _i2d(
        _lo(r), (_hi(r) & Int32(0x7FFFFFFF)) | (_hi(a) & Int32(-2147483648))
    )


# --------------------------------------------------------------------------- #
# CUDA 13 libdevice double routines of the special functions: lgamma, erfc,
# erfinv, pow, acos, sin, cos. Mojo's GPU code is contractible down to ptxas;
# CUDA contracts only where its SASS shows it (`_accurate_pow`), so every
# other `*` whose product a `+`/`-` consumes is `_mul_rn_d` (or the sum
# `_add_rn_d`/`_sub_rn_d`). `a + (-b)` is `_sub_rn_d(a, b)`: CUDA's fneg is a
# DADD operand modifier, which a NaN passes through unflipped, where Mojo's
# fneg into inline asm is a sign-bit xor. A routine returning a plain product
# (nv_erfinv) can still fuse with the caller's next add.
# Bit-exact against CUDA 13 on sm_90 (NaN payloads included); other GPUs and
# the host run the same algorithm with IEEE stand-ins for the PTX
# approximations (rcp/rsqrt seeds) and their own contraction: within an ulp.
# --------------------------------------------------------------------------- #


@always_inline
def _d(bits: UInt64) -> Float64:
    return Float64(from_bits=bits)


@always_inline
def _poly[*C: UInt64](x: Float64) -> Float64:
    """Horner: ((C0 x + C1) x + C2) ... with an fma per step."""
    var p = _d(C[0])
    comptime for i in range(1, len(C)):
        p = fma(p, x, _d(C[i]))
    return p


@always_inline
def _rcp_d(x: Float64) -> Float64:
    """1 / x as libdevice computes it: rcp.approx.ftz.f64, refined."""
    var r0 = _rcp_approx_ftz_d(x)
    var e = fma(-x, r0, Float64(1.0))
    e = fma(e, e, e)
    return fma(e, r0, r0)


@always_inline
def _sdiv2(j: Int32) -> Int32:
    """`sdiv i32 j, 2` (rounds toward zero)."""
    return (j + ((j >> Int32(31)) & Int32(1))) >> Int32(1)


@always_inline
def _sin_cos_kernel_d(z: Float64, i: Int32) -> Float64:
    """`__internal_sin_cos_kerneld` (the non-sm_35/37 arm): sin(z) for an
    even quadrant i, cos(z) for an odd one, from `__cudart_sin_cos_coeffs`;
    negated when i & 2."""
    var odd = (i & Int32(1)) != Int32(0)
    var z2 = z * z
    var p = fma(
        _d(0xBDA8FF8320FD8164) if odd else _d(0x3DE5DB65F9785EBA),
        z2,
        _d(0x3E21EEA7C1EF8528) if odd else _d(0xBE5AE5F12CB0D246),
    )
    p = fma(p, z2, _d(0xBE927E4F8E06E6D9) if odd else _d(0x3EC71DE369ACE392))
    p = fma(p, z2, _d(0x3EFA01A019DDBCE9) if odd else _d(0xBF2A01A019DB62A1))
    p = fma(p, z2, _d(0xBF56C16C16C15D47) if odd else _d(0x3F81111111110818))
    p = fma(p, z2, _d(0x3FA5555555555551) if odd else _d(0xBFC5555555555554))
    p = fma(p, z2, Float64(-0.5) if odd else Float64(0.0))
    var r = fma(p, z2, Float64(1.0)) if odd else fma(p, z, z)
    if (i & Int32(2)) != Int32(0):
        r = fma(r, Float64(-1.0), Float64(0.0))
    return r


@always_inline
def _trig_reduce_d(a: Float64) -> Tuple[Float64, Int32]:
    """`__internal_trig_reduction_kerneld` with __nv_sin's infinity test:
    a - q pi/2 (three-part Cody-Waite, Payne-Hanek from 2**31 on) and q."""
    if _fabs_d(a) == _F64_INF:
        return (_mul_rn_d(a, Float64(0.0)), Int32(0))
    var q = _d2i_rn(a * _d(0x3FE45F306DC9C883))  # 2/pi
    var qf = q.cast[DType.float64]()
    var t = fma(-qf, _d(0x3FF921FB54442D18), a)
    t = fma(-qf, _d(0x3C91A62633145C00), t)
    t = fma(-qf, _d(0x397B839A252049C0), t)
    if _fabs_d(a) >= _d(0x41E0000000000000):  # 2**31
        return _trig_reduction_slowpath_d(a, q)
    return (t, q)


@always_inline
def nv_sin(a: Float64) -> Float64:
    """`__nv_sin` of CUDA 13's libdevice (the non-sm_35/37 arm)."""
    var r = _trig_reduce_d(a)
    return _sin_cos_kernel_d(r[0], r[1])


@always_inline
def nv_cos(a: Float64) -> Float64:
    """`__nv_cos` of CUDA 13's libdevice (the non-sm_35/37 arm)."""
    var r = _trig_reduce_d(a)
    return _sin_cos_kernel_d(r[0], r[1] + Int32(1))


# --------------------------------------------------------------------------- #
# __nv_lgamma
# --------------------------------------------------------------------------- #


@always_inline
def _lgamma_pos(a: Float64) -> Float64:
    """`__internal_lgamma_pos`: lgamma(a) for a >= +0 or +inf."""
    var hi = _hi(a)
    if hi >= Int32(0x40080000):  # 3
        if hi >= Int32(0x40200000):  # 8: Stirling, in 1/a
            var r = _rcp_d(a)
            var s = _poly[
                0xBF5AC321034783F9,
                0x3F4B68B992738FBF,
                0xBF4380D01E4F7B8C,
                0x3F4A019FA29F7264,
                0xBF66C16C16B2ACEC,
                0x3FB5555555555545,
            ](r * r)
            s = fma(s, r, _d(0x3FED67F1C864BEAE))
            var l = _log_core(a)
            var hl = _i2d(_lo(l), _hi(l) - Int32(0x100000))  # log(a) / 2
            var am = a - Float64(0.5)
            var t = fma(hl, am, -a) + fma(hl, am, s)
            return a if a == _F64_INF else t
        var x = a - Float64(3.0)
        var num = _poly[
            0xC0AF7040BB18FB05,
            0xC1122B7730207EF3,
            0xC1585A0DB81DE7D0,
            0xC18A992B8BA94677,
            0xC1AAC5CB6957CC20,
            0xC1BC0E2B308774BE,
            0xC1C6BA13DCAE7F67,
            0xC1CCF33B9C3D120C,
        ](x)
        var den = x - _d(0x408FF62E0BE189FE)
        den = fma(den, x, _d(0xC10074FACE10C93F))
        den = fma(den, x, _d(0xC151B662F8D75791))
        den = fma(den, x, _d(0xC18EE64AB4D207F7))
        den = fma(den, x, _d(0xC1B9051687C9951A))
        den = fma(den, x, _d(0xC1D2B866BF0B853D))
        den = fma(den, x, _d(0xC1D4E2130E9DC133))
        return num / den + x
    if hi >= Int32(0x3FF80000):  # 1.5
        var x = a - Float64(2.0)
        return _mul_rn_d(
            _poly[
                0x3E452636124338B3,
                0xBE71FA71D78C0EE2,
                0x3E8D111F31E61306,
                0xBEA0502BBE1B2706,
                0x3EB06850B2970292,
                0xBEC108474875033D,
                0x3ED24ACCC62909DC,
                0xBEE3CB25209E63BE,
                0x3EF581CBBC8CDC7B,
                0xBF078E04B85C7597,
                0x3F1A12730CF45051,
                0xBF2D3FD354062012,
                0x3F40B36B0B4DE323,
                0xBF538AC5C6D0317A,
                0x3F67ADD6EAAB19FC,
                0xBF7E404FC20E4D5B,
                0x3F951322AC7DA390,
                0xBFB13E001A5578A3,
                0x3FD4A34CC4A60FA3,
                0x3FDB0EE6072093CF,
            ](x),
            x,
        )
    if hi >= Int32(0x3FE66666):  # 0.7
        var x = Float64(1.0) - a
        return _mul_rn_d(
            _poly[
                0x3F881F6D2A4C4310,
                0x3FA3EB504359EB88,
                0x3FAE35D8DEB06317,
                0x3FAED469A8B6ECCE,
                0x3FACC1B1C357BEFE,
                0x3FAD7154DB67F79F,
                0x3FAFCC622CF2F7BB,
                0x3FB11747A4D1CC43,
                0x3FB24CE16A21B8AC,
                0x3FB3B1C21A7BCB00,
                0x3FB556723452ED57,
                0x3FB748C00891544F,
                0x3FB9A0207808CF40,
                0x3FBC80673B8AE26B,
                0x3FC010B364B7E555,
                0x3FC2703A1D239658,
                0x3FC5B40CB1137E6E,
                0x3FCA8B9C17AC4F03,
                0x3FD151322AC7CB52,
                0x3FD9A4D55BEAB1D4,
                0x3FEA51A6625307D6,
                0x3FE2788CFC6FB619,
            ](x),
            x,
        )
    var p = _poly[
        0xBE7844988BFE6590,
        0x3EA7B77CEB0625E8,
        0xBE998C69C8710CC4,
        0xBEF6527A5A11CF6E,
        0x3F20EC2950B1B5DE,
        0xBF2C4D80C24BA278,
        0xBF5315B4E8CC0D09,
        0x3F7D917F15D50020,
        0xBF83B4ABB41CB6FA,
        0xBFA59AF1275B7120,
        0x3FC5512321A168A0,
        0xBFA5815E8FDCE74C,
        0xBFE4FCF4026ADD1A,
        0x3FE2788CFC6FB5C8,
    ](a)
    return -_log_core(fma(p * a, a, a))  # -log(1 / Gamma(a))


@always_inline
def _sinpi_d(x: Float64) -> Float64:
    """`__nv_sinpi` as inlined in __nv_lgamma (non-sm_35/37 arm), x finite."""
    var two_x = _i2d(_lo(x), _hi(x) + Int32(0x100000))
    var rn = _rint_d(two_x)
    var i = rn.cast[DType.int64]().cast[DType.int32]()
    var t = fma(-rn, Float64(0.5), x)
    var r = fma(t, _d(0x400921FB54442D18), t * _d(0x3CA1A62633145C07))
    var s = _sin_cos_kernel_d(r, i)
    if x == _trunc_d(x):
        s = _mul_rn_d(x, Float64(0.0))
    return s


@always_inline
def nv_lgamma(a: Float64) -> Float64:
    """`__nv_lgamma` of CUDA 13's libdevice: `_lgamma_pos(|a|)`, and for a
    negative non-integer log(pi / |x sinpi(x)|) - lgamma(x)."""
    var x = _fabs_d(a)
    if not (x <= _F64_INF):
        return a + a
    var t = _lgamma_pos(x)
    if _hi(a) >= Int32(0):
        return t
    if x == _trunc_d(x):
        return _F64_INF
    var tiny = _hi(x) < Int32(0x3C000000)  # 2**-63
    var u = x
    if not tiny:
        u = _d(0x400921FB54442D18) / (_fabs_d(_sinpi_d(x)) * x)
    var l = _log_core(u)
    return -l if tiny else _sub_rn_d(l, t)


# --------------------------------------------------------------------------- #
# __nv_erfc
# --------------------------------------------------------------------------- #


@always_inline
def nv_erfc(a: Float64) -> Float64:
    """`__nv_erfc` of CUDA 13's libdevice: a rational approximation of
    erfcx(|a|) in (|a| - 4) / (|a| + 4) times a compensated exp(-a^2),
    2 - erfc(|a|) for negative a."""
    var hi = _hi(a)
    var ahi = hi & Int32(0x7FFFFFFF)
    var alo = _lo(a)
    if ahi >= Int32(0x7FF00000):
        if ahi == Int32(0x7FF00000) and alo == Int32(0):
            return Float64(2.0) if hi < Int32(0) else Float64(0.0)
        return a + a
    var x = _i2d(alo, ahi)
    var y = _erfcx_kernel(x)
    # exp(-x^2) as __nv_exp, the rounding of x^2 corrected
    var mx2 = (-x) * x
    var kk = fma(mx2, _d(0x3FF71547652B82FE), _d(0x4338000000000000))
    var j = _lo(kk)
    var kf = _add_rn_d(kk, _d(0xC338000000000000))
    var rr = fma(kf, _d(0xBFE62E42FEFA39EF), mx2)
    rr = fma(kf, _d(0xBC7ABC9E3B39803F), rr)
    var e = _poly[
        0x3E5ADE1569CE2BDF,
        0x3E928AF3FCA213EA,
        0x3EC71DEE62401315,
        0x3EFA01997C89EB71,
        0x3F2A01A014761F65,
        0x3F56C16C1852B7AF,
        0x3F81111111122322,
        0x3FA55555555502A1,
        0x3FC5555555555511,
        0x3FE000000000000B,
        0x3FF0000000000000,
        0x3FF0000000000000,
    ](rr)
    var jh = _sdiv2(j)
    var s1 = _i2d(_lo(e), _hi(e) + (jh << Int32(20)))
    var s2 = _i2d(Int32(0), Int32(0x3FF00000) + ((j - jh) << Int32(20)))
    var ex = s1 * s2
    ex = fma(ex, fma(-x, x, -mx2), ex)
    var t1 = _mul_rn_d(y, ex)
    if ahi > Int32(0x403B4000):  # 27.25
        t1 = Float64(0.0)
    if hi < Int32(0):
        t1 = _sub_rn_d(Float64(2.0), t1)
    return t1


# --------------------------------------------------------------------------- #
# __nv_erfinv
# --------------------------------------------------------------------------- #


@always_inline
def nv_erfinv(a: Float64) -> Float64:
    """`__nv_erfinv` of CUDA 13's libdevice: w = -log(1 - a^2) (an inlined
    log), then a polynomial in w - 3.125, sqrt(w) - 3.25 or sqrt(w) - 5,
    times a."""
    var r: Float64
    if _fabs_f(_fr(_hi(a))) < _fr(Int32(0x3FF00000)):  # |a| < 1
        var w = fma(a, -a, Float64(1.0))
        var whi = _hi(w)
        var e = _i32((_u32(whi) >> UInt32(20)) & UInt32(2046)) - Int32(1022)
        var ef = _i2d(e ^ Int32(-2147483648), Int32(0x43300000)) - _i2d(
            Int32(-2147483648), Int32(0x43300000)
        )
        var m = _i2d(_lo(w), (whi & Int32(-2145386497)) + Int32(0x3FE00000))
        var u = m - Float64(1.0)
        var rc = _rcp_d(m + Float64(1.0))
        var q = u * rc
        var s = fma(Float64(-2.0), q, u)
        s = fma(-q, u, s)
        var z = fma(s, rc, q)
        var z2 = z * z
        var p = _poly[
            0x3FB5C5C218C775C9,
            0x3FA55CF59CDC5D89,
            0x3FAEFD18CF6EBB9C,
            0x3FB10682EDCB8D1B,
            0x3FB3B1DD3AC7FC96,
            0x3FB745CB459B54A6,
            0x3FBC71C741A0669F,
            0x3FC249249209112E,
            0x3FC99999999A06C1,
            0x3FD5555555555535,
        ](z2)
        var l = fma(p * z2, z, z)
        var lg = fma(ef, _d(0x3FE62E42FEFA39EF), l + l)  # log(1 - a^2)
        if _u32(_hi(lg)) < UInt32(0xC0190000):  # -lg < 6.25
            r = _poly[
                0xBBB135D2E746E627,
                0xBC08DDF93324D327,
                0x3C37B83EEF0B7C9F,
                0x3C69BA72CD589B91,
                0xBCA33689090A6B96,
                0x3C782E11898132E0,
                0x3CFDE4ACFD9E26BA,
                0xBD26D33EED66C487,
                0xBD36F2167040D8E2,
                0x3D872A22C2D77E20,
                0xBDAC8859C4E5C0AF,
                0xBDCDC583D118A561,
                0x3E120F47CCF46B3C,
                0xBE31A9E38DC84D60,
                0xBE5F36CD6D3D46A9,
                0x3E9C6B4F5D03B787,
                0xBEB6E8A5434AE8A2,
                0xBEED1D1F7B8736F6,
                0x3F2879C2A212F024,
                0xBF4845769484FCA8,
                0xBF78B6C33114F909,
                0x3FCEBD80D9B13E28,
                0x3FFA755E7C99AE86,
            ](-lg - Float64(3.125))
        else:
            var sq = _std_sqrt(-lg)
            if _hi(sq) < Int32(0x40100000):  # 4
                r = _poly[
                    0x3E23040F87DBD932,
                    0x3E785CBE52878635,
                    0xBE92777453DD3955,
                    0x3E5395ABCD554C6C,
                    0x3EB936388A3790AD,
                    0xBED0D5DB812B5083,
                    0x3EC8860CD5D652F6,
                    0x3EEA29A0CACDFB23,
                    0xBF08CEF1F80281F2,
                    0x3F11E684D0B9188A,
                    0x3EF932CD54C8A222,
                    0xBF37448A89EF8AA3,
                    0x3F4F3CC55AD40C25,
                    0xBF5BA924132F38B1,
                    0x3F6468EECA533CF8,
                    0xBF6EBADABB891BBD,
                    0x3F75FFCFE5B76AFC,
                    0x3FF0158A6D641D39,
                    0x4008ABCC380D5A48,
                ](sq - Float64(3.25))
            else:
                r = _poly[
                    0xBDBDCEC3A7785389,
                    0xBDF18FEEC0E38727,
                    0x3E19E6BF2DDA45E3,
                    0xBE30468FB24E2F5F,
                    0x3E405AC6A8FBA182,
                    0xBE50102E495FB9C0,
                    0x3E5F4C20E1334AF8,
                    0xBE722D220FDF9C3E,
                    0x3E8EBC8BB824CB54,
                    0xBEB0A8D40EA372CC,
                    0x3ED2FBD29D093D2B,
                    0xBEF4A3497E1E0FAC,
                    0x3F13EBF4EB00938F,
                    0xBF2C2F36A8FC5D53,
                    0xBF222EA5DF04047C,
                    0x3FF02A30D1FBA0DC,
                    0x4013664DDD1AD7FB,
                ](sq - Float64(5.0))
    elif not (_fabs_d(a) <= _F64_INF):
        r = a
    elif _fabs_d(a) == Float64(1.0):
        r = _F64_INF
    else:
        r = _d(0xFFF8000000000000)
    return r * a


# --------------------------------------------------------------------------- #
# __nv_pow
# --------------------------------------------------------------------------- #


@always_inline
def _accurate_pow(a: Float64, b: Float64) -> Float64:
    """`__internal_accurate_pow`: exp(b log a) for a >= 0, the log in
    double-double. Every two-sum is spelled with `_add_rn_d`/`_sub_rn_d` so
    nothing can reassociate it away."""
    var ihi = _hi(a)
    var ilo = _lo(a)
    var expo = _i32(_u32(ihi) >> UInt32(20))
    if expo == Int32(0):
        var s = a * _d(0x4350000000000000)  # 2**54
        ihi = _hi(s)
        ilo = _lo(s)
        expo = _i32(_u32(ihi) >> UInt32(20)) - Int32(54)
    var e = expo - Int32(1023)
    var mhi = (ihi & Int32(-2146435073)) | Int32(0x3FF00000)
    var m = _i2d(ilo, mhi)
    if _u32(mhi) > UInt32(1073127582):  # m > sqrt(2)
        m = _i2d(_lo(m), _hi(m) - Int32(0x100000))
        e = e + Int32(1)
    var u = m - Float64(1.0)
    var rc = _rcp_d(m + Float64(1.0))
    var q = _mul_rn_d(u, rc)
    var w = _add_rn_d(q, q)
    var w2 = w * w
    var p = _poly[
        0x3EB0F5FF7D2CAFE2,
        0x3ED0F5D241AD3B5A,
        0x3EF3B20A75488A3F,
        0x3F1745CDE4FAECD5,
        0x3F3C71C7258A578B,
        0x3F6249249242B910,
        0x3F89999999999DFB,
    ](w2)
    var cc = fma(-w, u, Float64(2.0) * (u - w))
    var lo45 = _mul_rn_d(rc, cc)
    # (c_lo, c_hi) = 1/12 + p w2, two-sum, plus the 1/12 tail. CUDA's SASS
    # contracts both adds of the product p w2 into fmas (and q + q into
    # fma(u, rc, q), which equals 2q).
    var c3 = _d(0x3FB5555555555555)
    var s46 = fma(p, w2, c3)
    var s48 = fma(p, w2, _sub_rn_d(c3, s46))
    var s50 = _add_rn_d(_add_rn_d(s48, Float64(0.0)), _d(0xBC46A4CB00B9E7B0))
    var c_hi = _add_rn_d(s46, s50)
    var c_lo = _add_rn_d(_sub_rn_d(s46, c_hi), s50)
    # w^3 in double-double
    var ww = _mul_rn_d(w, w)
    var ww_lo = fma(w, w, -ww)
    ww_lo = fma(w, _i2d(_lo(lo45), _hi(lo45) + Int32(0x100000)), ww_lo)
    var www = _mul_rn_d(ww, w)
    var www_lo = fma(ww, w, -www)
    www_lo = fma(ww, lo45, www_lo)
    www_lo = fma(ww_lo, w, www_lo)
    # c * w^3
    var t_hi = _mul_rn_d(c_hi, www)
    var t_lo = fma(c_hi, www, -t_hi)
    t_lo = fma(c_hi, www_lo, t_lo)
    t_lo = fma(c_lo, www, t_lo)
    var v_hi = _add_rn_d(t_hi, t_lo)
    var v_lo = _add_rn_d(_sub_rn_d(t_hi, v_hi), t_lo)
    # log(m) = w + lo45 + c w^3
    var l80 = _add_rn_d(w, v_hi)
    var l84 = _add_rn_d(
        _add_rn_d(_add_rn_d(_sub_rn_d(w, l80), v_hi), v_lo), lo45
    )
    var lm_hi = _add_rn_d(l80, l84)
    var lm_lo = _add_rn_d(_sub_rn_d(l80, lm_hi), l84)
    # + e ln 2
    var ef = _i2d(e ^ Int32(-2147483648), Int32(0x43300000)) - _i2d(
        Int32(-2147483648), Int32(0x43300000)
    )
    var l94 = fma(ef, _d(0x3FE62E42FEFA39EF), lm_hi)
    var l96 = fma(-ef, _d(0x3FE62E42FEFA39EF), l94)
    var l98 = _sub_rn_d(lm_lo, _sub_rn_d(l96, lm_hi))
    var l99 = fma(ef, _d(0x3C7ABC9E3B39803F), l98)
    var lg_hi = _add_rn_d(l94, l99)
    var lg_lo = _add_rn_d(_sub_rn_d(l94, lg_hi), l99)
    # b * log(a), b scaled down when huge so the product cannot overflow
    var bhi = _hi(b)
    if _u32(bhi + bhi) >= UInt32(0xFE000000):
        bhi = bhi & Int32(-15728641)
    var bb = _i2d(_lo(b), bhi)
    var y_hi = _mul_rn_d(lg_hi, bb)
    var y_lo = fma(lg_hi, bb, -y_hi)
    y_lo = fma(lg_lo, bb, y_lo)
    var y = _add_rn_d(y_hi, y_lo)
    var y_tail = _add_rn_d(_sub_rn_d(y_hi, y), y_lo)
    var z = nv_exp(y)
    if _fabs_d(z) == _F64_INF:
        return z
    return fma(z, y_tail, z)


@always_inline
def nv_pow(a: Float64, b: Float64) -> Float64:
    """`__nv_pow` of CUDA 13's libdevice, every C99 special case included."""
    var ahi = _hi(a)
    var bhi = _hi(b)
    # b odd integer: its units bit is the top bit once shifted left by
    # exponent - 1012; PTX clamps an out-of-range shift to 0.
    var sh = _u32(((bhi & Int32(0x7FF00000)) >> Int32(20)) - Int32(1012))
    var bits = b.to_bits[DType.uint64]()
    var odd = sh < UInt32(64) and (bits << _z64(sh)) == UInt64(
        0x8000000000000000
    )
    var t = _accurate_pow(_fabs_d(a), b)
    if ahi < Int32(0) and odd:
        t = -t
    if a == Float64(0.0):
        if _fabs_d(b) == Float64(0.5):
            odd = False
        var thi = ahi if odd else Int32(0)
        if bhi < Int32(0):
            thi = thi | Int32(0x7FF00000)
        t = _i2d(Int32(0), thi)
    elif ahi < Int32(0) and not (b == _trunc_d(b)):
        t = _d(0xFFF8000000000000)
    if (_hi(a + b) & Int32(0x7FF00000)) == Int32(0x7FF00000):
        if not (_fabs_d(a) <= _F64_INF) or not (_fabs_d(b) <= _F64_INF):
            t = _add_rn_d(a, b)
        elif _fabs_d(b) == _F64_INF:
            var thi = Int32(0x7FF00000) if _fabs_d(a) > Float64(1.0) else Int32(
                0
            )
            if bhi < Int32(0):
                thi = thi ^ Int32(0x7FF00000)
            if a == Float64(-1.0):
                thi = Int32(0x3FF00000)
            t = _i2d(Int32(0), thi)
        elif _fabs_d(a) == _F64_INF:
            var thi = Int32(0x7FF00000) if bhi >= Int32(0) else Int32(0)
            if (
                ahi < Int32(0)
                and odd
                and (bhi & Int32(0x7FFFFFFF)) != Int32(0x3FE00000)
            ):
                thi = thi ^ Int32(-2147483648)
            t = _i2d(Int32(0), thi)
    if a == Float64(1.0) or b == Float64(0.0):
        t = Float64(1.0)
    return t


# --------------------------------------------------------------------------- #
# __nv_acos
# --------------------------------------------------------------------------- #


@always_inline
def nv_acos(a: Float64) -> Float64:
    """`__nv_acos` of CUDA 13's libdevice: pi/2 - asin(|a|) by a polynomial
    below |a| = 0.575, else 2 asin(sqrt((1 - |a|) / 2)) with a refined
    rsqrt.approx.ftz.f64; reflected for negative a."""
    var hi = _hi(a)
    var x = _fabs_d(a)
    if _hi(x) < Int32(0x3FE26666):
        var x2 = x * x
        var p = _poly[
            0x3FB0066BDC1895E9,
            0xBFB3823B180754AF,
            0x3FB11E52CC2F79AE,
            0xBF924EAF3526861B,
            0x3F91DF02A31E6CB7,
            0x3F847D18B0EEC6CC,
            0x3F8D0AF961BA53B0,
            0x3F91BF7734CF1C48,
            0x3F96E91483144EF7,
            0x3F9F1C6E0A4F9F81,
            0x3FA6DB6DC27FA92B,
            0x3FB333333320F91B,
            0x3FC5555555555F4D,
        ](x2)
        var s = fma(p * x2, x, x)  # asin(|a|)
        if hi < Int32(0):
            return _add_rn_d(
                _d(0x3FF921FB54442D18), _add_rn_d(s, _d(0x3C91A62633145C07))
            )
        return _sub_rn_d(
            _d(0x3FF921FB54442D18), _add_rn_d(s, _d(0xBC91A62633145C07))
        )
    var d = Float64(1.0) - x
    var dh = _i2d(_lo(d), _hi(d) - Int32(0x100000))  # d / 2
    var rs = _rsqrt_approx_ftz_d(dh)
    var rh = _i2d(_lo(rs), _hi(rs) - Int32(0x100000))
    var s0 = dh * rs
    var s1 = fma(fma(s0, -s0, dh), rh, s0)
    var rh2 = fma(fma(rs, -s1, Float64(1.0)), rh, rh)
    var s2 = fma(fma(s1, -s1, dh), rh2, s1)  # sqrt(d / 2)
    var sq = _i2d(_lo(s2), _hi(s2) + Int32(0x100000))
    var p = _poly[
        0x3EC715B371155F70,
        0xBEBAC2FE66FAAC4B,
        0x3ED9A9B88EFCD9B8,
        0x3EDD0F40A8A0C4C3,
        0x3EF46D4CFA9E0E1F,
        0x3F079C168D1E2422,
        0x3F1C9A88C3BCA540,
        0x3F31C4E64BD476DF,
        0x3F46E8BA60009C8F,
        0x3F5F1C71C62B05A2,
        0x3F76DB6DB6DC9F2C,
        0x3F9333333333329C,
        0x3FB5555555555555,
    ](d)
    var dhi = _hi(d)
    var t: Float64
    if dhi <= Int32(0):
        t = _mul_rn_d(x, Float64(0.0))
    else:
        t = fma(p * d, sq, sq)
    if dhi < Int32(0):
        t = _mul_rn_d(t, _F64_INF)
    if hi < Int32(0):
        t = _sub_rn_d(
            _d(0x400921FB54442D18), _add_rn_d(t, _d(0xBCA1A62633145C07))
        )
    return t


# --------------------------------------------------------------------------- #
# __nv_erfcx
# --------------------------------------------------------------------------- #


@always_inline
def _erfcx_kernel(x: Float64) -> Float64:
    """erfcx(x) for 0 <= x < 32: the rational kernel of `__nv_erfc`."""
    var r = _rcp_d(x + Float64(4.0))
    var q = (x - Float64(4.0)) * r
    var f = fma(Float64(-4.0), _add_rn_d(q, Float64(1.0)), x)
    f = fma(-q, x, f)
    var z = fma(r, f, q)
    var p = _poly[
        0xBDF8774AD4E0BFD7,
        0xBE44E1C6FD03D328,
        0xBE4330149F7A56B6,
        0x3E7BEDDED8376273,
        0x3E6F9254C3ABF22B,
        0xBEAB9068C2148CF0,
        0x3E94C6454DB34009,
        0x3ED7F1C378F2311D,
        0xBEE78E051C6D5C58,
        0xBEF995B4EAD14A90,
        0x3F23BE27CF0A29B2,
        0xBF2A1DEF3E81672E,
        0xBF48D4ABE68C1713,
        0x3F749C67210DD6B4,
        0xBF9096238568E357,
        0x3FA3079EDF8C2DC9,
        0xBFB0FB06DFF601FC,
        0x3FB7FEE004DFBCDC,
        0xBFB9DDB23C3DB8C6,
        0x3FB16ECEFCFA5FDA,
        0x3F8F7F5DF66FB6D6,
        0xBFC1DF1AD154A29D,
        0x3FF3BA5916E9FD7F,
    ](z)
    var rd = _rcp_d(fma(Float64(2.0), x, Float64(1.0)))
    var g = p * rd
    var h = _sub_rn_d(fma(x, Float64(-2.0) * g, p), g)
    return fma(h, rd, g)


@always_inline
def nv_erfcx(a: Float64) -> Float64:
    """`__nv_erfcx` of CUDA 13's libdevice: `__nv_erfc`'s rational kernel
    below |a| = 32, an asymptotic series in 1/a above; for a < 0,
    2 exp(a^2) (the rounding of a^2 folded back in) minus that."""
    var hi = _hi(a)
    var t1: Float64
    if _fabs_f(_fr(hi)) < _fr(Int32(0x40400000)):  # |a| < 32
        t1 = _erfcx_kernel(_i2d(_lo(a), hi & Int32(0x7FFFFFFF)))
    else:
        var r = Float64(1.0) / a
        var r2 = r * r
        var p = _poly[
            0xC03D880000000000,
            0x401A400000000000,
            0xBFFE000000000000,
            0x3FE8000000000000,
            0xBFE0000000000000,
            0x3FF0000000000000,
        ](r2)
        t1 = _mul_rn_d(p, r * _d(0x3FE20DD750429B6D))  # 1/sqrt(pi)
    if hi >= Int32(0):
        return t1
    var x2 = a * a
    var x2lo = fma(a, a, -x2)
    var ex = nv_exp(x2)
    var e2 = _add_rn_d(ex, ex)
    if e2 == _F64_INF:
        return e2
    return _sub_rn_d(fma(e2, x2lo, e2), t1)


# --------------------------------------------------------------------------- #
# float64 unary routines of CUDA 13's libdevice (the nvidia-nvvm 13.0 wheel)
# that ATen's double kernels call: NVIDIA only (rsqrt / rcp.approx seeds),
# each a translation of the `__nv_*` body with `__nvvm_reflect` = 0. Sums
# whose rounding the algorithm relies on go through `_add_rn_d` (the GPU
# build's fast-math flags may reassociate or contract plain ones), and every
# NaN / infinity test is on the bits.
# --------------------------------------------------------------------------- #


comptime _D_MAGIC = Float64(from_bits=UInt64(0x4338000000000000))  # 1.5 * 2^52
comptime _D_L2E = Float64(from_bits=UInt64(0x3FF71547652B82FE))
comptime _D_LN2_HI = Float64(from_bits=UInt64(0x3FE62E42FEFA39EF))
comptime _D_LN2_LO = Float64(from_bits=UInt64(0x3C7ABC9E3B39803F))


@always_inline
def _dc(bits: UInt64) -> Float64:
    return Float64(from_bits=bits)


@always_inline
def _ub(x: Float64) -> UInt64:
    return x.to_bits[DType.uint64]()


@always_inline
def _is_nan_d(x: Float64) -> Bool:
    return (_ub(x) & UInt64(0x7FFFFFFFFFFFFFFF)) > UInt64(0x7FF0000000000000)


@always_inline
def _copysign_d(mag: Float64, sgn: Float64) -> Float64:
    return _dc(
        (_ub(mag) & UInt64(0x7FFFFFFFFFFFFFFF))
        | (_ub(sgn) & UInt64(0x8000000000000000))
    )


@always_inline
def _exp_poly_d(r: Float64) -> Float64:
    """`__nv_exp`'s polynomial: e^r for |r| <= ln(2) / 2."""
    var p = fma(_dc(0x3E5ADE1569CE2BDF), r, _dc(0x3E928AF3FCA213EA))
    p = fma(p, r, _dc(0x3EC71DEE62401315))
    p = fma(p, r, _dc(0x3EFA01997C89EB71))
    p = fma(p, r, _dc(0x3F2A01A014761F65))
    p = fma(p, r, _dc(0x3F56C16C1852B7AF))
    p = fma(p, r, _dc(0x3F81111111122322))
    p = fma(p, r, _dc(0x3FA55555555502A1))
    p = fma(p, r, _dc(0x3FC5555555555511))
    p = fma(p, r, _dc(0x3FE000000000000B))
    p = fma(p, r, Float64(1.0))
    return fma(p, r, Float64(1.0))


@always_inline
def nv_sinh(a: Float64) -> Float64:
    """`__nv_sinh`: an odd polynomial below |a| = 1, else u + u / (2u + 1)
    with u = expm1(|a|) / 2 (`__internal_expm1_scaled`), inf from
    |a| >= 710.4758600739439; signed by `a`."""
    var ahi = _hi(a) & Int32(0x7FFFFFFF)
    var x = _i2d(_lo(a), ahi)
    var z: Float64
    if _u32(ahi) < UInt32(1072693248):
        var x2 = x * x
        var p = fma(_dc(0x3D6B4C75AB274C53), x2, _dc(0x3DE611A561D87DEF))
        p = fma(p, x2, _dc(0x3E5AE64671B18F5C))
        p = fma(p, x2, _dc(0x3EC71DE3A465B1E4))
        p = fma(p, x2, _dc(0x3F2A01A01A02899D))
        p = fma(p, x2, _dc(0x3F811111111110A6))
        p = fma(p, x2, _dc(0x3FC5555555555556))
        p = p * x2
        z = fma(p, x, x)
    else:
        var t = fma(x, _D_L2E, _D_MAGIC)
        var i = _lo(t) - Int32(1)
        var k = _add_rn_d(t, -_D_MAGIC)
        var r = fma(k, -_D_LN2_HI, x)
        r = fma(k, -_D_LN2_LO, r)
        var hi2 = _u32(ahi) << UInt32(1)
        var small = hi2 < UInt32(2142496327)
        var zz = x if small else r
        if small:
            i = Int32(0)
        var q = fma(_dc(0x3E21F4076ACD15B6), zz, _dc(0x3E5AF86D8EBD13CD))
        q = fma(q, zz, _dc(0x3E927E5092BA033D))
        q = fma(q, zz, _dc(0x3EC71DDE6C5F9DA1))
        q = fma(q, zz, _dc(0x3EFA01A018D034E6))
        q = fma(q, zz, _dc(0x3F2A01A01B3B6940))
        q = fma(q, zz, _dc(0x3F56C16C16C1B5DD))
        q = fma(q, zz, _dc(0x3F8111111110F74D))
        q = fma(q, zz, _dc(0x3FA555555555554D))
        q = fma(q, zz, _dc(0x3FC5555555555557))
        q = fma(q, zz, Float64(0.5))
        q = q * zz
        var e = fma(q, zz, zz)
        var shi = (i << Int32(20)) + Int32(1072693248)
        if i == Int32(1024):
            shi = Int32(2145386496)
        var s = _i2d(Int32(0), shi)
        var u = fma(e, s, _add_rn_d(s, Float64(-0.5)))
        if i == Int32(1024):
            u = u + u
        if hi2 == UInt32(0):
            u = x
        z = _add_rn_d(u, u / fma(Float64(2.0), u, Float64(1.0)))
        var xb = _ub(x)
        if xb >= UInt64(0x408633CE8FB9F87E) and xb <= UInt64(
            0x7FF0000000000000
        ):
            z = _F64_INF
    return _copysign_d(z, a)


@always_inline
def nv_cosh(a: Float64) -> Float64:
    """`__nv_cosh`: 2 (E + 1 / (16 E)), E = e^|a| / 4, below |a| =
    0x40862E4F..., else inf (NaN stays)."""
    var ahi = _hi(a) & Int32(0x7FFFFFFF)
    var x = _i2d(_lo(a), ahi)
    var res: Float64
    if _u32(ahi) < UInt32(1082536911):
        var t = fma(x, _D_L2E, _D_MAGIC)
        var j = _lo(t)
        var k = _add_rn_d(t, -_D_MAGIC)
        var r = fma(k, -_D_LN2_HI, x)
        r = fma(k, -_D_LN2_LO, r)
        var p = _exp_poly_d(r)
        var e = _i2d(_lo(p), (j << Int32(20)) - Int32(2097152) + _hi(p))
        var r0 = _rcp_approx_ftz_d(e)
        var n = fma(-e, r0, Float64(1.0))
        n = fma(n, n, n)
        res = fma(fma(n, r0, r0), Float64(0.0625), e)
    else:
        res = a if _is_nan_d(a) else _F64_INF
    return res + res
