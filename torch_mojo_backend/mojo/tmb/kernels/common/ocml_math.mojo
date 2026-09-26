# ===----------------------------------------------------------------------=== #
# Scalar ports of ROCm's device libm (ocml) for the unary math ops.
#
# Stock torch on ROCm compiles the same .cu sources and jiterator strings as
# on CUDA (none of UnaryOpsKernel.cu, UnarySpecialOpsKernel.cu,
# UnaryGammaKernels.cu or cuda/Math.cuh branches on USE_ROCM), and hipcc /
# hiprtc resolve their float libm calls (`::asinf`, `::expm1f`, `log`,
# `sin`, ...) to ocml's `__ocml_*_f32`. Each function below is a literal
# translation of the matching file of ROCm-Device-Libs `ocml/src/*F.cl`
# (llvm-project amd-staging, amd/device-libs), with:
#
# * MATH_MAD (`__ocml_fmuladd_f32`) as an fma: gfx9 has fast fma32
#   (HAVE_FAST_FMA32), so the fma branches of ocml are the ones taken;
# * MATH_FAST_RCP / MATH_FAST_SQRT as `llvm.amdgcn.rcp` / `llvm.amdgcn.sqrt`
#   (v_rcp_f32 / v_sqrt_f32, ~1 ulp), and the IEEE operation elsewhere, so
#   the host (a unit test of the port) runs the same algorithm;
# * BUILTIN_LOG_F32 / BUILTIN_EXP2_F32 / BUILTIN_LOG10_F32 (what ocml's
#   log / exp2 / log10 are) as the `llvm.log` / `llvm.exp2` / `llvm.log10`
#   intrinsics, which the AMDGPU backend expands exactly as for ocml;
# * torch's ROCm build not finite-only, so every `!FINITE_ONLY_OPT()` branch
#   is kept.
# ===----------------------------------------------------------------------=== #

from std.math import fma
from std.sys import llvm_intrinsic
from std.sys.info import is_amd_gpu

from tmb.kernels.common.libdevice_port import _trig_reduction_slowpath_f

comptime _INF = Float32(from_bits=UInt32(0x7F800000))
comptime _NAN = Float32(from_bits=UInt32(0x7FC00000))


@always_inline
def _hex(bits: UInt32) -> Float32:
    return Float32(from_bits=bits)


@always_inline
def _abs(x: Float32) -> Float32:
    return llvm_intrinsic["llvm.fabs", Float32, has_side_effect=False](x)


@always_inline
def _copysign(mag: Float32, sgn: Float32) -> Float32:
    return Float32(
        from_bits=(mag.to_bits[DType.uint32]() & UInt32(0x7FFFFFFF))
        | (sgn.to_bits[DType.uint32]() & UInt32(0x80000000))
    )


@always_inline
def _rint(x: Float32) -> Float32:
    return llvm_intrinsic["llvm.roundeven", Float32, has_side_effect=False](x)


@always_inline
def _isnan(x: Float32) -> Bool:
    return (x.to_bits[DType.uint32]() & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000)


@always_inline
def _isinf(x: Float32) -> Bool:
    return (x.to_bits[DType.uint32]() & UInt32(0x7FFFFFFF)) == UInt32(
        0x7F800000
    )


@always_inline
def _fast_rcp(x: Float32) -> Float32:
    """MATH_FAST_RCP: v_rcp_f32."""
    comptime if is_amd_gpu():
        return llvm_intrinsic[
            "llvm.amdgcn.rcp", Float32, has_side_effect=False
        ](x)
    else:
        return Float32(1.0) / x


@always_inline
def _fast_sqrt(x: Float32) -> Float32:
    """MATH_FAST_SQRT: v_sqrt_f32."""
    comptime if is_amd_gpu():
        return llvm_intrinsic[
            "llvm.amdgcn.sqrt", Float32, has_side_effect=False
        ](x)
    else:
        return llvm_intrinsic["llvm.sqrt", Float32, has_side_effect=False](x)


@always_inline
def oc_sqrtf(x: Float32) -> Float32:
    """`__ocml_sqrt_f32` (MATH_SQRT): correctly rounded."""
    return llvm_intrinsic["llvm.sqrt", Float32, has_side_effect=False](x)


@always_inline
def oc_logf(x: Float32) -> Float32:
    """`__ocml_log_f32`: BUILTIN_LOG_F32."""
    return llvm_intrinsic["llvm.log", Float32, has_side_effect=False](x)


@always_inline
def oc_expf(x: Float32) -> Float32:
    """`__ocml_exp_f32`: BUILTIN_EXP_F32."""
    return llvm_intrinsic["llvm.exp", Float32, has_side_effect=False](x)


@always_inline
def oc_exp2f(x: Float32) -> Float32:
    """`__ocml_exp2_f32`: BUILTIN_EXP2_F32."""
    return llvm_intrinsic["llvm.exp2", Float32, has_side_effect=False](x)


@always_inline
def oc_log10f(x: Float32) -> Float32:
    """`__ocml_log10_f32`: BUILTIN_LOG10_F32."""
    return llvm_intrinsic["llvm.log10", Float32, has_side_effect=False](x)


@always_inline
def oc_expm1f(x: Float32) -> Float32:
    """`ocml/src/expm1F.cl` (the default, not EXTRA_ACCURACY, body)."""
    var q = _rint(x * _hex(0x3FB8AA3B))  # 0x1.715476p+0
    var t = fma(-q, _hex(0xB102E308), fma(-q, _hex(0x3F317218), x))
    var p = fma(t, _hex(0x395133B1), _hex(0x3AB69700))
    p = fma(t, p, _hex(0x3C0887F9))
    p = fma(t, p, _hex(0x3D2AAA81))
    p = fma(t, p, _hex(0x3E2AAAAB))
    p = fma(t, p, _hex(0x3F000000))
    p = fma(t, t * p, t)
    # ocml's (int)fn (q here), kept in the exponent range: out-of-range
    # lanes (|x| > 88.7, NaN) take a select below.
    var fc = min(max(q, Float32(-126.0)), Float32(127.0))
    var e = 127 if q == Float32(128.0) else Int(fc)
    var s = Float32(from_bits=UInt32(e + 127) << UInt32(23))
    var z = fma(s, p, s - Float32(1.0))
    z = Float32(2.0) * z if q == Float32(128.0) else z
    z = _INF if x > _hex(0x42B17217) else z  # 0x1.62e42ep+6
    z = Float32(-1.0) if x < Float32(-17.0) else z
    return z


@always_inline
def oc_asinf(x: Float32) -> Float32:
    """`ocml/src/asinF.cl`."""
    var ax = _abs(x)
    var tx = fma(ax, Float32(-0.5), Float32(0.5))
    var x2 = x * x
    var r = tx if ax >= Float32(0.5) else x2
    var p = fma(r, _hex(0x3D1C21A7), _hex(0x3C5FC5DA))
    p = fma(r, p, _hex(0x3D034C3C))
    p = fma(r, p, _hex(0x3D3641B1))
    p = fma(r, p, _hex(0x3D999BC8))
    p = fma(r, p, _hex(0x3E2AAAAC))
    var u = r * p
    var s = _fast_sqrt(r)
    var ret = fma(
        _hex(0x3F6EE581), _hex(0x3FD774EB), Float32(-2.0) * fma(s, u, s)
    )
    var xux = fma(ax, u, ax)
    ret = xux if ax < Float32(0.5) else ret
    return _copysign(ret, x)


@always_inline
def _oc_atanred(v: Float32) -> Float32:
    """`ocml/src/atanredF.cl`."""
    var t = v * v
    var z = fma(t, _hex(0x3B2D2A58), _hex(0xBC7A590C))
    z = fma(t, z, _hex(0x3D29FB3F))
    z = fma(t, z, _hex(0xBD97D4D7))
    z = fma(t, z, _hex(0x3DD931B2))
    z = fma(t, z, _hex(0xBE1160E6))
    z = fma(t, z, _hex(0x3E4CB8BF))
    z = fma(t, z, _hex(0xBEAAAA62))
    return fma(v, t * z, v)


@always_inline
def oc_atanf(x: Float32) -> Float32:
    """`ocml/src/atanF.cl`."""
    var v = _abs(x)
    var g = v > Float32(1.0)
    var vi = _fast_rcp(v)
    v = vi if g else v
    var a = _oc_atanred(v)
    var y = fma(_hex(0x3F6EE581), _hex(0x3FD774EB), -a)
    a = y if g else a
    return _copysign(a, x)


@always_inline
def oc_erfinvf(x: Float32) -> Float32:
    """`ocml/src/erfinvF.cl` (the fast-fma branch)."""
    var ax = _abs(x)
    var p: Float32
    if ax < Float32(0.375):
        var t = ax * ax
        p = fma(t, _hex(0x3E245B65), _hex(0xBCD14985))
        p = fma(t, p, _hex(0x3DB2D85A))
        p = fma(t, p, _hex(0x3DAAC0D7))
        p = fma(t, p, _hex(0x3E02D52B))
        p = fma(t, p, _hex(0x3E6D93A4))
        p = fma(t, p, _hex(0x3F62DFC5))
    else:
        var w = -oc_logf(fma(-ax, ax, Float32(1.0)))
        if w < Float32(5.0):
            w = w - Float32(2.5)
            p = fma(w, _hex(0x32F16588), _hex(0x34B84B36))
            p = fma(w, p, _hex(0xB66C7357))
            p = fma(w, p, _hex(0xB6935AC1))
            p = fma(w, p, _hex(0x396532DB))
            p = fma(w, p, _hex(0xBAA45408))
            p = fma(w, p, _hex(0xBB88E4EF))
            p = fma(w, p, _hex(0x3E7C8F63))
            p = fma(w, p, _hex(0x3FC02E2F))
        else:
            w = oc_sqrtf(w) - Float32(3.0)
            p = fma(w, _hex(0xB951F09B), _hex(0x38D3B56B))
            p = fma(w, p, _hex(0x3AB0DC72))
            p = fma(w, p, _hex(0xBB70BDE7))
            p = fma(w, p, _hex(0x3BBC127B))
            p = fma(w, p, _hex(0xBBF9C5D7))
            p = fma(w, p, _hex(0x3C1AA57E))
            p = fma(w, p, _hex(0x3F8036DB))
            p = fma(w, p, _hex(0x40354F7E))
    var ret = p * ax
    ret = _NAN if ax > Float32(1.0) else ret
    ret = _INF if ax == Float32(1.0) else ret
    return _copysign(ret, x)


@always_inline
def _oc_trigred(ax: Float32) -> Tuple[Float32, Int32]:
    """`trigredF.cl`: the fma Cody-Waite reduction of `trigredsmallF.cl`
    below 2^17; above, a Payne-Hanek reduction (libdevice's, standing in for
    `trigredlargeF.cl`: both exact to the float result)."""
    if ax >= Float32(131072.0):
        var r = _trig_reduction_slowpath_f(ax)
        return (r[0], r[1] & Int32(3))
    var q = _rint(ax * _hex(0x3F22F983))
    var r = fma(
        q,
        -_hex(0x27C234C4),
        fma(q, -_hex(0x33A22168), fma(q, -_hex(0x3FC90FDA), ax)),
    )
    var i = Int32(0) if _isnan(q) else Int32(Int(q)) & Int32(3)
    return (r, i)


@always_inline
def oc_sinf(x_in: Float32) -> Float32:
    """`ocml/src/sinF.cl` with `sincosredF.cl` (not EXTRA_PRECISION)."""
    var x = _NAN if _isinf(x_in) else x_in
    var ax = _abs(x)
    var red = _oc_trigred(ax)
    var r = red[0]
    var t = r * r
    var s = fma(
        r,
        t
        * fma(t, fma(t, _hex(0xB94C1982), _hex(0x3C0881C4)), _hex(0xBE2AAA9D)),
        r,
    )
    var c = fma(t, _hex(0x37D75334), _hex(0xBAB64F3B))
    c = fma(t, c, _hex(0x3D2AABF7))
    c = fma(t, c, _hex(0xBF000004))
    c = fma(t, c, Float32(1.0))
    var v = c if (red[1] & Int32(1)) != Int32(0) else s
    var flip = UInt32(0x80000000) if red[1] > Int32(1) else UInt32(0)
    var bits = (
        v.to_bits[DType.uint32]()
        ^ flip
        ^ (x.to_bits[DType.uint32]() ^ ax.to_bits[DType.uint32]())
    )
    return Float32(from_bits=bits)
