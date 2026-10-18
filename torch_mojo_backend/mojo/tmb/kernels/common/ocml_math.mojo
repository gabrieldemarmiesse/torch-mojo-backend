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
# * BUILTIN_LOG_F32 / BUILTIN_LOG2_F32 / BUILTIN_EXP2_F32 / BUILTIN_LOG10_F32
#   (what ocml's log / log2 / exp2 / log10 are) as the `llvm.log` /
#   `llvm.log2` / `llvm.exp2` / `llvm.log10` intrinsics, which the AMDGPU
#   backend expands exactly as for ocml;
# * torch's ROCm build not finite-only, so every `!FINITE_ONLY_OPT()` branch
#   is kept.
# ===----------------------------------------------------------------------=== #

from std.math import fma
from std.sys import llvm_intrinsic
from std.sys._assembly import inlined_assembly
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
def oc_log2f(x: Float32) -> Float32:
    """`__ocml_log2_f32`: BUILTIN_LOG2_F32."""
    return llvm_intrinsic["llvm.log2", Float32, has_side_effect=False](x)


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


# --------------------------------------------------------------------------- #
# asinh: `asinhF.cl` / `asinhD.cl` over the double-word arithmetic of `ep.h`
# and `lnepF.cl` / `lnepD.cl`. A pair is (hi, lo). Every add, subtract and
# multiply of an error-free transform goes through `_o_add` / `_o_mul`, which
# the GPU build's fast-math flags cannot reassociate or contract.
# --------------------------------------------------------------------------- #


@always_inline
def _o_add[dt: DType](a: Scalar[dt], b: Scalar[dt]) -> Scalar[dt]:
    comptime if is_amd_gpu() and dt == DType.float32:
        return rebind[Scalar[dt]](
            inlined_assembly[
                "v_add_f32 $0, $1, $2",
                Float32,
                constraints="=v,v,v",
                has_side_effect=False,
            ](rebind[Float32](a), rebind[Float32](b))
        )
    elif is_amd_gpu():
        return rebind[Scalar[dt]](
            inlined_assembly[
                "v_add_f64 $0, $1, $2",
                Float64,
                constraints="=v,v,v",
                has_side_effect=False,
            ](rebind[Float64](a), rebind[Float64](b))
        )
    else:
        return a + b


@always_inline
def _o_sub[dt: DType](a: Scalar[dt], b: Scalar[dt]) -> Scalar[dt]:
    comptime if is_amd_gpu() and dt == DType.float32:
        return rebind[Scalar[dt]](
            inlined_assembly[
                "v_sub_f32 $0, $1, $2",
                Float32,
                constraints="=v,v,v",
                has_side_effect=False,
            ](rebind[Float32](a), rebind[Float32](b))
        )
    elif is_amd_gpu():
        return rebind[Scalar[dt]](
            inlined_assembly[
                "v_add_f64 $0, $1, -$2",
                Float64,
                constraints="=v,v,v",
                has_side_effect=False,
            ](rebind[Float64](a), rebind[Float64](b))
        )
    else:
        return a - b


@always_inline
def _o_mul[dt: DType](a: Scalar[dt], b: Scalar[dt]) -> Scalar[dt]:
    comptime if is_amd_gpu() and dt == DType.float32:
        return rebind[Scalar[dt]](
            inlined_assembly[
                "v_mul_f32 $0, $1, $2",
                Float32,
                constraints="=v,v,v",
                has_side_effect=False,
            ](rebind[Float32](a), rebind[Float32](b))
        )
    elif is_amd_gpu():
        return rebind[Scalar[dt]](
            inlined_assembly[
                "v_mul_f64 $0, $1, $2",
                Float64,
                constraints="=v,v,v",
                has_side_effect=False,
            ](rebind[Float64](a), rebind[Float64](b))
        )
    else:
        return a * b


@always_inline
def _ep_rcp[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    """MATH_FAST_RCP: v_rcp_f32; v_rcp_f64 and two Newton steps."""
    comptime if dt == DType.float32:
        return rebind[Scalar[dt]](_fast_rcp(rebind[Float32](x)))
    comptime if not is_amd_gpu():
        return Scalar[dt](1.0) / x
    var r = llvm_intrinsic[
        "llvm.amdgcn.rcp", Scalar[dt], has_side_effect=False
    ](x)
    r = fma(fma(-x, r, Scalar[dt](1.0)), r, r)
    return fma(fma(-x, r, Scalar[dt](1.0)), r, r)


@always_inline
def _ep_div[dt: DType](x: Scalar[dt], y: Scalar[dt]) -> Scalar[dt]:
    """MATH_FAST_DIV: x * rcp(y); in double, the refined quotient."""
    comptime if dt == DType.float32 or not is_amd_gpu():
        return _o_mul(x, _ep_rcp(y))
    var r = _ep_rcp(y)
    var q = _o_mul(x, r)
    return fma(fma(-y, q, x), r, q)


@always_inline
def _ep_sqrt[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    """MATH_FAST_SQRT: v_sqrt_f32; in double, v_rsq_f64 refined."""
    comptime if dt == DType.float32:
        return rebind[Scalar[dt]](_fast_sqrt(rebind[Float32](x)))
    comptime if not is_amd_gpu():
        return llvm_intrinsic["llvm.sqrt", Scalar[dt], has_side_effect=False](x)
    var y = llvm_intrinsic[
        "llvm.amdgcn.rsq", Scalar[dt], has_side_effect=False
    ](x)
    var s0 = _o_mul(x, y)
    var h0 = _o_mul(Scalar[dt](0.5), y)
    var r0 = fma(-h0, s0, Scalar[dt](0.5))
    var h1 = fma(h0, r0, h0)
    var s1 = fma(s0, r0, s0)
    var d0 = fma(-s1, s1, x)
    var ret = fma(d0, h1, s1)
    return x if x == Scalar[dt](0.0) else ret


@always_inline
def _ep_fadd[
    dt: DType
](a: Scalar[dt], b: Scalar[dt]) -> Tuple[Scalar[dt], Scalar[dt]]:
    """fadd(T, T): fast two-sum."""
    var s = _o_add(a, b)
    return (s, _o_sub(b, _o_sub(s, a)))


@always_inline
def _ep_add[
    dt: DType
](a: Scalar[dt], b: Scalar[dt]) -> Tuple[Scalar[dt], Scalar[dt]]:
    """add(T, T): two-sum."""
    var s = _o_add(a, b)
    var d = _o_sub(s, a)
    return (s, _o_add(_o_sub(a, _o_sub(s, d)), _o_sub(b, d)))


@always_inline
def _ep_fsub[
    dt: DType
](a: Scalar[dt], b: Scalar[dt]) -> Tuple[Scalar[dt], Scalar[dt]]:
    """fsub(T, T): fast two-difference."""
    var d = _o_sub(a, b)
    return (d, _o_sub(_o_sub(a, d), b))


@always_inline
def _ep_mul[
    dt: DType
](a: Scalar[dt], b: Scalar[dt]) -> Tuple[Scalar[dt], Scalar[dt]]:
    """mul(T, T) and sqr(T): fma two-product."""
    var p = _o_mul(a, b)
    return (p, fma(a, b, -p))


@always_inline
def _ep_nrm[
    dt: DType
](a: Tuple[Scalar[dt], Scalar[dt]]) -> Tuple[Scalar[dt], Scalar[dt]]:
    return _ep_fadd(a[0], a[1])


@always_inline
def _ep_ldx[
    dt: DType
](a: Tuple[Scalar[dt], Scalar[dt]], e: Int32) -> Tuple[Scalar[dt], Scalar[dt]]:
    """ldx: both words times 2^e, exact (2^e is a normal number here)."""
    var p: Scalar[dt]
    comptime if dt == DType.float32:
        p = rebind[Scalar[dt]](
            Float32(from_bits=UInt32(e + Int32(127)) << UInt32(23))
        )
    else:
        p = rebind[Scalar[dt]](
            Float64(from_bits=UInt64(Int64(e) + Int64(1023)) << UInt64(52))
        )
    return (_o_mul(a[0], p), _o_mul(a[1], p))


@always_inline
def _oc_lnep[
    dt: DType
](a: Tuple[Scalar[dt], Scalar[dt]], ea: Int32) -> Scalar[dt]:
    """`lnepF.cl` / `lnepD.cl`: log of the pair `a` (a.hi normal and
    positive, or not finite), plus ea * ln 2."""
    # frexp(a.hi): mantissa in [0.5, 1) and exponent, from the bits
    var mh: Scalar[dt]
    var ex: Int32
    comptime if dt == DType.float32:
        var b = rebind[Float32](a[0]).to_bits[DType.uint32]()
        ex = Int32((b >> UInt32(23)) & UInt32(0xFF)) - Int32(126)
        mh = rebind[Scalar[dt]](
            Float32(from_bits=(b & UInt32(0x807FFFFF)) | UInt32(0x3F000000))
        )
    else:
        var b = rebind[Float64](a[0]).to_bits[DType.uint64]()
        ex = Int32((b >> UInt64(52)) & UInt64(0x7FF)) - Int32(1022)
        mh = rebind[Scalar[dt]](
            Float64(
                from_bits=(b & UInt64(0x800FFFFFFFFFFFFF))
                | UInt64(0x3FE0000000000000)
            )
        )
    var two_thirds: Scalar[dt]
    comptime if dt == DType.float32:
        two_thirds = Scalar[dt](0.6666667)
    else:
        two_thirds = Scalar[dt](0.6666666666666666)
    var e = ex - (Int32(1) if mh < two_thirds else Int32(0))
    var m = _ep_ldx(a, -e)

    # x = (m - 1) / (m + 1): fadd(-1, m) / fadd(1, m), div(T2, T2)
    var n = _ep_fadd(Scalar[dt](-1.0), m[0])
    n = _ep_nrm((n[0], _o_add(n[1], m[1])))
    var d = _ep_fadd(Scalar[dt](1.0), m[0])
    d = _ep_nrm((d[0], _o_add(d[1], m[1])))
    var y = _ep_rcp(d[0])
    var qhi = _o_mul(n[0], y)
    var r = _ep_fsub2(n, _ep_mul2(qhi, d))
    var qmi = _o_mul(r[0], y)
    r = _ep_fsub2(r, _ep_mul2(qmi, d))
    var qlo = _o_mul(r[0], y)
    var x = _ep_fadd(qhi, qmi)
    x = _ep_nrm((x[0], _o_add(x[1], qlo)))

    var s = _o_mul(x[0], x[0])
    var p: Scalar[dt]
    var ln2: Tuple[Scalar[dt], Scalar[dt]]
    comptime if dt == DType.float32:
        p = fma(
            s,
            fma(s, Scalar[dt](0.3035711), Scalar[dt](0.39957914)),
            Scalar[dt](0.6666695),
        )
        ln2 = (Scalar[dt](0.6931472), Scalar[dt](-1.9046542e-09))
    else:
        p = fma(
            s, Scalar[dt](0.15367016165258582), Scalar[dt](0.15250306356700066)
        )
        p = fma(s, p, Scalar[dt](0.18186352772476638))
        p = fma(s, p, Scalar[dt](0.2222213848310113))
        p = fma(s, p, Scalar[dt](0.28571429397443554))
        p = fma(s, p, Scalar[dt](0.39999999996131963))
        p = fma(s, p, Scalar[dt](0.6666666666667282))
        ln2 = (
            Scalar[dt](0.6931471805599453),
            Scalar[dt](2.3190468138462996e-17),
        )

    # mul(con(ln2), e + ea): mul(T2, T)
    var k = (e + ea).cast[dt]()
    var lk = _ep_mul(ln2[0], k)
    lk = _ep_nrm((lk[0], fma(ln2[1], k, lk[1])))
    # fadd(ldx(x, 1), s * x.hi * p): fadd(T2, T)
    var x2 = _ep_ldx(x, Int32(1))
    var t = _ep_fadd(x2[0], _o_mul(_o_mul(s, x[0]), p))
    t = _ep_nrm((t[0], _o_add(t[1], x2[1])))
    # add(T2, T2)
    var hs = _ep_add(lk[0], t[0])
    var ls = _ep_add(lk[1], t[1])
    hs = _ep_nrm((hs[0], _o_add(hs[1], ls[0])))
    hs = _ep_nrm((hs[0], _o_add(hs[1], ls[1])))
    return hs[0]


@always_inline
def _ep_mul2[
    dt: DType
](a: Scalar[dt], b: Tuple[Scalar[dt], Scalar[dt]]) -> Tuple[
    Scalar[dt], Scalar[dt]
]:
    """mul(T, T2)."""
    var p = _ep_mul(a, b[0])
    return _ep_nrm((p[0], fma(a, b[1], p[1])))


@always_inline
def _ep_fsub2[
    dt: DType
](a: Tuple[Scalar[dt], Scalar[dt]], b: Tuple[Scalar[dt], Scalar[dt]]) -> Tuple[
    Scalar[dt], Scalar[dt]
]:
    """fsub(T2, T2)."""
    var d = _ep_fsub(a[0], b[0])
    return _ep_nrm((d[0], _o_sub(_o_add(d[1], a[1]), b[1])))


@always_inline
def _oc_asinh[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    """`asinhF.cl` / `asinhD.cl`."""
    var big: Scalar[dt]
    var small_scale: Scalar[dt]
    var tiny: Scalar[dt]
    var ebig: Int32
    comptime if dt == DType.float32:
        big = Scalar[dt](1.8446744e19)  # 2^64
        small_scale = Scalar[dt](5.421011e-20)  # 2^-64
        tiny = Scalar[dt](0.00024414062)  # 2^-12
        ebig = Int32(64)
    else:
        big = Scalar[dt](1.3407807929942597e154)  # 2^512
        small_scale = Scalar[dt](7.458340731200207e-155)  # 2^-512
        tiny = Scalar[dt](7.450580596923828e-09)  # 2^-27
        ebig = Int32(512)
    var y = llvm_intrinsic["llvm.fabs", Scalar[dt], has_side_effect=False](x)
    var b = y >= big
    var s = small_scale if b else Scalar[dt](1.0)
    var sy = _o_mul(y, s)
    # add(sqr(sy), s * s): add(T2, T)
    var q = _ep_mul(sy, sy)
    var q2 = _ep_add(q[0], _o_mul(s, s))
    q = _ep_nrm((q2[0], _o_add(q2[1], q[1])))
    # root2(T2)
    var shi = _ep_sqrt(q[0])
    var e = _ep_fsub2(q, _ep_mul(shi, shi))
    var slo = _ep_div(e[0], _o_mul(Scalar[dt](2.0), shi))
    var rt = _ep_fadd(shi, Scalar[dt](0.0) if q[0] == Scalar[dt](0.0) else slo)
    # add(sy, root): add(T, T2)
    var a = _ep_add(sy, rt[0])
    a = _ep_nrm((a[0], _o_add(a[1], rt[1])))
    var z = _oc_lnep(a, ebig if b else Int32(0))
    z = y if y < tiny else z
    comptime if dt == DType.float32:
        var yb = rebind[Float32](y).to_bits[DType.uint32]()
        z = y if yb == UInt32(0x7F800000) else z
        return rebind[Scalar[dt]](
            Float32(
                from_bits=(
                    rebind[Float32](z).to_bits[DType.uint32]()
                    & UInt32(0x7FFFFFFF)
                )
                | (
                    rebind[Float32](x).to_bits[DType.uint32]()
                    & UInt32(0x80000000)
                )
            )
        )
    else:
        var yb = rebind[Float64](y).to_bits[DType.uint64]()
        z = y if yb == UInt64(0x7FF0000000000000) else z
        return rebind[Scalar[dt]](
            Float64(
                from_bits=(
                    rebind[Float64](z).to_bits[DType.uint64]()
                    & UInt64(0x7FFFFFFFFFFFFFFF)
                )
                | (
                    rebind[Float64](x).to_bits[DType.uint64]()
                    & UInt64(0x8000000000000000)
                )
            )
        )


@always_inline
def oc_asinhf(x: Float32) -> Float32:
    """`__ocml_asinh_f32`."""
    return _oc_asinh[DType.float32](x)


@always_inline
def oc_asinh(x: Float64) -> Float64:
    """`__ocml_asinh_f64`."""
    return _oc_asinh[DType.float64](x)
