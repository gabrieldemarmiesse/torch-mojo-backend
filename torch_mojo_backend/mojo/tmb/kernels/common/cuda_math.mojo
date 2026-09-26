# ===----------------------------------------------------------------------=== #
# Scalar ports of the CUDA float math routines ATen's unary kernels call.
#
# Stock torch on CUDA runs `::asinf`, `::erfcf`, `::lgammaf`, `::sinf`, ...
# (UnaryOpsKernel.cu, UnarySpecialOpsKernel.cu, and the jiterator strings of
# cuda/Math.cuh). Each body below is a literal translation of the matching
# `__nv_*` function of libdevice.10.bc with every `__nvvm_reflect` branch taken
# at value 0 (PyTorch is built without -ftz / fast math), the same convention
# as `libdevice_port.mojo`, whose `nv_logf` / `nv_expf` / `nv_tanf` these
# reuse. `llvm.nvvm.fma.rn` is `fma`; `mul.rn` / `add.rn` stay unfused
# (`_mul_rn` / `_add_rn`) because Mojo would contract them into an fma.
#
# The PTX approximation instructions libdevice uses (rcp/lg2/rsqrt.approx.ftz,
# ex2.approx, sqrt.approx) are emitted verbatim on NVIDIA and replaced by the
# IEEE operation elsewhere, so every other target (and the CPU, which runs the
# torch.compile graph on CPU tensors) computes the same algorithm to within
# an ulp rather than bit for bit.
# ===----------------------------------------------------------------------=== #

from std.math import exp2, fma, log2, sqrt
from std.sys import llvm_intrinsic
from std.sys._assembly import inlined_assembly
from std.sys.info import is_apple_gpu, is_nvidia_gpu

from tmb.kernels.common.libdevice_port import (
    _f2i_rn,
    _trig_reduction_slowpath_f,
    nv_logf,
)


@always_inline
def _f(bits: UInt32) -> Float32:
    return Float32(from_bits=bits)


comptime _INF = Float32(from_bits=UInt32(0x7F800000))


@always_inline
def _abs(x: Float32) -> Float32:
    return llvm_intrinsic["llvm.fabs", Float32, has_side_effect=False](x)


@always_inline
def _trunc(x: Float32) -> Float32:
    return llvm_intrinsic["llvm.trunc", Float32, has_side_effect=False](x)


@always_inline
def _round_away(x: Float32) -> Float32:
    """llvm.nvvm.round.f: nearest integer, halfway cases away from zero."""
    return llvm_intrinsic["llvm.round", Float32, has_side_effect=False](x)


@always_inline
def _floor(x: Float32) -> Float32:
    return llvm_intrinsic["llvm.floor", Float32, has_side_effect=False](x)


@always_inline
def _copysign(mag: Float32, sgn: Float32) -> Float32:
    return Float32(
        from_bits=(mag.to_bits[DType.uint32]() & UInt32(0x7FFFFFFF))
        | (sgn.to_bits[DType.uint32]() & UInt32(0x80000000))
    )


@always_inline
def _mul_rn(a: Float32, b: Float32) -> Float32:
    """mul.rn.f32: a rounded product Mojo may not fuse into a later add."""
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "mul.rn.f32 $0, $1, $2;",
            Float32,
            constraints="=f,f,f",
            has_side_effect=False,
        ](a, b)
    else:
        return a * b


@always_inline
def _add_rn(a: Float32, b: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "add.rn.f32 $0, $1, $2;",
            Float32,
            constraints="=f,f,f",
            has_side_effect=False,
        ](a, b)
    else:
        return a + b


@always_inline
def _rcp_approx_ftz(x: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "rcp.approx.ftz.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x)
    else:
        return 1 / x


@always_inline
def _sqrt_approx(x: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "sqrt.approx.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x)
    else:
        return sqrt(x)


@always_inline
def _rsqrt_approx_ftz(x: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "rsqrt.approx.ftz.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x)
    else:
        return 1 / sqrt(x)


@always_inline
def _lg2_approx_ftz(x: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "lg2.approx.ftz.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x)
    else:
        return log2(x)


@always_inline
def _ex2_approx(x: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "ex2.approx.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x)
    else:
        return exp2(x)


@always_inline
def _ex2_approx_ftz(x: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "ex2.approx.ftz.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x)
    else:
        return exp2(x)


@always_inline
def ieee_sqrtf(x: Float32) -> Float32:
    """`::sqrtf` under nvcc's default -prec-sqrt=true: sqrt.rn.f32."""
    return llvm_intrinsic["llvm.sqrt", Float32, has_side_effect=False](x)


# --------------------------------------------------------------------------- #
# exp2f / log10f / expm1f
# --------------------------------------------------------------------------- #


@always_inline
def nv_exp2f(a: Float32) -> Float32:
    """`__nv_exp2f`: one ex2.approx.f32 (full range, non-ftz)."""
    return _ex2_approx(a)


@always_inline
def nv_log10f(a: Float32) -> Float32:
    """`__nv_log10f`: the `__nv_logf` body scaled by log10(e) (its zero case
    already reads -inf through the product)."""
    return nv_logf(a) * _f(0x3EDE5BD9)


@always_inline
def nv_expm1f(a: Float32) -> Float32:
    """`__nv_expm1f`. On NVIDIA the reduction rounds to nearest even and the
    ex2 flushes subnormals, as the CUDA 12 expm1f in stock torch's SASS
    does (one FRND and one MUFU.EX2 where libdevice.10.bc's round-away and
    non-ftz ex2 cost ~9 more instructions); they differ from libdevice
    only on exact ties of x log2(e) and where the result is -1 anyway."""
    var t = Float32(0.0)
    var j: Float32
    var e: Float32
    comptime if is_nvidia_gpu():
        if not (_abs(a) < _f(0x3ED1EB85)):  # 0.41
            t = llvm_intrinsic[
                "llvm.roundeven", Float32, has_side_effect=False
            ](a * _f(0x3FB8AA3B))
        j = Float32(127.0) if t == Float32(128.0) else t
        e = _ex2_approx_ftz(j)
    else:
        if not (_abs(a) < _f(0x3ED1EB85)):  # 0.41
            t = _round_away(a * _f(0x3FB8AA3B))
        j = Float32(127.0) if t == Float32(128.0) else t
        e = _ex2_approx(j)
    var r = fma(-t, _f(0x3F317200), a)
    r = fma(-t, _f(0x35BFBE8E), r)
    var p = fma(_f(0x3AB5EBE6), r, _f(0x3C095663))
    p = fma(p, r, _f(0x3D2AABE3))
    p = fma(p, r, _f(0x3E2AA9F6))
    p = fma(p, r, _f(0x3EFFFFFE))
    p = r * p
    p = fma(p, r, r)
    var u = fma(p, e, e + Float32(-1.0))
    if t == Float32(128.0):
        u = u + u
    if j > Float32(128.0):
        u = _INF
    if j < Float32(-25.0):
        u = Float32(-1.0)
    if a == Float32(0.0):
        u = a + a
    return u


@always_inline
def nv_tanhf(a: Float32) -> Float32:
    """`__nv_tanhf` (what `c10::cuda::compat::tanh` runs for float): an odd
    polynomial below |a| = 0.6, else 1 - 2 / (e^(2|a|) + 1) through
    ex2/rcp.approx, saturated to 1 from |a| = 9.0109 and signed by `a`. NaN
    takes the polynomial branch (the comparison is unordered)."""
    var s = _abs(a)
    comptime if is_nvidia_gpu():
        # Both branches, then a select (what nvcc emits for torch's tanhf):
        # a per-element branch would serialize the SIMD lanes of a thread
        # into separate divergence regions (mish f32 16M: 91 -> 75 us).
        var a2 = a * a
        var p = fma(_f(0x3C80F082), a2, _f(0xBD563CAE))
        p = fma(p, a2, _f(0x3E085941))
        p = fma(p, a2, _f(0xBEAAA9ED))
        p = fma(p, a2, Float32(0.0))
        var small = fma(p, a, a)
        var e = _ex2_approx_ftz(_mul_rn(s, _f(0x4038AA3B)))  # 2 * log2(e)
        var r = fma(
            _rcp_approx_ftz(e + Float32(1.0)), Float32(-2.0), Float32(1.0)
        )
        r = Float32(1.0) if s >= _f(0x41102CB4) else r  # 9.0109
        return _copysign(r, a) if s >= _f(0x3F19999A) else small
    if not (s >= _f(0x3F19999A)):  # 0.6, `fcmp ult`
        var a2 = a * a
        var p = fma(_f(0x3C80F082), a2, _f(0xBD563CAE))
        p = fma(p, a2, _f(0x3E085941))
        p = fma(p, a2, _f(0xBEAAA9ED))
        p = fma(p, a2, Float32(0.0))
        return fma(p, a, a)
    var e = _ex2_approx_ftz(_mul_rn(s, _f(0x4038AA3B)))  # 2 * log2(e)
    var r = fma(_rcp_approx_ftz(e + Float32(1.0)), Float32(-2.0), Float32(1.0))
    if s >= _f(0x41102CB4):  # 9.0109
        r = Float32(1.0)
    return _copysign(r, a)


@always_inline
def _div_approx(a: Float32, b: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "div.approx.f32 $0, $1, $2;",
            Float32,
            constraints="=f,f,f",
            has_side_effect=False,
        ](a, b)
    else:
        return a / b


@always_inline
def _fma_rz(a: Float32, b: Float32, c: Float32) -> Float32:
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "fma.rz.f32 $0, $1, $2, $3;",
            Float32,
            constraints="=f,f,f,f",
            has_side_effect=False,
        ](a, b, c)
    else:
        return fma(a, b, c)


@always_inline
def nv_fmodf(x: Float32, y: Float32) -> Float32:
    """`__nv_fmodf`: an exact remainder, so every correct algorithm agrees
    bit for bit; this one takes a truncated approximate quotient below
    |x| / |y| = 2**23 and reduces 23 exponent bits per step above it,
    where a bit-at-a-time long division loops up to 254 times."""
    # The special cases on the bits first (NaN operand, infinite x or zero
    # y give NaN; |x| < |y|, infinite y included, gives x), as libdevice's
    # final selects would: the GPU build's fast-math flags may fold float
    # NaN tests.
    var ixb = x.to_bits[DType.uint32]() & UInt32(0x7FFFFFFF)
    var iyb = y.to_bits[DType.uint32]() & UInt32(0x7FFFFFFF)
    if iyb == 0 or ixb >= UInt32(0x7F800000) or iyb > UInt32(0x7F800000):
        return Float32(from_bits=UInt32(0x7FFFFFFF))
    if ixb < iyb:
        return x
    var ay = _abs(y)
    var ax = _abs(x)
    var ans: Float32
    var y23 = ay * _f(0x4B000000)  # 2**23
    if not (ax > y23):
        var q = _trunc(_div_approx(ax, ay))
        var r = fma(-ay, q, ax)
        var rb = r.to_bits[DType.uint32]()
        if not (rb < ay.to_bits[DType.uint32]()):
            if rb > UInt32(0x80000000):
                var q1 = q + Float32(-1.0)
                q = q1 + Float32(-1.0) if r < -ay else q1
            else:
                q = q + Float32(1.0)
                if not (r < ay * Float32(2.0)):
                    q = q + Float32(1.0)
                    var t = fma(Float32(-3.0), ay, r)
                    q = q + Float32(1.0) if t >= Float32(0.0) else q
        ans = fma(-ay, q, ax)
    else:
        var xb = ax.to_bits[DType.uint32]()
        var yb = y23.to_bits[DType.uint32]()
        var ey = yb & UInt32(0xFF800000)
        var cur = Float32(
            from_bits=(xb & UInt32(0x7FFFFF)) | UInt32(0x3F800000)
        )
        var fy = Float32(from_bits=(yb & UInt32(0x7FFFFF)) | UInt32(0x3F800000))
        var rcp = _rcp_approx_ftz(fy)
        var i = (xb + UInt32(0x0B800000) - ey) & UInt32(0xFF800000)
        while i != 0:
            var step = min(i, UInt32(0x0B800000))
            var t = Float32(from_bits=cur.to_bits[DType.uint32]() + step)
            var q = fma(t, rcp, Float32(-0.0))
            var e1 = fma(-fy, q, t)
            q = fma(e1, rcp, q)
            var e2 = fma(-fy, q, t)
            q = _trunc(_fma_rz(e2, rcp, q))
            cur = fma(-fy, q, t)
            i -= step
            if cur.to_bits[DType.uint32]() == 0:
                break
        ans = Float32(from_bits=ey) * (cur * _f(0x34000000))  # 2**-23
    return Float32(
        from_bits=(x.to_bits[DType.uint32]() & UInt32(0x80000000))
        | ans.to_bits[DType.uint32]()
    )


@always_inline
def cuda_powf_core(ax: Float32, y: Float32) -> Float32:
    """|x| ** y for a finite positive `ax` and a finite nonzero `y` (the
    caller selects every special case: zeros, infinities, NaN, x = 1, a
    negative base's sign and domain), as the CUDA 12 math library's `powf`
    computes it -- read off the SASS of stock torch's
    `pow_tensor_scalar_kernel_impl<float>` (libtorch_cuda.so, sm_90), which
    is not libdevice.10.bc's `__nv_powf`: log2|x| in double-float from
    u = 2(m - 1) / (m + 1), times y with the product's error kept, then a
    degree-6 exp2 polynomial scaled by two exact powers of two. Products that
    feed an add are `_mul_rn` so LLVM cannot contract them into an fma the
    SASS does not have."""
    comptime log2e = _f(0x3FB8AA3B)
    var tiny = ax < _f(0x00800000)  # 2**-126
    var a = ax * _f(0x4B800000) if tiny else ax  # 2**24
    var ab = a.to_bits[DType.uint32]().cast[DType.int32]()
    var e = (ab - Int32(0x3F3504F3)) & Int32(-8388608)  # 0xFF800000
    var m = Float32(from_bits=(ab - e).cast[DType.uint32]())
    var rcp = _rcp_approx_ftz(m + Float32(1.0))
    var mm1 = m + Float32(-1.0)
    var u = _mul_rn(rcp, mm1 + mm1)
    var ex = fma(
        e.cast[DType.float32](),
        _f(0x34000000),  # 2**-23
        Float32(-24.0) if tiny else Float32(0.0),
    )
    var d = mm1 - u
    var ulo = _mul_rn(rcp, fma(mm1, -u, d + d))
    var u2 = _mul_rn(u, u)
    var hi = fma(u, log2e, ex)
    var p = fma(u2, _f(0x3A2C32E4), _f(0x3B52E7DB))
    p = fma(u2, p, _f(0x3C93BB73))
    p = fma(u2, p, _f(0x3DF6384F))
    var r = fma(ulo, log2e, fma(u, log2e, ex - hi))
    p = _mul_rn(u2, p)
    r = fma(u, _f(0x32A55E34), r)  # log2(e) low part
    r = fma(ulo, _mul_rn(p, Float32(3.0)), r)
    var lo = fma(u, p, r)
    var l = hi + lo
    var t = _mul_rn(l, y)
    var n: Float32
    comptime if is_apple_gpu():
        # Round to nearest even without `llvm.roundeven`, which the Metal
        # backend's code generator rejects (an internal compiler error at
        # pipeline creation).
        n = t.__round__()
    else:
        n = llvm_intrinsic["llvm.roundeven", Float32, has_side_effect=False](t)
    var tl = fma(lo - (l - hi), y, fma(l, y, -t))
    var f = tl + (t - n)
    var q = fma(f, _f(0x391FCB8E), _f(0x3AAF85ED))
    q = fma(f, q, _f(0x3C1D9856))
    q = fma(f, q, _f(0x3D6357BB))
    q = fma(f, q, _f(0x3E75FDEC))
    q = fma(f, q, _f(0x3F317218))
    q = fma(f, q, Float32(1.0))
    if _abs(t) > Float32(152.0):
        return _INF if t >= Float32(0.0) else Float32(0.0)
    var bias = UInt32(0) if n > Float32(0.0) else UInt32(0x83000000)
    var scale = Float32(
        from_bits=(Int32(n).cast[DType.uint32]() << UInt32(23)) - bias
    )
    return _mul_rn(q, Float32(from_bits=bias + UInt32(0x7F000000))) * scale


# --------------------------------------------------------------------------- #
# asinf / atanf
# --------------------------------------------------------------------------- #


@always_inline
def nv_asinf(a: Float32) -> Float32:
    """`asinf` as the CUDA 12 math library stock torch links computes it
    (read from the SASS of torch's float `asin_kernel_cuda`; libdevice.10.bc's
    `__nv_asinf` splits at 0.57 with a different polynomial and differs by up
    to 3 ulp on ~40% of inputs): split at 0.56, sqrt((1 - |x|) / 2) as
    rsqrt.approx plus one Newton step (0 at |x| = 1), an odd degree-9
    polynomial, pi/2 - 2 r above the split, the sign of x unless NaN."""
    var x = _abs(a)
    var big = x > _f(0x3F0F5C29)  # 0.56
    var t = fma(-x, Float32(0.5), Float32(0.5))
    var rs = _rsqrt_approx_ftz(t)
    var h = t * rs
    var e = fma(-h, rs * Float32(0.5), Float32(0.5))
    var s = fma(h, e, h)
    s = s if x != Float32(1.0) else Float32(0.0)
    var d = s if big else x
    var d2 = d * d
    var p = fma(d2, _f(0x3D4DD2F7), _f(0x3C99CA97))
    p = fma(d2, p, _f(0x3D3F90E8))
    p = fma(d2, p, _f(0x3D993CCF))
    p = fma(d2, p, _f(0x3E2AAC04))
    p = d2 * p
    var r = fma(d, p, d)
    if big:
        r = fma(_f(0x3FD774EB), _f(0x3F6EE581), r * Float32(-2.0))
    if r == r:  # ordered: copy the sign of the input
        return _copysign(r, a)
    return r


@always_inline
def cuda_acosf(a: Float32) -> Float32:
    """acosf as the CUDA math library computes it (stock torch's
    `acos_kernel_cuda<float>` SASS, libtorch_cuda.so sm_90): for |a| > 0.56
    the argument is sqrt((1 - |a|) / 2) from a refined rsqrt (0 at |a| = 1),
    else a itself; one odd polynomial gives its asin, then pi/2 - asin(a),
    2 asin(s) or pi - 2 asin(s)."""
    var ax = _abs(a)
    var t = fma(-ax, Float32(0.5), Float32(0.5))
    var y = _rsqrt_approx_ftz(t)
    var s0 = t * y
    var e = fma(-s0, y * Float32(0.5), Float32(0.5))
    var s = fma(s0, e, s0)
    if not (ax != Float32(1.0)):
        s = Float32(0.0)
    var big = ax > _f(0x3F0F5C29)  # 0.56
    var d = _copysign(s if big else ax, a)
    var d2 = d * d
    var p = fma(d2, _f(0x3D10ECEF), _f(0x3C8B1ABB))
    p = fma(d2, p, _f(0x3CFC028C))
    p = fma(d2, p, _f(0x3D372139))
    p = fma(d2, p, _f(0x3D9993DB))
    p = fma(d2, p, _f(0x3E2AAAC6))
    p = d2 * p
    var r = fma(d, p, d)
    var r2 = r if big else -r
    if not (a > _f(0x3F0F5C29)):
        r = fma(_f(0x3FD774EB), _f(0x3F6EE581), r2)  # pi/2 + r2
    if big:
        r = r + r
    return r


@always_inline
def cuda_hypotf(x: Float32, y: Float32) -> Float32:
    """hypotf as the CUDA math library computes it (stock torch's
    `hypot_kernel_cuda` SASS, libtorch_cuda.so sm_90): both magnitudes
    scaled by a power of two taken from the larger one's exponent, one
    fma-summed square, an IEEE sqrt, the scale restored; a zero smaller
    magnitude returns the larger, two infinities infinity (C99's
    hypot(inf, NaN) = inf included)."""
    var a = (_abs(x) + Float32(-0.0)).to_bits[DType.uint32]()
    var b = (_abs(y) + Float32(-0.0)).to_bits[DType.uint32]()
    var mx = max(a, b)
    var mn = min(a, b)
    var e = mx & UInt32(0xFE000000)
    var scale = Float32(from_bits=UInt32(0x7E800000) - e)
    var t = Float32(from_bits=mn) * scale
    var u = Float32(from_bits=mx) * scale
    var r = ieee_sqrtf(fma(u, u, t * t))
    var fmn = Float32(from_bits=mn)
    var res = Float32(from_bits=mx)
    if fmn != Float32(0.0):
        res = Float32(from_bits=e | UInt32(0x800000)) * r
    if fmn.to_bits[DType.uint32]() == UInt32(0x7F800000):
        res = _INF
    return res


@always_inline
def cuda_atan2f(y: Float32, x: Float32) -> Float32:
    """atan2f as the CUDA math library computes it (stock torch's
    `atan2_kernel_cuda` SASS, libtorch_cuda.so sm_90): t = min / max of the
    magnitudes by IEEE division, one odd polynomial, the octant fixed up
    from the signs, then C99's |y| == |x| and zero cases and NaN."""
    var ay = _abs(y)
    var ax = _abs(x)
    var swap = ay > ax
    var mx = ay if swap else ax
    var mn = ax if swap else ay
    var t = mn / mx
    var t2 = t * t
    var p = fma(t2, _f(0x3B33710B), _f(0xBC807748))
    p = fma(t2, p, _f(0x3D2CDAB2))
    p = fma(t2, p, _f(0xBD992D10))
    p = fma(t2, p, _f(0x3DD9EA6C))
    p = fma(t2, p, _f(0xBE117CB1))
    p = fma(t2, p, _f(0x3E4CBCE0))
    p = fma(t2, p, _f(0xBEAAAA7D))
    var r = fma(t2 * p, t, t)
    var x_pos = x.to_bits[DType.uint32]() < UInt32(0x80000000)  # +0 too
    if swap:
        # pi/2 -+ r, pi/2 split as a product for the fma.
        r = fma(_f(0x3F6EE581), _f(0x3FD774EB), -r if x_pos else r)
    elif not x_pos:
        r = fma(_f(0x3FEEE581), _f(0x3FD774EB), -r)  # pi - r
    if ax == ay:
        r = _f(0x3F490FDB) if x_pos else _f(0x4016CBE4)  # pi/4, 3pi/4
    if not (mx != Float32(0.0)):
        r = Float32(0.0) if x_pos else _f(0x40490FDB)  # 0, pi
    r = _copysign(r, y)
    var s = ax + ay
    if s.to_bits[DType.uint32]() > UInt32(0x7F800000):
        return s  # NaN
    return r


@always_inline
def nv_atanf(a: Float32) -> Float32:
    """`__nv_atanf`; on NVIDIA the CUDA 12 atanf of stock torch's SASS
    (`atan_kernel_cuda`, libtorch_cuda.so sm_90) instead: rcp.approx of |a|
    above 1 and one odd polynomial, where libdevice.10.bc divides twice with
    full IEEE division (atan bf16 16M: 56 us against torch's 39)."""
    comptime if is_nvidia_gpu():
        var ax = _abs(a)
        var big = ax > Float32(1.0)
        var t = _rcp_approx_ftz(ax) if big else ax
        var t2 = t * t
        var p = fma(t2, _f(0x3B2090AA), _f(0xBC6BE14F))
        p = fma(t2, p, _f(0x3D23397E))
        p = fma(t2, p, _f(0xBD948A7A))
        p = fma(t2, p, _f(0x3DD76B21))
        p = fma(t2, p, _f(0xBE111E88))
        p = fma(t2, p, _f(0x3E4CAF60))
        p = fma(t2, p, _f(0xBEAAAA27))
        var r = fma(t, t2 * p, t)
        if big:
            # pi/2 - r, with pi/2 split as a product for the fma.
            r = fma(_f(0x3F6EE581), _f(0x3FD774EB), -r)
        if (ax.to_bits[DType.uint32]()) > UInt32(0x7F800000):
            return r  # NaN
        return _copysign(r, a)
    var x = _abs(a)
    var big = x > Float32(1.0)
    var t = Float32(1.0) / x if big else x
    var t2 = _mul_rn(t, t)
    var p = fma(t2, _f(0xBF52C7EA), _f(0xC0B59883))
    p = fma(p, t2, _f(0xC0D21907))
    p = t2 * p
    p = t * p
    var q = t2 + _f(0x41355DC0)
    q = fma(q, t2, _f(0x41E6BD60))
    q = fma(q, t2, _f(0x419D92C8))
    var r = fma(p, Float32(1.0) / q, t)
    if big:
        r = _f(0x3FC90FDB) - r
    if x == x:
        return _copysign(r, a)
    return r


# --------------------------------------------------------------------------- #
# erfcf / erfinvf
# --------------------------------------------------------------------------- #


@always_inline
def nv_erfcf(a: Float32) -> Float32:
    """`__nv_erfcf`."""
    var x = _abs(a)
    var num = x + Float32(-4.0)
    var rden = _rcp_approx_ftz(x + Float32(4.0))
    var q = _mul_rn(num, rden)
    var e = fma(Float32(-4.0), q + Float32(1.0), x)
    e = fma(-q, x, e)
    q = fma(rden, e, q)
    var p = fma(_f(0x3A69A091), q, _f(0x3BE6E05B))
    p = fma(p, q, _f(0xBC81FB4B))
    p = fma(p, q, _f(0x3D15373B))
    p = fma(p, q, _f(0xBD887C5A))
    p = fma(p, q, _f(0x3DC021D5))
    p = fma(p, q, _f(0xBDCED424))
    p = fma(p, q, _f(0x3D8B74DE))
    p = fma(p, q, _f(0x3C7BF170))
    p = fma(p, q, _f(0xBE0EF8D4))
    p = fma(p, q, _f(0x3F9DD2C9))
    var rd2 = _rcp_approx_ftz(fma(Float32(2.0), x, Float32(1.0)))
    var y = _mul_rn(p, rd2)
    var w = fma(x, y * Float32(-2.0), p)
    var f = fma(w - y, rd2, y)
    # exp(-x^2), with the rounding error of -x^2 folded back in.
    var nx2 = x * (-x)
    var jt = _trunc(_mul_rn(nx2, _f(0x3FB8AA3B)))
    var j = _copysign(Float32(126.0), jt) if _abs(jt) > Float32(126.0) else jt
    var r = fma(j, _f(0xBF317218), nx2)
    r = fma(j, _f(0x3102E308), r)
    var scale = Float32(
        from_bits=(j + _f(0x4B40007F)).to_bits[DType.uint32]() << UInt32(23)
    )
    var ex = _ex2_approx_ftz(r * _f(0x3FB8AA3B)) * scale
    var err = fma(-x, x, -nx2)
    var res = fma(ex, err, ex) * f
    if x > _f(0x4120E148):  # 10.055
        res = Float32(0.0)
    if a < Float32(0.0):
        res = Float32(2.0) - res
    return res


@always_inline
def nv_erfinvf(a: Float32) -> Float32:
    """`erfinvf` as the jiterator's NVRTC (CUDA 13) compiles it: libdevice's
    `__nv_erfinvf`, except that the tail takes sqrt(-w) as -w * rsqrt(-w)
    (read from the SASS of torch's cached `erfinv_kernel`), where
    libdevice.10.bc divides 1 / rsqrt(-w): 2 ulp apart near |a| = 1."""
    var w = _lg2_approx_ftz(fma(a, -a, Float32(1.0)))
    if w < _f(0xC1033333):  # -8.2
        var s = _rsqrt_approx_ftz(-w)
        var p = fma(_f(0xBF1704A1), s, _f(0xBF29BAA5))
        p = fma(p, s, _f(0x3FCC6ADC))
        p = fma(p, s, _f(0xBF2CDAED))
        p = fma(p, s, _f(0xBDC30537))
        p = fma(p, s, _f(0x3F55D9B9))
        var r = p * (w * -s)
        if w == -_INF:  # |a| = 1: -w * 0 would be NaN
            r = -w
        return _copysign(r, a)
    var t = -w
    var p = fma(_f(0xAF8A6370), t, _f(0x3221F645))
    p = fma(p, t, _f(0xB4016FDA))
    p = fma(p, t, _f(0x3468F846))
    p = fma(p, t, _f(0x370742AA))
    p = fma(p, t, _f(0xB804DB4D))
    p = fma(p, t, _f(0xBA4AFEA1))
    p = fma(p, t, _f(0x3BB5C027))
    p = fma(p, t, _f(0x3E24AE0F))
    p = fma(p, t, _f(0x3F62DFC4))
    return a * p


@always_inline
def nv_erff(a: Float32) -> Float32:
    """CUDA's `erff`: an odd polynomial in a below |a| = 1.00296, else
    1 - 2**poly(|a|) with the sign of a restored. The coefficients of the
    |a| >= 1.00296 branch are the CUDA 12 toolkit's, read from the SASS of
    torch's GeluBackwardCUDAKernelImpl in libtorch_cuda.so: they differ from
    libdevice.10.bc's `__nv_erff` (whose small-|a| branch is the same)."""
    var x = _abs(a)
    var big = x >= _f(0x3F8060FE)
    var t = x if big else a * a
    var p = fma(
        _f(0x38EB4C3A) if big else _f(0x38B1E96A),
        t,
        _f(0xBAAE005B) if big else _f(0xBA574D20),
    )
    p = fma(p, t, _f(0x3C09919F) if big else _f(0x3BAAD5EA))
    p = fma(p, t, _f(0xBD24D99A) if big else _f(0xBCDC1BE7))
    p = fma(p, t, _f(0x3E235519) if big else _f(0x3DE718AF))
    p = fma(p, t, _f(0x3F69B4F9) if big else _f(0xBEC093AC))
    p = fma(p, t, _f(0x3F210A14) if big else _f(0x3E0375D3))
    var u = -t if big else a
    var r = fma(p, u, u)
    if big:
        r = _copysign(Float32(1.0) - _ex2_approx_ftz(r), a)
    return r


@always_inline
def nv_erf(a: Float64) -> Float64:
    """`__nv_erf`: 1 - exp(-q(|a|)), q a degree-23 polynomial, the
    exponential split into 2**k and a corrected expm1 of the residual; 1 from
    |a| >= 5.9215 on; the sign of a restored."""
    var x = llvm_intrinsic["llvm.fabs", Float64, has_side_effect=False](a)
    comptime C = [
        UInt64(0xBCF0679AFBA6F279),
        UInt64(0x3D47088FDB46FA5F),
        UInt64(0xBD8DF9F9B976A9B2),
        UInt64(0x3DC7F1F5590CC332),
        UInt64(0xBDFA28A3CD2D56C4),
        UInt64(0x3E2485EE67835925),
        UInt64(0xBE476DB45919F583),
        UInt64(0x3E62D698D98C8D71),
        UInt64(0xBE720A2C7155D5C6),
        UInt64(0xBE41D29B37CA1397),
        UInt64(0x3EA2EF6CC0F67A49),
        UInt64(0xBEC102B892333B6F),
        UInt64(0x3ECA30375BA9A84E),
        UInt64(0x3ECAAD18DEDEA43E),
        UInt64(0xBEFF05355BC5B225),
        UInt64(0x3F10E37A3108BC8B),
        UInt64(0x3EFB292D828E5CB2),
        UInt64(0xBF4356626EBF9BFA),
        UInt64(0x3F5BCA68F73D6AFC),
        UInt64(0xBF2B6B69EBBC280B),
        UInt64(0xBF9396685912A453),
        UInt64(0x3FBA4F4E2A1ABEF8),
        UInt64(0x3FE45F306DC9C8BB),
        UInt64(0x3FC06EBA8214DB69),
    ]
    comptime c0 = C[0]
    comptime c1 = C[1]
    var p = fma(Float64(from_bits=c0), x, Float64(from_bits=c1))
    comptime for i in range(2, 24):
        comptime ci = C[i]
        p = fma(p, x, Float64(from_bits=ci))
    var t = fma(p, x, x)
    var tail = fma(p, x, x - t)
    var hi = -t
    var lo = -tail
    var k = _round_away(hi.cast[DType.float32]() * _f(0x3FB8AA3B))
    var kd = k.cast[DType.float64]()
    var r = fma(-kd, Float64(from_bits=UInt64(0x3FE62E42FEFA39EF)), hi)
    comptime E = [
        UInt64(0x3E928A27F89B6999),
        UInt64(0x3EC71DE715FF7E07),
        UInt64(0x3EFA019A6B0AC45A),
        UInt64(0x3F2A01A017EED94F),
        UInt64(0x3F56C16C17F2A71B),
        UInt64(0x3F811111111173C4),
        UInt64(0x3FA555555555211A),
        UInt64(0x3FC5555555555540),
        UInt64(0x3FE0000000000005),
    ]
    comptime e0 = E[0]
    var q = fma(
        Float64(from_bits=UInt64(0x3E5AE904A4741B81)), r, Float64(from_bits=e0)
    )
    comptime for i in range(1, 9):
        comptime ei = E[i]
        q = fma(q, r, Float64(from_bits=ei))
    var em1 = fma(q * r, r, lo) + r
    var s = _ex2_approx_ftz(k).cast[DType.float64]()
    var res = fma(-em1, s, Float64(1.0) - s)
    if x >= Float64(from_bits=UInt64(0x4017AFB48DC96626)) and not _is_nan_d(a):
        res = Float64(1.0)
    return _copysign_d(res, a)


# --------------------------------------------------------------------------- #
# double routines of the activation kernels (erf above, expm1, tanh)
# --------------------------------------------------------------------------- #


@always_inline
def _is_nan_d(x: Float64) -> Bool:
    """On the bits: the GPU build's fast-math flags may fold `x != x`."""
    return (x.to_bits[DType.uint64]() & UInt64(0x7FFFFFFFFFFFFFFF)) > UInt64(
        0x7FF0000000000000
    )


@always_inline
def _copysign_d(mag: Float64, sgn: Float64) -> Float64:
    return Float64(
        from_bits=(mag.to_bits[DType.uint64]() & UInt64(0x7FFFFFFFFFFFFFFF))
        | (sgn.to_bits[DType.uint64]() & UInt64(0x8000000000000000))
    )


@always_inline
def _hi_word(x: Float64) -> Int32:
    return Int32(Int(x.to_bits[DType.uint64]() >> UInt64(32)) & 0xFFFFFFFF)


@always_inline
def _add_rn_d(a: Float64, b: Float64) -> Float64:
    """add.rn.f64: a rounded sum Mojo may not fold into a neighbouring fma."""
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
def _rcp_approx_ftz_d(x: Float64) -> Float64:
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "rcp.approx.ftz.f64 $0, $1;",
            Float64,
            constraints="=d,d",
            has_side_effect=False,
        ](x)
    else:
        return 1 / x


@always_inline
def nv_expm1(a: Float64) -> Float64:
    """`__nv_expm1`: a * log2(e) rounded to k, a Cody-Waite residual, a
    degree-11 polynomial for expm1 of it, scaled back by 2**k; -1 / +inf
    past the range, the argument itself at +-0 (and tiny denormals)."""
    var hi = _hi_word(a)
    var mag = hi & Int32(0x7FFFFFFF)
    # CUDA compares the high word as a float: a < 709.78 and a > -37.43
    # (NaN / inf high words fail both).
    var in_range = (hi >= 0 and hi < Int32(1082535491)) or (
        hi < 0 and mag < Int32(0x40495C00)
    )
    if not in_range:
        if _is_nan_d(a):
            return a + a
        return Float64(-1.0) if hi < 0 else Float64(
            from_bits=UInt64(0x7FF0000000000000)
        )
    var t = fma(
        a,
        Float64(from_bits=UInt64(0x3FF71547652B82FE)),
        Float64(from_bits=UInt64(0x4338000000000000)),
    )
    var i = Int32(Int(t.to_bits[DType.uint64]() & UInt64(0xFFFFFFFF)))
    var k = _add_rn_d(t, Float64(from_bits=UInt64(0xC338000000000000)))
    var z = fma(k, Float64(from_bits=UInt64(0xBFE62E42FEFA39EF)), a)
    z = fma(k, Float64(from_bits=UInt64(0xBC7ABC9E3B39803F)), z)
    var hi2 = UInt32(Int(hi) & 0xFFFFFFFF) + UInt32(Int(hi) & 0xFFFFFFFF)
    if hi2 < UInt32(2142496327):
        z = a
        i = 0
    comptime P = [
        UInt64(0x3E5AF86D8EBD13CD),
        UInt64(0x3E927E5092BA033D),
        UInt64(0x3EC71DDE6C5F9DA1),
        UInt64(0x3EFA01A018D034E6),
        UInt64(0x3F2A01A01B3B6940),
        UInt64(0x3F56C16C16C1B5DD),
        UInt64(0x3F8111111110F74D),
        UInt64(0x3FA555555555554D),
        UInt64(0x3FC5555555555557),
    ]
    comptime p0 = P[0]
    var p = fma(
        Float64(from_bits=UInt64(0x3E21F4076ACD15B6)), z, Float64(from_bits=p0)
    )
    comptime for j in range(1, 9):
        comptime pj = P[j]
        p = fma(p, z, Float64(from_bits=pj))
    p = fma(p, z, Float64(0.5))
    var q = p * z
    var r = fma(q, z, z)
    var j = i - 1 if i == 1024 else i
    var sc = Float64(
        from_bits=UInt64(Int((j << 20) + Int32(1072693248)) & 0xFFFFFFFF)
        << UInt64(32)
    )
    var res = fma(r, sc, sc - Float64(1.0))
    if i == 1024:
        res = res + res
    if hi2 == 0:
        res = z
    return res


@always_inline
def nv_tanh(a: Float64) -> Float64:
    """`__nv_tanh`: an odd polynomial below |a| = 0.6552, else
    1 - 2 / (1 + e**(2|a|)) with the exponential split as in `nv_erf` and a
    refined reciprocal; 1 from |a| >= 19.06 on; the sign of a restored."""
    if _is_nan_d(a):
        return a + a
    var ax = llvm_intrinsic["llvm.fabs", Float64, has_side_effect=False](a)
    if ax >= Float64(from_bits=UInt64(0x3FE4F92224DD2F1A)):
        var t2 = Float64(2.0) * ax
        var k = _round_away(t2.cast[DType.float32]() * _f(0x3FB8AA3B))
        var r = fma(
            -k.cast[DType.float64](),
            Float64(from_bits=UInt64(0x3FE62E42FEFA39EF)),
            t2,
        )
        comptime E = [
            UInt64(0x3E928A27F89B6999),
            UInt64(0x3EC71DE715FF7E07),
            UInt64(0x3EFA019A6B0AC45A),
            UInt64(0x3F2A01A017EED94F),
            UInt64(0x3F56C16C17F2A71B),
            UInt64(0x3F811111111173C4),
            UInt64(0x3FA555555555211A),
            UInt64(0x3FC5555555555540),
            UInt64(0x3FE0000000000005),
        ]
        comptime e0 = E[0]
        var q = fma(
            Float64(from_bits=UInt64(0x3E5AE904A4741B81)),
            r,
            Float64(from_bits=e0),
        )
        comptime for i in range(1, 9):
            comptime ei = E[i]
            q = fma(q, r, Float64(from_bits=ei))
        var u = fma(q * r, r, r)
        var s = _ex2_approx_ftz(k).cast[DType.float64]()
        var e = fma(-u, s, Float64(1.0) - s)
        var d = Float64(2.0) - e
        var rc = _rcp_approx_ftz_d(d)
        var nr = fma(-d, rc, Float64(1.0))
        nr = fma(nr, nr, nr)
        rc = fma(nr, rc, rc)
        var res = fma(Float64(2.0), -rc, Float64(1.0))
        if (_hi_word(a) & Int32(0x7FFFFFFF)) >= Int32(1077088194):
            res = Float64(1.0)
        return _copysign_d(res, a)
    var x2 = a * a
    comptime S = [
        UInt64(0xBF2DF9F0728C5D84),
        UInt64(0x3F4337D1CEC4F033),
        UInt64(0xBF57D6E9674335B3),
        UInt64(0x3F6D6D000D7AAD3D),
        UInt64(0xBF8226E1F3CF1EF5),
        UInt64(0x3F9664F47EC0C8CF),
        UInt64(0xBFABA1BA1B80AB40),
        UInt64(0x3FC111111110FA4A),
        UInt64(0xBFD5555555555550),
    ]
    var p = fma(
        Float64(from_bits=UInt64(0xBEF0BC46E2F5E964)),
        x2,
        Float64(from_bits=UInt64(0x3F14359F420AFC3D)),
    )
    comptime for i in range(9):
        comptime si = S[i]
        p = fma(p, x2, Float64(from_bits=si))
    p = fma(p, x2, Float64(0.0))
    return fma(p, a, a)


# --------------------------------------------------------------------------- #
# sinf / cosf
# --------------------------------------------------------------------------- #


@always_inline
def _trig_reduce_f(a: Float32) -> Tuple[Float32, Int32]:
    """The argument reduction `__nv_sinf` / `__nv_cosf` share: a * 2/pi to the
    nearest quadrant, a three-term Cody-Waite residual, Payne-Hanek from
    |a| >= 105615 on."""
    var j = _f2i_rn(a * _f(0x3F22F983))
    var jf = j.cast[DType.float32]()
    var t = fma(jf, _f(0xBFC90FDA), a)
    t = fma(jf, _f(0xB3A22168), t)
    t = fma(jf, _f(0xA7C234C5), t)
    if _abs(a) >= Float32(105615.0):
        if _abs(a) == _INF:
            return (_mul_rn(a, Float32(0.0)), Int32(0))
        return _trig_reduction_slowpath_f(a)
    return (t, j)


@always_inline
def _sin_poly(t: Float32, i: Int32) -> Float32:
    """`__internal_sin_cos_kernel`: sin(t) for even quadrants, cos(t) for odd
    ones, negated in quadrants 2 and 3."""
    var t2 = _mul_rn(t, t)
    var even = (i & Int32(1)) == Int32(0)
    var s = t if even else Float32(1.0)
    var u = fma(t2, s, Float32(0.0))
    var c_odd = fma(_f(0x37CBAC00), t2, _f(0xBAB607ED))
    var c2 = _f(0xBE2AAAA8) if even else _f(0xBEFFFFFF)
    var c1 = _f(0x3C0885E4) if even else _f(0x3D2AAABB)
    var c0 = _f(0xB94D4153) if even else c_odd
    var p = fma(c0, t2, c1)
    p = fma(p, t2, c2)
    var z = fma(p, u, s)
    if (i & Int32(2)) != Int32(0):
        z = fma(z, Float32(-1.0), Float32(0.0))
    return z


@always_inline
def nv_sinf(a: Float32) -> Float32:
    """`__nv_sinf`."""
    var r = _trig_reduce_f(a)
    return _sin_poly(r[0], r[1])


@always_inline
def nv_cosf(a: Float32) -> Float32:
    """`__nv_cosf`: the sine kernel one quadrant on."""
    var r = _trig_reduce_f(a)
    return _sin_poly(r[0], r[1] + Int32(1))


# --------------------------------------------------------------------------- #
# lgammaf
# --------------------------------------------------------------------------- #


@always_inline
def _lgammaf_pos_one_log(x: Float32) -> Float32:
    """`_lgammaf_pos` with its two logarithms (of 1 / Gamma(x) below 0.7,
    of x in the Stirling tail) folded into one call on a selected argument.
    The GPU build if-converts the x < 3 branches (every polynomial runs for
    every element), and it hoisted the Stirling log out of its branch as
    well: two logs per element where one is used (lgamma f32 16M: 132 us,
    torch 87). Same operations on the value each case uses."""
    var p7 = fma(_f(0x3B6B1C86), x, _f(0xBBB34878))
    p7 = fma(p7, x, _f(0xBD36CAEF))
    p7 = fma(p7, x, _f(0x3E2B5555))
    p7 = fma(p7, x, _f(0xBD2C96C7))
    p7 = fma(p7, x, _f(0xBF27E6EB))
    p7 = fma(p7, x, _f(0x3F13C463))
    p7 = x * p7
    var g = fma(p7, x, x)  # 1 / Gamma(x), for x < 0.7
    var low = x < _f(0x3F333333)  # 0.7
    var l = nv_logf(g if low else x)
    if x < Float32(3.0):
        if x < Float32(1.5):
            if low:
                return -l
            var y = Float32(1.0) - x
            var p = fma(_f(0x3D3BEF76), y, _f(0x3DD47577))
            p = fma(p, y, _f(0x3DFB8079))
            p = fma(p, y, _f(0x3E0295B5))
            p = fma(p, y, _f(0x3E12A765))
            p = fma(p, y, _f(0x3E2D6867))
            p = fma(p, y, _f(0x3E5462BF))
            p = fma(p, y, _f(0x3E8A8A72))
            p = fma(p, y, _f(0x3ECD26A4))
            p = fma(p, y, _f(0x3F528D32))
            p = fma(p, y, _f(0x3F13C468))
            return y * p
        var y = x + Float32(-2.0)
        var p = fma(_f(0x385007FA), y, _f(0xB967A002))
        p = fma(p, y, _f(0x3A0DE6FC))
        p = fma(p, y, _f(0xBA9DE0E2))
        p = fma(p, y, _f(0x3B3D05B7))
        p = fma(p, y, _f(0xBBF1EB10))
        p = fma(p, y, _f(0x3CA89A28))
        p = fma(p, y, _f(0xBD89F01A))
        p = fma(p, y, _f(0x3EA51A66))
        p = fma(p, y, _f(0x3ED87730))
        return y * p
    if x < _f(0x40F9999A):  # 7.8
        var y = x + Float32(-3.0)
        var n = fma(_f(0xC43B38FB), y, _f(0xC640F6F8))
        n = fma(n, y, _f(0xC7206560))
        n = fma(n, y, _f(0xC73CB6AA))
        n = fma(n, y, _f(0xC80BAE5A))
        var d = y + _f(0xC381A020)
        d = fma(d, y, _f(0xC62864B8))
        d = fma(d, y, _f(0xC7B50686))
        d = fma(d, y, _f(0xC8498465))
        return fma(n, _rcp_approx_ftz(d), y)
    var r = _rcp_approx_ftz(x)
    var r2 = r * r
    var s = fma(_f(0x3A4BE755), r2, _f(0xBB360953))
    s = fma(s, r2, _f(0x3DAAAAA3))
    s = fma(s, r, _f(0x3F6B3F8E))
    var hl = l * Float32(0.5)
    var a = _mul_rn(hl, x + Float32(-0.5))
    var res = (a - x) + _add_rn(a, s)
    if x == _INF:
        res = _INF
    return res


@always_inline
def _lgammaf_pos(x: Float32) -> Float32:
    """`__internal_lgammaf_pos`: lgamma(|a|)."""
    comptime if is_nvidia_gpu():
        return _lgammaf_pos_one_log(x)
    if x < Float32(3.0):
        if x < Float32(1.5):
            if x < _f(0x3F333333):  # 0.7
                var p = fma(_f(0x3B6B1C86), x, _f(0xBBB34878))
                p = fma(p, x, _f(0xBD36CAEF))
                p = fma(p, x, _f(0x3E2B5555))
                p = fma(p, x, _f(0xBD2C96C7))
                p = fma(p, x, _f(0xBF27E6EB))
                p = fma(p, x, _f(0x3F13C463))
                p = x * p
                var g = fma(p, x, x)  # 1 / Gamma(x)
                return -nv_logf(g)
            var y = Float32(1.0) - x
            var p = fma(_f(0x3D3BEF76), y, _f(0x3DD47577))
            p = fma(p, y, _f(0x3DFB8079))
            p = fma(p, y, _f(0x3E0295B5))
            p = fma(p, y, _f(0x3E12A765))
            p = fma(p, y, _f(0x3E2D6867))
            p = fma(p, y, _f(0x3E5462BF))
            p = fma(p, y, _f(0x3E8A8A72))
            p = fma(p, y, _f(0x3ECD26A4))
            p = fma(p, y, _f(0x3F528D32))
            p = fma(p, y, _f(0x3F13C468))
            return y * p
        var y = x + Float32(-2.0)
        var p = fma(_f(0x385007FA), y, _f(0xB967A002))
        p = fma(p, y, _f(0x3A0DE6FC))
        p = fma(p, y, _f(0xBA9DE0E2))
        p = fma(p, y, _f(0x3B3D05B7))
        p = fma(p, y, _f(0xBBF1EB10))
        p = fma(p, y, _f(0x3CA89A28))
        p = fma(p, y, _f(0xBD89F01A))
        p = fma(p, y, _f(0x3EA51A66))
        p = fma(p, y, _f(0x3ED87730))
        return y * p
    if x < _f(0x40F9999A):  # 7.8
        var y = x + Float32(-3.0)
        var n = fma(_f(0xC43B38FB), y, _f(0xC640F6F8))
        n = fma(n, y, _f(0xC7206560))
        n = fma(n, y, _f(0xC73CB6AA))
        n = fma(n, y, _f(0xC80BAE5A))
        var d = y + _f(0xC381A020)
        d = fma(d, y, _f(0xC62864B8))
        d = fma(d, y, _f(0xC7B50686))
        d = fma(d, y, _f(0xC8498465))
        return fma(n, _rcp_approx_ftz(d), y)
    # Stirling: (x - 1/2) log x - x + log(2 pi)/2 + series(1/x).
    var r = _rcp_approx_ftz(x)
    var r2 = r * r
    var s = fma(_f(0x3A4BE755), r2, _f(0xBB360953))
    s = fma(s, r2, _f(0x3DAAAAA3))
    s = fma(s, r, _f(0x3F6B3F8E))
    var hl = nv_logf(x) * Float32(0.5)
    var a = _mul_rn(hl, x + Float32(-0.5))
    var res = (a - x) + _add_rn(a, s)
    if x == _INF:
        res = _INF
    return res


@always_inline
def nv_lgammaf(a: Float32) -> Float32:
    """`__nv_lgammaf`: the positive kernel, and the reflection
    log(pi / |x sin(pi x)|) - lgamma(|x|) for negative non-integers."""
    var x = _abs(a)
    var t = _lgammaf_pos(x)
    if not (a < Float32(0.0)):
        return t
    if x == _floor(x):
        return _INF
    if x < _f(0x1FEC1E4A):  # 1e-19
        comptime if is_nvidia_gpu():
            # t is -log(1 / Gamma(x)) there, and 1 / Gamma(x) rounds to x
            # exactly below 1e-19: the same value, without a second log the
            # GPU build hoists out of this rare branch into every element.
            return t
        return -nv_logf(x)
    # sin(pi x) by the sine kernel on a quadrant reduction of 2x.
    var q = _round_away(x * Float32(2.0))
    var qi = q.cast[DType.int32]()
    var r = fma(-q, Float32(0.5), x) * _f(0x40490FDB)
    var s = _abs(_sin_poly(r, qi))
    return (_f(0x3F928682) - nv_logf(x * s)) - t
