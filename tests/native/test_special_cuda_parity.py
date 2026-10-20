"""The special functions against stock CUDA, bit for bit (the live oracle,
tests/native/cuda_oracle.py): erf/erfc/erfinv, the gamma family (lgamma,
digamma, polygamma, mvlgamma, igamma/igammac, _standard_gamma_grad), the
Bessel / Airy / error-function family of torch.special, sinc, logit (and
its backward), xlogy / xlog1py, zeta and the orthogonal polynomials.

Every dtype CUDA computes is covered, float64 included, and every dtype it
refuses must be refused too (the same exception type). float16 / bfloat16
take all 65536 bit patterns; float32 / float64 take the special values,
every power of two with its neighbours, pseudo-random bit patterns, a
log-uniform sweep and dense sweeps over each function's branch points.
"""

import math
from collections.abc import Callable

import pytest
import torch

from tests.native.cuda_oracle import CudaOracle

S = torch.special
_F = (torch.float32, torch.float16, torch.bfloat16, torch.float64)
_INTS = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool)
_BITS = {torch.float32: torch.int32, torch.float64: torch.int64}


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _from_bits(bits: list[int] | torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    int_dtype = _BITS[dtype]
    width = torch.iinfo(int_dtype).bits
    if isinstance(bits, torch.Tensor) and width == 32:
        bits = bits.to(torch.int64)
        bits = torch.where(bits >= 1 << 31, bits - (1 << 32), bits)
    if not isinstance(bits, torch.Tensor):
        # unsigned patterns -> the signed integers of the same bits
        bits = torch.tensor(
            [b - (1 << width) if b >= 1 << (width - 1) else b for b in bits],
            dtype=int_dtype,
        )
    return bits.to(int_dtype).view(dtype)


def _special_bits(dtype: torch.dtype) -> list[int]:
    """Zeros, infinities, NaNs with several payloads (quiet, signalling,
    negative), denormals, the normal boundary and the largest finite."""
    if dtype == torch.float64:
        pos = [
            0x0,
            0x7FF0000000000000,
            0x7FF8000000000000,
            0x7FFFFFFFFFFFFFFF,
            0x7FF0000000000001,
            0x7FF4000000000000,
            0x1,
            0xFFFFFFFFFFFFF,
            0x10000000000000,
            0x7FEFFFFFFFFFFFFF,
        ]
        sign = 1 << 63
    else:
        pos = [
            0x0,
            0x7F800000,
            0x7FC00000,
            0x7FFFFFFF,
            0x7F800001,
            0x7FA00000,
            0x1,
            0x7FFFFF,
            0x800000,
            0x7F7FFFFF,
        ]
        sign = 1 << 31
    return pos + [b | sign for b in pos]


def float_inputs(
    dtype: torch.dtype, sweeps: tuple[tuple[float, float], ...] = ()
) -> torch.Tensor:
    """The inputs of a float `dtype` (see the module docstring); `sweeps`
    are (lo, hi) intervals sampled densely on top."""
    if dtype in (torch.float16, torch.bfloat16):
        return (
            torch.arange(-(1 << 15), 1 << 15, dtype=torch.int32)
            .to(torch.int16)
            .view(dtype)
        )
    is64 = dtype == torch.float64
    mant = 52 if is64 else 23
    top = 2046 if is64 else 254
    parts = [_from_bits(_special_bits(dtype), dtype)]
    # every power of two (subnormal ones too) with both neighbours, both signs
    exps = torch.arange(1, top + 1, dtype=torch.int64) << mant
    sub = torch.tensor([1 << k for k in range(mant)], dtype=torch.int64)
    pw = torch.cat([exps, sub])
    pw = torch.cat([pw - 1, pw, pw + 1])
    # the negatives: the same bits with the sign bit, as signed integers
    neg = pw - (1 << 63) if is64 else pw - (1 << 31)
    parts.append(_from_bits(torch.cat([pw, neg]), dtype))
    g = _gen(1234)
    n = 1 << 15
    if is64:
        rb = torch.randint(-(2**63), 2**63 - 1, (n,), generator=g, dtype=torch.int64)
    else:
        rb = torch.randint(-(2**31), 2**31 - 1, (n,), generator=g, dtype=torch.int64)
        rb = rb.to(torch.int32)
    parts.append(_from_bits(rb, dtype))
    lim = 700.0 if is64 else 87.0
    mag = torch.empty(n, dtype=torch.float64).uniform_(-lim, lim, generator=g).exp()
    sgn = torch.randint(0, 2, (n,), generator=g).to(torch.float64) * 2 - 1
    parts.append((mag * sgn).to(dtype))
    ints = torch.arange(-60, 61, dtype=torch.float64)
    parts.append(torch.cat([ints, ints + 0.5, ints + 1e-3, ints - 1e-3]).to(dtype))
    for lo, hi in sweeps:
        pts = torch.empty(1 << 13, dtype=torch.float64).uniform_(lo, hi, generator=g)
        parts.append(
            torch.cat([pts, torch.linspace(lo, hi, 1 << 12, dtype=torch.float64)]).to(
                dtype
            )
        )
    return torch.cat([p.to(dtype) for p in parts])


def int_inputs(dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.bool:
        return torch.tensor([True, False, True])
    info = torch.iinfo(dtype)
    vals = torch.arange(max(info.min, -300), min(info.max, 300) + 1)
    return torch.cat([vals, torch.tensor([info.min, info.max])]).to(dtype)


def _cuda_refuses_half(dtype: torch.dtype, half_ok: bool) -> bool:
    return not half_ok and dtype in (torch.float16, torch.bfloat16)


# (name, callable, in-place method, CUDA computes float16 / bfloat16,
# dense sweeps around the routine's branch points)
_UNARY: list[
    tuple[
        str,
        Callable[..., torch.Tensor],
        str | None,
        bool,
        tuple[tuple[float, float], ...],
    ]
] = [
    ("erf", torch.erf, "erf_", True, ((-6.5, 6.5),)),
    ("erfc", torch.erfc, "erfc_", True, ((-10.0, 30.0),)),
    ("erfinv", torch.erfinv, "erfinv_", True, ((-1.0, 1.0), (0.99, 1.0))),
    ("lgamma", torch.lgamma, "lgamma_", True, ((-40.0, 40.0), (0.0, 8.0))),
    ("digamma", torch.digamma, "digamma_", True, ((-40.0, 40.0),)),
    ("i0", torch.i0, "i0_", True, ((-100.0, 100.0),)),
    ("sinc", torch.sinc, "sinc_", True, ((-30.0, 30.0),)),
    ("logit", torch.logit, "logit_", True, ((-0.5, 1.5),)),
    ("ndtr", S.ndtr, None, True, ((-40.0, 40.0),)),
    ("entr", S.entr, None, True, ((-1.0, 30.0),)),
    ("erfcx", S.erfcx, None, False, ((-30.0, 60.0), (-7.0, 7.0))),
    ("i0e", S.i0e, None, True, ((-100.0, 100.0),)),
    ("i1", S.i1, None, True, ((-100.0, 100.0),)),
    ("i1e", S.i1e, None, True, ((-100.0, 100.0),)),
    ("ndtri", S.ndtri, None, False, ((0.0, 1.0), (0.0, 0.2), (0.8, 1.0))),
    ("log_ndtr", S.log_ndtr, None, False, ((-40.0, 10.0),)),
    ("airy_ai", S.airy_ai, None, False, ((-30.0, 30.0), (-3.0, 10.0))),
    ("bessel_j0", S.bessel_j0, None, False, ((-40.0, 40.0),)),
    ("bessel_j1", S.bessel_j1, None, False, ((-40.0, 40.0),)),
    ("bessel_y0", S.bessel_y0, None, False, ((0.0, 40.0),)),
    ("bessel_y1", S.bessel_y1, None, False, ((0.0, 40.0),)),
    ("modified_bessel_i0", S.modified_bessel_i0, None, False, ((-100.0, 100.0),)),
    ("modified_bessel_i1", S.modified_bessel_i1, None, False, ((-100.0, 100.0),)),
    (
        "modified_bessel_k0",
        S.modified_bessel_k0,
        None,
        False,
        ((0.0, 100.0), (0.0, 3.0)),
    ),
    (
        "modified_bessel_k1",
        S.modified_bessel_k1,
        None,
        False,
        ((0.0, 100.0), (0.0, 3.0)),
    ),
    (
        "scaled_modified_bessel_k0",
        S.scaled_modified_bessel_k0,
        None,
        False,
        ((0.0, 100.0), (0.0, 3.0)),
    ),
    (
        "scaled_modified_bessel_k1",
        S.scaled_modified_bessel_k1,
        None,
        False,
        ((0.0, 100.0), (0.0, 3.0)),
    ),
    ("spherical_bessel_j0", S.spherical_bessel_j0, None, False, ((-40.0, 40.0),)),
]
_UNARY_IDS = [c[0] for c in _UNARY]


@pytest.mark.parametrize("dtype", _F, ids=str)
@pytest.mark.parametrize(
    ("name", "fn", "method", "half_ok", "sweeps"), _UNARY, ids=_UNARY_IDS
)
def test_unary_special_matches_cuda(
    cuda_oracle: CudaOracle,
    name: str,
    fn: Callable[..., torch.Tensor],
    method: str | None,
    half_ok: bool,
    sweeps: tuple[tuple[float, float], ...],
    dtype: torch.dtype,
):
    x = float_inputs(dtype, sweeps)
    cuda_oracle.assert_bitwise_equal(fn, x)


@pytest.mark.parametrize("dtype", _INTS, ids=str)
@pytest.mark.parametrize(
    ("name", "fn", "method", "half_ok", "sweeps"), _UNARY, ids=_UNARY_IDS
)
def test_unary_special_integer_inputs_match_cuda(
    cuda_oracle: CudaOracle,
    name: str,
    fn: Callable[..., torch.Tensor],
    method: str | None,
    half_ok: bool,
    sweeps: tuple[tuple[float, float], ...],
    dtype: torch.dtype,
):
    cuda_oracle.assert_bitwise_equal(fn, int_inputs(dtype))


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128), ids=str)
@pytest.mark.parametrize(
    ("name", "fn", "method", "half_ok", "sweeps"), _UNARY, ids=_UNARY_IDS
)
def test_unary_special_complex_inputs_match_cuda(
    cuda_oracle: CudaOracle,
    name: str,
    fn: Callable[..., torch.Tensor],
    method: str | None,
    half_ok: bool,
    sweeps: tuple[tuple[float, float], ...],
    dtype: torch.dtype,
):
    """CUDA computes sinc on complex numbers and refuses the rest."""
    re = float_inputs(torch.float32, ((-5.0, 5.0),))[:4096]
    im = float_inputs(torch.float32, ((-5.0, 5.0),)).flip(0)[:4096]
    cuda_oracle.assert_bitwise_equal(fn, torch.complex(re, im).to(dtype))


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128), ids=str)
def test_sinc_complex_wide_range_and_variants_match_cuda(
    cuda_oracle: CudaOracle, dtype: torch.dtype
):
    g = _gen(5)
    mag = 10 ** (torch.rand(4096, generator=g) * 12 - 6)
    re = (torch.randn(4096, generator=g) * mag).to(torch.float32)
    im = (torch.randn(4096, generator=g) * mag.flip(0)).to(torch.float32)
    z = torch.complex(re, im).to(dtype)
    eq = cuda_oracle.assert_bitwise_equal
    eq(torch.sinc, z, what="wide range")

    def variants(t: torch.Tensor) -> object:
        out = torch.empty_like(t)
        torch.sinc(t, out=out)
        strided = torch.zeros(2 * t.numel(), dtype=t.dtype, device=t.device)[::2]
        torch.sinc(t, out=strided)
        return out, strided, t.clone().sinc_(), t.t().sinc()

    eq(variants, z.reshape(64, 64), what="out= / in-place / transposed")


def _variants(fn: Callable[..., torch.Tensor], method: str | None, dtype: torch.dtype):
    """out= (dense, strided, another dtype), a transposed and an offset
    input, the in-place method on a view, a 0-dim and an empty input."""
    base = float_inputs(torch.float32, ((-3.0, 3.0),))[: 64 * 33].reshape(64, 33)

    def run(device: str) -> object:
        x = base.to(device=device, dtype=dtype)
        outs: dict[str, object] = {"t": fn(x.t()), "offset": fn(x.flatten()[1:])}
        res = fn(x)
        o = torch.empty_like(res)
        fn(x, out=o)
        outs["out"] = o
        o64 = torch.zeros(64, 33, dtype=torch.float64, device=device)
        fn(x, out=o64)
        outs["out64"] = o64
        strided = torch.full((33, 64), -7.0, dtype=res.dtype, device=device).t()
        fn(x, out=strided)
        outs["strided"] = strided
        outs["scalar"] = fn(x[3, 4].clone())
        outs["empty"] = fn(x[:0])
        if method is not None and dtype.is_floating_point:
            y = x.clone()
            getattr(y[:, 1:], method)()
            outs["inplace"] = y
        return outs

    return run


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.int32), ids=str)
@pytest.mark.parametrize(
    ("name", "fn", "method", "half_ok", "sweeps"), _UNARY, ids=_UNARY_IDS
)
def test_unary_special_variants_match_cuda(
    cuda_oracle: CudaOracle,
    name: str,
    fn: Callable[..., torch.Tensor],
    method: str | None,
    half_ok: bool,
    sweeps: tuple[tuple[float, float], ...],
    dtype: torch.dtype,
):
    cuda_oracle.assert_bitwise_equal_by_device(_variants(fn, method, dtype))


# --------------------------------------------------------------------------
# parameterized unary ops: logit(eps), polygamma(n), mvlgamma(p)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _F, ids=str)
@pytest.mark.parametrize("eps", [None, 0.0, 1e-6, 0.25, 0.5, 0.75, -0.5])
def test_logit_eps_matches_cuda(
    cuda_oracle: CudaOracle, eps: float | None, dtype: torch.dtype
):
    x = float_inputs(dtype, ((-0.5, 1.5),))
    cuda_oracle.assert_bitwise_equal(lambda t: torch.logit(t, eps), x)
    cuda_oracle.assert_bitwise_equal(lambda t: t[1:].logit_(eps), x)


@pytest.mark.parametrize("dtype", (*_F, torch.int32), ids=str)
@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 7, 30])
def test_polygamma_matches_cuda(cuda_oracle: CudaOracle, n: int, dtype: torch.dtype):
    if dtype.is_floating_point:
        x = float_inputs(dtype, ((-20.0, 30.0), (0.0, 3.0)))
    else:
        x = int_inputs(dtype)
    cuda_oracle.assert_bitwise_equal(lambda t: torch.polygamma(n, t), x)
    if dtype.is_floating_point:
        cuda_oracle.assert_bitwise_equal(lambda t: t.polygamma_(n), x)


@pytest.mark.parametrize("dtype", (*_F, torch.int64), ids=str)
@pytest.mark.parametrize("p", [1, 2, 5])
def test_mvlgamma_matches_cuda(cuda_oracle: CudaOracle, p: int, dtype: torch.dtype):
    lo = (p - 1) / 2
    g = _gen(p)
    if dtype.is_floating_point:
        x = torch.empty(1 << 14, dtype=torch.float64).uniform_(
            lo + 0.01, lo + 60, generator=g
        )
        x = x.to(dtype)
        x = x[x > lo]
    else:
        x = torch.arange(int(lo) + 1, int(lo) + 300).to(dtype)
    cuda_oracle.assert_bitwise_equal(lambda t: torch.mvlgamma(t, p), x)
    # below (p - 1) / 2 torch 2.11 has no domain check (CPU and CUDA return lgamma's values)
    cuda_oracle.assert_bitwise_equal(lambda t: torch.mvlgamma(t, p), x - 100)


# --------------------------------------------------------------------------
# binary ops
# --------------------------------------------------------------------------

_SPECIAL_VALUES = [
    0.0,
    -0.0,
    math.inf,
    -math.inf,
    math.nan,
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.0,
    -2.0,
    3.0,
    1e-30,
    -1e-30,
    1e30,
    -1e30,
    1e-3,
    0.999,
    1.001,
    5.0,
    7.5,
    10.0,
    20.0,
    30.5,
    100.0,
    1e4,
    -3.5,
    -10.0,
    1e-45,
    6.0,
    0.1,
    200.0,
    250.0,
    1e-8,
]


def pair_inputs(
    dtype: torch.dtype, a_range: tuple[float, float], b_range: tuple[float, float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """The cartesian product of special values, pairs sampled in the given
    ranges, and log-uniform pairs of both signs."""
    sp = torch.tensor(_SPECIAL_VALUES, dtype=torch.float64)
    a = [sp.repeat_interleave(len(sp))]
    b = [sp.repeat(len(sp))]
    g = _gen(7)
    n = 1 << 15
    a.append(torch.empty(n, dtype=torch.float64).uniform_(*a_range, generator=g))
    b.append(torch.empty(n, dtype=torch.float64).uniform_(*b_range, generator=g))
    for side in (a, b):
        mag = torch.empty(n, dtype=torch.float64).uniform_(-30, 30, generator=g).exp()
        sgn = (
            torch.randint(0, 4, (n,), generator=g).to(torch.float64).clamp(max=1) * 2
            - 1
        )
        side.append(mag * sgn)
    if dtype in (torch.float16, torch.bfloat16):
        # every pair of 256 patterns spread over the whole range
        pat = (torch.arange(256, dtype=torch.int32) * 257).to(torch.int16).view(dtype)
        a.append(pat.repeat_interleave(256).to(torch.float64))
        b.append(pat.repeat(256).to(torch.float64))
    if not dtype.is_floating_point:
        return (
            torch.cat(a).nan_to_num(0).clamp(-100, 100).to(dtype),
            torch.cat(b).nan_to_num(0).clamp(-100, 100).to(dtype),
        )
    return torch.cat(a).to(dtype), torch.cat(b).to(dtype)


_BINARY: list[
    tuple[
        str, Callable[..., torch.Tensor], bool, tuple[float, float], tuple[float, float]
    ]
] = [
    ("xlogy", torch.xlogy, True, (-10.0, 10.0), (-10.0, 10.0)),
    ("xlog1py", S.xlog1py, True, (-10.0, 10.0), (-2.0, 10.0)),
    ("zeta", S.zeta, False, (0.0, 30.0), (-10.0, 30.0)),
    ("igamma", torch.igamma, False, (0.0, 50.0), (0.0, 50.0)),
    ("igammac", torch.igammac, False, (0.0, 50.0), (0.0, 50.0)),
    ("igamma_large", torch.igamma, False, (100.0, 400.0), (50.0, 500.0)),
    ("igammac_large", torch.igammac, False, (100.0, 400.0), (50.0, 500.0)),
]
_BINARY_IDS = [c[0] for c in _BINARY]


@pytest.mark.parametrize("dtype", (*_F, torch.int32, torch.bool), ids=str)
@pytest.mark.parametrize(
    ("name", "fn", "half_ok", "a_range", "b_range"), _BINARY, ids=_BINARY_IDS
)
def test_binary_special_matches_cuda(
    cuda_oracle: CudaOracle,
    name: str,
    fn: Callable[..., torch.Tensor],
    half_ok: bool,
    a_range: tuple[float, float],
    b_range: tuple[float, float],
    dtype: torch.dtype,
):
    a, b = pair_inputs(dtype, a_range, b_range)
    cuda_oracle.assert_bitwise_equal(fn, a, b)


@pytest.mark.parametrize(
    ("name", "fn", "half_ok", "a_range", "b_range"), _BINARY, ids=_BINARY_IDS
)
def test_binary_special_scalars_and_promotion_match_cuda(
    cuda_oracle: CudaOracle,
    name: str,
    fn: Callable[..., torch.Tensor],
    half_ok: bool,
    a_range: tuple[float, float],
    b_range: tuple[float, float],
):
    """Python scalars on either side (the CPU-scalar route of the kernel),
    0-dim tensors, mixed dtypes, broadcasting, out= and in-place."""
    a, b = pair_inputs(torch.float32, a_range, b_range)
    a, b = a[:4096], b[:4096]
    eq = cuda_oracle.assert_bitwise_equal
    for s in (2.5, 3, True, -0.0, math.nan, 1e-3):
        eq(lambda t, s=s: fn(t, s), b, what=f"(tensor, {s!r})")
        eq(lambda t, s=s: fn(s, t), a, what=f"({s!r}, tensor)")
    eq(lambda t: fn(t, torch.tensor(2.5, device=t.device)), a, what="0-dim f32")
    eq(
        lambda t: fn(t, torch.tensor(2.5, dtype=torch.float64, device=t.device)),
        a,
        what="0-dim f64",
    )
    eq(
        lambda t: fn(torch.tensor(0.75, dtype=torch.float64, device=t.device), t),
        b,
        what="0-dim f64 lhs",
    )
    eq(
        lambda t: fn(t.half(), torch.tensor(2.5, device=t.device)),
        a,
        what="f16, 0-dim f32",
    )
    eq(lambda t, u: fn(t.half(), u), a, b, what="f16, f32")
    # xlogy / xlog1py in float64 with a NaN x and a log(y) that is NaN: the
    # sign of the result depends on the CUDA launch path (contiguous
    # vectorized kernel: log's -NaN; the casting/strided kernel: x's +NaN),
    # because ptxas orders the dmul operands per instantiation. Probed on
    # 4 vs 66692 elements. The contiguous case is exact in
    # test_binary_special_matches_cuda; here only the NaN sign may differ.
    eq(
        lambda t, u: fn(t.double(), u),
        a,
        b,
        nan="any" if name in ("xlogy", "xlog1py") else "exact",
        what="f64, f32",
    )
    eq(
        lambda t, u: fn(t.to(torch.int32), u),
        a.nan_to_num(0).clamp(-50, 50),
        b,
        what="i32, f32",
    )
    eq(
        lambda t, u: fn(t.reshape(64, 64), u.reshape(64, 64)[:1]),
        a,
        b,
        what="broadcast",
    )
    eq(
        lambda t, u: fn(t.reshape(64, 64).t(), u.reshape(64, 64)),
        a,
        b,
        what="transposed",
    )

    def out_variants(t: torch.Tensor, u: torch.Tensor) -> object:
        o = torch.empty_like(t)
        fn(t, u, out=o)
        o64 = torch.empty(t.shape, dtype=torch.float64, device=t.device)
        fn(t, u, out=o64)
        strided = torch.full((2 * t.numel(),), -7.0, device=t.device)[::2]
        fn(t, u, out=strided)
        return o, o64, strided

    eq(out_variants, a, b, what="out=")
    method = {"xlogy": "xlogy_", "igamma": "igamma_", "igammac": "igammac_"}.get(name)
    if method is not None:
        eq(lambda t, u: getattr(t, method)(u), a, b, what="in-place")
        eq(lambda t, u: getattr(t.half(), method)(u), a, b, what="in-place f16 self")


_POLY = [
    "chebyshev_polynomial_t",
    "chebyshev_polynomial_u",
    "chebyshev_polynomial_v",
    "chebyshev_polynomial_w",
    "shifted_chebyshev_polynomial_t",
    "shifted_chebyshev_polynomial_u",
    "shifted_chebyshev_polynomial_v",
    "shifted_chebyshev_polynomial_w",
    "hermite_polynomial_h",
    "hermite_polynomial_he",
    "laguerre_polynomial_l",
    "legendre_polynomial_p",
]


def poly_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """x: special values, [-2, 2] (the cos(n acos x) regime of |x| < 1
    included), [-50, 50]; n: integers from -2 to 40 plus non-integers,
    NaN and infinities (C++ casts n to int64)."""
    g = _gen(11)
    sp = torch.tensor(_SPECIAL_VALUES, dtype=torch.float64)
    x = torch.cat(
        [
            sp.repeat_interleave(12),
            torch.empty(1 << 14, dtype=torch.float64).uniform_(-2, 2, generator=g),
            torch.empty(1 << 12, dtype=torch.float64).uniform_(-50, 50, generator=g),
            torch.tensor([0.3, -0.3, 0.9, 1.0, -1.0, 0.0] * 8, dtype=torch.float64),
        ]
    )
    n_sp = torch.tensor([0, 1, 2, 3, 5, 6, 7, 12, 25, -1, -2, 40], dtype=torch.float64)
    n = torch.cat(
        [
            n_sp.repeat(len(_SPECIAL_VALUES)),
            torch.randint(-2, 41, (1 << 14,), generator=g).to(torch.float64),
            torch.randint(-2, 41, (1 << 12,), generator=g).to(torch.float64),
            torch.tensor(
                [1.5, 2.7, -0.5, math.nan, math.inf, -math.inf] * 8, dtype=torch.float64
            ),
        ]
    )
    return x.to(dtype), n.to(dtype)


@pytest.mark.parametrize("dtype", (*_F, torch.int64), ids=str)
@pytest.mark.parametrize("name", _POLY)
def test_polynomial_matches_cuda(
    cuda_oracle: CudaOracle, name: str, dtype: torch.dtype
):
    fn = getattr(S, name)
    x, n = poly_inputs(torch.float64)
    if dtype.is_floating_point:
        x, n = x.to(dtype), n.to(dtype)
    else:
        x, n = (
            x.nan_to_num(0).clamp(-50, 50).to(dtype),
            n.nan_to_num(0).clamp(-5, 40).to(dtype),
        )
    cuda_oracle.assert_bitwise_equal(fn, x, n)


@pytest.mark.parametrize("name", _POLY)
def test_polynomial_scalars_match_cuda(cuda_oracle: CudaOracle, name: str):
    fn = getattr(S, name)
    x, n = poly_inputs(torch.float32)
    eq = cuda_oracle.assert_bitwise_equal
    for k in (0, 1, 2, 3, 6, 7, 30, 2.5, -1):
        eq(lambda t, k=k: fn(t, k), x, what=f"n={k}")
    if name == "legendre_polynomial_p":
        # CUDA's recurrence never ends for |x| < 1 and an infinite degree
        # (int64 max steps of a bounded, never-NaN sequence).
        n = n.nan_to_num(0.0, 100.0, -100.0)
    for v in (0.3, -0.7, 2.0, 0.5):
        eq(lambda u, v=v: fn(v, u), n, what=f"x={v}")
    eq(lambda t, u: fn(t.double(), u), x, n, what="f64, f32")
    eq(
        lambda t, u: fn(t, u.to(torch.int64)),
        x,
        n.nan_to_num(0).clamp(-5, 40),
        what="f32, i64",
    )

    def out_variants(t: torch.Tensor, u: torch.Tensor) -> object:
        o = torch.empty_like(t)
        fn(t, u, out=o)
        o64 = torch.empty(t.shape, dtype=torch.float64, device=t.device)
        fn(t, u, out=o64)
        return o, o64

    eq(out_variants, x, n, what="out=")


# --------------------------------------------------------------------------
# backward-style binary ops
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", (*_F, torch.int32), ids=str)
def test_standard_gamma_grad_matches_cuda(cuda_oracle: CudaOracle, dtype: torch.dtype):
    alpha, x = pair_inputs(dtype, (0.0, 20.0), (0.0, 20.0))
    cuda_oracle.assert_bitwise_equal(torch.ops.aten._standard_gamma_grad, alpha, x)


@pytest.mark.parametrize("dtype", (*_F, torch.int32), ids=str)
@pytest.mark.parametrize("eps", [None, 0.0, 1e-6, 0.25, -0.5])
def test_logit_backward_matches_cuda(
    cuda_oracle: CudaOracle, eps: float | None, dtype: torch.dtype
):
    grad, x = pair_inputs(dtype, (-3.0, 3.0), (-0.5, 1.5))
    cuda_oracle.assert_bitwise_equal(
        lambda g_, t: torch.ops.aten.logit_backward(g_, t, eps), grad, x
    )
