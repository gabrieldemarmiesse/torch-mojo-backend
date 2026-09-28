"""Pointwise math and parameterized activations on the native mojo device
(tmb/ops/pointwise.mojo): pow, lerp (Scalar and Tensor), gelu_backward,
clamp.Tensor, rsub, deg2rad/rad2deg/ldexp/frexp and the binary math family
(atan2, hypot, copysign, fmod, fmax, fmin, heaviside, nextafter, gcd, lcm,
bitwise_left_shift.Tensor, bitwise_right_shift.Tensor, logaddexp(2), xlogy,
xlog1py, zeta, igamma/igammac, the special polynomials), and elu, hardtanh,
leaky_relu, softplus, threshold, hardshrink, softshrink, hardsigmoid,
hardswish, mish, logsigmoid, rrelu with their backwards, and logit_backward.

Everything is compared with the same computation on CPU torch through the
public API, over edge values (signed zeros, infinities, NaN, huge, tiny,
denormals), broadcasting, strided operands, out= and in-place forms.
"""

import contextlib
import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from tests.native.conftest import skip_if_metal
from torch_mojo_backend import get_accelerators, native

FLOATS = [torch.float32, torch.float16, torch.bfloat16]
INTS = [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8]


@contextlib.contextmanager
def ran(*op_names: str):
    native.op_counting(True)
    before = {name: native.op_count(name) for name in op_names}
    yield
    assert any(native.op_count(name) > before[name] for name in op_names), (
        f"none of {op_names} ran natively"
    )


_SPECIAL = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.5,
    -3.75,
    7.0,
    1e-30,
    -1e-30,
    1e30,
    -1e30,
    1e-40,
    float("inf"),
    float("-inf"),
    float("nan"),
]


def _tol(dtype: torch.dtype, ulps: float = 1.0) -> dict[str, float]:
    if dtype == torch.float32:
        return {"rtol": 1.3e-6 * ulps, "atol": 1e-5}
    if dtype == torch.float16:
        return {"rtol": 1e-3, "atol": 1e-5}
    if dtype == torch.bfloat16:
        return {"rtol": 1.6e-2, "atol": 1e-5}
    return {"rtol": 0.0, "atol": 0.0}


def _pairs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Every ordered pair of edge values, plus a random bulk."""
    a, b = zip(*itertools.product(_SPECIAL, _SPECIAL))
    torch.manual_seed(0)
    ra = torch.randn(300) * 4
    rb = torch.randn(300) * 4
    x = torch.cat([torch.tensor(a), ra]).to(dtype)
    y = torch.cat([torch.tensor(b), rb]).to(dtype)
    return x, y


def _flushes_subnormals(device: str) -> bool:
    """Apple GPUs flush float32 (and bfloat16) subnormals to zero in float
    arithmetic and compares, as torch MPS's own Metal kernels then do: a
    subnormal operand or result cannot match CPU torch there."""
    accelerators = list(get_accelerators())
    idx = int(device.rsplit(":", 1)[-1])
    return idx < len(accelerators) and accelerators[idx].api == "metal"


def _subnormal(t: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(t.dtype).tiny
    return (t != 0) & (t.abs() < tiny)


def _close(
    actual: torch.Tensor | None,
    expected: torch.Tensor | None,
    rtol: float | None = None,
    atol: float | None = None,
):
    assert actual is not None and expected is not None
    torch.testing.assert_close(
        actual.cpu(), expected, equal_nan=True, rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.uint8])
def test_integer_pow(mojo_gpu, dtype):
    torch.manual_seed(9)
    lo = 0 if dtype == torch.uint8 else -5
    base = torch.randint(lo, 6, (300,), dtype=dtype)
    expo = torch.randint(0, 9, (300,), dtype=dtype)
    with ran("aten::pow.Tensor_Tensor"):
        _close(torch.pow(base.to(mojo_gpu), expo.to(mojo_gpu)), torch.pow(base, expo))
    _close(torch.pow(base.to(mojo_gpu), 3), torch.pow(base, 3))
    _close(torch.pow(2, expo.to(mojo_gpu)), torch.pow(2, expo))
    if dtype != torch.uint8:
        neg = torch.tensor([-1, -2, -3, -1, 0], dtype=dtype)
        b = torch.tensor([1, -1, -1, 2, 5], dtype=dtype)
        _close(torch.pow(b.to(mojo_gpu), neg.to(mojo_gpu)), torch.pow(b, neg))
        with pytest.raises(RuntimeError, match="negative integer powers"):
            torch.pow(b.to(mojo_gpu), -2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lerp_scalar_half(mojo_gpu, dtype):
    torch.manual_seed(10)
    s = torch.randn(33, 5).to(dtype)
    e = torch.randn(33, 5).to(dtype)
    for w in (0.3, 0.7, 1.4):
        with ran("aten::lerp.Scalar"):
            actual = torch.lerp(s.to(mojo_gpu), e.to(mojo_gpu), w)
        _close(actual, torch.lerp(s, e, w), **_tol(dtype, 2))
    x, xc = s.to(mojo_gpu), s.clone()
    x.lerp_(e.to(mojo_gpu), 0.25)
    xc.lerp_(e, 0.25)
    _close(x, xc, **_tol(dtype, 2))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16, torch.bfloat16])
def test_lerp_scalar_inplace_never_resizes_self(mojo_gpu, dtype):
    """lerp_ writes into self: a broadcast shape larger than self raises,
    like torch, instead of resizing self (float32 keeps binary.mojo's route;
    every other dtype is the pointwise one)."""
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "Metal has no float64")
    a = torch.randn(1, 3, dtype=dtype)
    b = torch.randn(2, 3, dtype=dtype)
    with pytest.raises(RuntimeError):
        a.clone().lerp_(b, 0.5)
    x = a.to(mojo_gpu)
    with pytest.raises(RuntimeError):
        x.lerp_(b.to(mojo_gpu), 0.5)
    assert x.shape == (1, 3)
    _close(x, a)
    # A self whose elements share one memory location is rejected too.
    e = torch.randn(1, 3, dtype=dtype).to(mojo_gpu).expand(2, 3)
    with pytest.raises(RuntimeError):
        e.lerp_(b.to(mojo_gpu), 0.5)
    # Broadcasting `end` into a larger self still works.
    y, yc = b.to(mojo_gpu), b.clone()
    with ran("aten::lerp_.Scalar"):
        y.lerp_(a.to(mojo_gpu), 0.25)
    yc.lerp_(a, 0.25)
    _close(y, yc, **_tol(dtype, 2))


# C99 pow's special values (Annex F.9.4.4) and the saturating magnitudes, for
# the float64 double-double core: an overflow in the product y * log|x| must
# saturate to inf / 0, not become NaN in the double-double correction.
_POW64 = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.0,
    -2.0,
    3.0,
    -3.0,
    2.5,
    -2.5,
    10.0,
    -10.0,
    1e-300,
    1e300,
    -1e300,
    5e-324,
    1.7976931348623157e308,
    -1.7976931348623157e308,
    1e308,
    -1e308,
    9007199254740993.0,
    -9007199254740991.0,
    float("inf"),
    float("-inf"),
    float("nan"),
]


def test_pow_float64_special_values_saturate(mojo_gpu):
    skip_if_metal(mojo_gpu, "Metal has no float64")
    base, expo = zip(*[(x, y) for x in _POW64 for y in _POW64])
    x = torch.tensor(base, dtype=torch.float64)
    y = torch.tensor(expo, dtype=torch.float64)
    want = torch.pow(x, y)
    with ran("aten::pow.Tensor_Tensor"):
        got = torch.pow(x.to(mojo_gpu), y.to(mojo_gpu)).cpu()
    # Within an ulp of CPU's std::pow on the finite results...
    _close(got, want, rtol=2.3e-16, atol=0)
    # ...and exact, sign included, on the infinities, zeros and NaNs.
    special = ~torch.isfinite(want) | (want == 0)
    assert torch.equal(got[special].isnan(), want[special].isnan())
    same = special & ~want.isnan()
    assert torch.equal(got[same], want[same])
    assert torch.equal(got[same].signbit(), want[same].signbit())
    # The tensor ** scalar route runs the same core.
    for e in (1e308, -1e308, 1e300, -1e300, 3e20, -3e20):
        with ran("aten::pow.Tensor_Scalar"):
            got = torch.pow(x.to(mojo_gpu), e)
        _close(got, torch.pow(x, e), rtol=2.3e-16, atol=0)


def test_pow_float64_near_overflow_and_underflow(mojo_gpu):
    """y * log|x| just past exp's overflow (709.78) or underflow (-745.13)
    bounds: pow(2, 1024.003) is inf, never inf - inf = NaN from the
    double-double correction of an exp that already overflowed."""
    skip_if_metal(mojo_gpu, "Metal has no float64")
    bases = torch.tensor([2.0, math.e, 10.0, 0.5, 1.0001, 123.456], dtype=torch.float64)
    t = torch.cat(
        [
            torch.linspace(709.0, 710.5, 4001, dtype=torch.float64),
            torch.linspace(-746.0, -744.0, 4001, dtype=torch.float64),
        ]
    )
    x = bases.repeat_interleave(len(t))
    y = t.repeat(len(bases)) / torch.log(x)
    x = torch.cat([x, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)])
    y = torch.cat(
        [y, torch.tensor([1024.003, 1023.9999, -1075.1], dtype=torch.float64)]
    )
    want = torch.pow(x, y)
    with ran("aten::pow.Tensor_Tensor"):
        got = torch.pow(x.to(mojo_gpu), y.to(mojo_gpu)).cpu()
    assert not got.isnan().any()
    # Zeros and infinities are compared as masks, both ways: a result
    # flushed to 0 (or saturated to inf) where the reference is finite and
    # nonzero fails too. The reference for them is stock CUDA torch, not
    # CPU: CUDA's double pow returns 0 once y * log|x| <= -745 (__nv_exp's
    # cutoff), where the exact result still exceeds 2**-1075 down to
    # -745.1332 and CPU's std::pow rounds it up to the smallest subnormal:
    # pow(2.0, -1074.9) is 0 on CUDA and here, 5e-324 on CPU.
    if torch.cuda.is_available() and torch.version.cuda:
        edge = torch.pow(x.cuda(), y.cuda()).cpu()
    else:
        # No CUDA reference: CPU's, where the two libraries agree; in the
        # band between the cutoffs, either 0 or CPU's 5e-324.
        band = (want == 5e-324) & (y * torch.log(x) < -744.999)
        assert ((got[band] == 0) | (got[band] == 5e-324)).all()
        edge = torch.where(band, got, want)
    assert torch.equal(got == 0, edge == 0)
    assert torch.equal(got == math.inf, edge == math.inf)
    assert torch.equal(got == -math.inf, edge == -math.inf)
    finite = torch.isfinite(want) & (want != 0) & (got != 0)
    # Within an ulp; below 2**-1022 an ulp is the absolute 5e-324.
    _close(got[finite], want[finite], rtol=2.3e-16, atol=1e-323)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA reference")
def test_pow_float64_matches_cuda(mojo_gpu):
    """Against stock CUDA torch's double pow: bit for bit on C99's special
    values and the saturating exponents, within an ulp on a broad sample
    (the double-double core is not CUDA's algorithm, and rounds a near-tie
    the other way on ~0.03% of the elements)."""
    skip_if_metal(mojo_gpu, "Metal has no float64")
    big = torch.tensor(
        [1e308, -1e308, 1e300, -1e300, 5e20, -5e20, 709.0, -745.0], dtype=torch.float64
    )
    table = torch.tensor(_POW64, dtype=torch.float64)
    base, expo = zip(*[(x, y) for x in _POW64 for y in _POW64])
    x = torch.cat([torch.tensor(base, dtype=torch.float64), table.repeat(len(big))])
    y = torch.cat(
        [torch.tensor(expo, dtype=torch.float64), big.repeat_interleave(len(table))]
    )
    want = torch.pow(x.cuda(), y.cuda()).cpu()
    got = torch.pow(x.to(mojo_gpu), y.to(mojo_gpu)).cpu()
    assert torch.equal(got.isnan(), want.isnan())
    keep = ~want.isnan()
    special = keep & (~torch.isfinite(want) | (want == 0))
    assert torch.equal(got[special].view(torch.int64), want[special].view(torch.int64))
    _close(got, want, rtol=2.3e-16, atol=0)

    g = torch.Generator().manual_seed(12)
    n = 1 << 18
    x = torch.exp(torch.randn(n, dtype=torch.float64, generator=g) * 30)
    x = torch.where(torch.rand(n, generator=g) < 0.2, -x, x)
    y = torch.randn(n, dtype=torch.float64, generator=g) * 40
    y = torch.where(torch.rand(n, generator=g) < 0.3, torch.round(y), y)
    want = torch.pow(x.cuda(), y.cuda()).cpu()
    got = torch.pow(x.to(mojo_gpu), y.to(mojo_gpu)).cpu()
    assert torch.equal(got.isnan(), want.isnan())
    keep = ~want.isnan()
    ulps = (got[keep].view(torch.int64) - want[keep].view(torch.int64)).abs()
    assert int(ulps.max()) <= 1
    assert int((ulps != 0).sum()) < n // 1000


@pytest.mark.parametrize("dtype", FLOATS)
def test_lerp_tensor(mojo_gpu, dtype):
    torch.manual_seed(5)
    s = torch.randn(40, 3).to(dtype)
    e = torch.randn(40, 3).to(dtype)
    w = (torch.rand(40, 3) * 1.6 - 0.3).to(dtype)
    with ran("aten::lerp.Tensor"):
        actual = torch.lerp(s.to(mojo_gpu), e.to(mojo_gpu), w.to(mojo_gpu))
    _close(actual, torch.lerp(s, e, w), **_tol(dtype, 2))
    wb = w[:1]
    _close(
        torch.lerp(s.to(mojo_gpu), e.to(mojo_gpu), wb.to(mojo_gpu)),
        torch.lerp(s, e, wb),
        **_tol(dtype, 2),
    )
    x, x_cpu = s.to(mojo_gpu), s.clone()
    x.lerp_(e.to(mojo_gpu), w.to(mojo_gpu))
    x_cpu.lerp_(e, w)
    _close(x, x_cpu, **_tol(dtype, 2))


def test_lerp_tensor_out_and_promotion(mojo_gpu):
    torch.manual_seed(6)
    s, e, w = torch.randn(6), torch.randn(6), torch.rand(6)
    out = torch.empty(6, device=mojo_gpu)
    with ran("aten::lerp.Tensor_out"):
        torch.lerp(s.to(mojo_gpu), e.to(mojo_gpu), w.to(mojo_gpu), out=out)
    _close(out, torch.lerp(s, e, w))
    # A 0-d weight of another float dtype promotes like a number.
    w0 = torch.tensor(0.25, dtype=torch.float16)
    _close(
        torch.lerp(s.to(mojo_gpu), e.to(mojo_gpu), w0.to(mojo_gpu)),
        torch.lerp(s, e, w0),
    )
    # A dimensioned weight: `out` must have the result dtype, no cast.
    wide = torch.empty(6, dtype=torch.float64, device=mojo_gpu)
    with pytest.raises(RuntimeError):
        torch.lerp(s.to(mojo_gpu), e.to(mojo_gpu), w.to(mojo_gpu), out=wide)


# ---------------------------------------------------------------------------
# gelu_backward (functional): accurate tanh mode, half types accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("approximate", ["tanh", "none"])
@pytest.mark.parametrize("dtype", [*FLOATS, torch.float64])
def test_gelu_backward_functional(mojo_gpu, approximate, dtype):
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "Metal has no float64")
    # |x| < 1e18: past that x * x overflows float32 and CUDA's formula gives
    # NaN (inf * 0), where a float64 reference does not.
    special = [v for v in _SPECIAL if not abs(v) > 1e18]
    x = torch.cat([torch.linspace(-10, 10, 4001), torch.tensor(special)]).to(dtype)
    g = torch.randn(x.shape).to(dtype)
    # float64 reference of the same formula, rounded once.
    want = torch.ops.aten.gelu_backward(g.double(), x.double(), approximate=approximate)
    want = want.to(dtype)
    with ran("aten::gelu_backward"):
        got = torch.ops.aten.gelu_backward(
            g.to(mojo_gpu), x.to(mojo_gpu), approximate=approximate
        )
    assert got.dtype == dtype
    if dtype == torch.float32:
        # The formula cancels for x < -3 (0.5 * (1 + t) and 1 - t * t with
        # t ~ -1), which costs ~1e-6 absolute even with an exact tanhf; the
        # old tanh.approx.f32 was off by ~1e-3 there.
        tol = {"rtol": 1e-5, "atol": 4e-6}
    elif dtype == torch.float64:
        tol = {"rtol": 1e-12, "atol": 1e-13}
    else:
        tol = _tol(dtype)
    _close(got, want, **tol)
    # broadcasting grad (the pointwise route)
    g0 = torch.tensor(0.5, dtype=dtype)
    _close(
        torch.ops.aten.gelu_backward(
            g0.to(mojo_gpu), x.to(mojo_gpu), approximate=approximate
        ),
        torch.ops.aten.gelu_backward(
            g0.double(), x.double(), approximate=approximate
        ).to(dtype),
        **tol,
    )


# ---------------------------------------------------------------------------
# clamp.Tensor: tensor-valued bounds, broadcasting, mixed dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_clamp_tensor(mojo_gpu, dtype):
    torch.manual_seed(6)
    if dtype.is_floating_point:
        x = torch.cat([torch.tensor(_SPECIAL), torch.randn(40) * 3]).to(dtype)
    else:
        x = torch.randint(-20, 20, (57,), dtype=dtype)
    lo = (torch.arange(x.numel()) % 5 - 3).to(dtype)
    hi = (torch.arange(x.numel()) % 4).to(dtype)
    with ran("aten::clamp.Tensor"):
        actual = torch.clamp(x.to(mojo_gpu), lo.to(mojo_gpu), hi.to(mojo_gpu))
    _close(actual, torch.clamp(x, lo, hi))
    _close(torch.clamp(x.to(mojo_gpu), min=lo.to(mojo_gpu)), torch.clamp(x, min=lo))
    _close(torch.clamp(x.to(mojo_gpu), max=hi.to(mojo_gpu)), torch.clamp(x, max=hi))
    out = torch.empty_like(x, device=mojo_gpu)
    torch.clamp(x.to(mojo_gpu), lo.to(mojo_gpu), hi.to(mojo_gpu), out=out)
    _close(out, torch.clamp(x, lo, hi))
    if dtype.is_floating_point:
        nan_lo = lo.clone()
        nan_lo[3] = float("nan")
        _close(
            torch.clamp(x.to(mojo_gpu), nan_lo.to(mojo_gpu), hi.to(mojo_gpu)),
            torch.clamp(x, nan_lo, hi),
        )


def test_clamp_tensor_mixed_dtypes(mojo_gpu):
    x = torch.randn(5, 4)
    lo = torch.randint(-1, 1, (5, 4), dtype=torch.int32)
    hi = torch.randint(0, 2, (4,), dtype=torch.int64)
    _close(torch.clamp(x.to(mojo_gpu), lo.to(mojo_gpu)), torch.clamp(x, lo))
    _close(torch.clamp(x.to(mojo_gpu), None, hi.to(mojo_gpu)), torch.clamp(x, None, hi))
    _close(
        torch.clamp(x.to(mojo_gpu), lo.to(mojo_gpu), hi.to(mojo_gpu)),
        torch.clamp(x, lo, hi),
    )
    xb = x.bfloat16()
    _close(torch.clamp(xb.to(mojo_gpu), lo.to(mojo_gpu)), torch.clamp(xb, lo))


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize(
    "name,fn", (("igamma", torch.igamma), ("igammac", torch.igammac))
)
def test_igamma(mojo_gpu, name, fn, dtype):
    """Every regime of calc_igamma / calc_igammac: the boundaries (a or x at
    0, inf, NaN, negative), the series and the continued fraction on either
    side of x = a, and the uniform asymptotic expansion for large a ~ x."""
    grid = [0.0, 1e-3, 0.3, 0.5, 0.75, 1.0, 1.05, 1.2, 2.5, 7.0, 19.0]
    grid += [21.0, 24.0, 30.0, 150.0, 199.0, 210.0, 1000.0, 1040.0]
    edges = [float("inf"), float("nan"), -1.0]
    values = torch.tensor(grid + edges)
    a = values[:, None].expand(-1, len(values)).to(dtype)
    x = values[None, :].expand(len(values), -1).to(dtype)
    expected = fn(a, x)
    with ran(f"aten::{name}"):
        actual = fn(a.to(mojo_gpu), x.to(mojo_gpu))
    _close(actual, expected, **_tol(dtype, 8))
    # Broadcast against a 0-d tensor, a strided out= and the in-place method.
    x_row = x[3].contiguous()
    scalar = torch.tensor(2.5, dtype=dtype)
    _close(
        fn(x_row.to(mojo_gpu), scalar.to(mojo_gpu)), fn(x_row, scalar), **_tol(dtype, 8)
    )
    out = torch.zeros(len(values), 2 * len(values), dtype=dtype).to(mojo_gpu)
    fn(a.to(mojo_gpu), x.to(mojo_gpu), out=out[:, ::2])
    _close(out[:, ::2], expected, **_tol(dtype, 8))
    inplace = a.contiguous().to(mojo_gpu)
    getattr(inplace, name + "_")(x.to(mojo_gpu))
    _close(inplace, expected, **_tol(dtype, 8))


# --------------------------------------------------------------------------
# binary math: atan2, hypot, copysign, fmod (pointwise_math broadcast route)
# --------------------------------------------------------------------------


# name, torch function, overload names, ulps of float32 tolerance
_BINARY = [
    ("atan2", torch.atan2, ("aten::atan2",), 2),
    ("hypot", torch.hypot, ("aten::hypot",), 1),
    ("copysign", torch.copysign, ("aten::copysign.Tensor",), 0),
    ("fmax", torch.fmax, ("aten::fmax",), 0),
    ("fmin", torch.fmin, ("aten::fmin",), 0),
    ("fmod", torch.fmod, ("aten::fmod.Tensor",), 0),
    ("nextafter", torch.nextafter, ("aten::nextafter",), 0),
    ("logaddexp", torch.logaddexp, ("aten::logaddexp",), 2),
    ("logaddexp2", torch.logaddexp2, ("aten::logaddexp2",), 2),
    ("heaviside", torch.heaviside, ("aten::heaviside",), 0),
    ("xlogy", torch.xlogy, ("aten::xlogy.Tensor",), 2),
    ("xlog1py", torch.special.xlog1py, ("aten::special_xlog1py",), 2),
]


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("name,fn,ops,ulps", _BINARY, ids=[b[0] for b in _BINARY])
def test_binary_math_edges(mojo_gpu, name, fn, ops, ulps, dtype):
    if name == "nextafter" and not _cpu_has_nextafter(dtype):
        pytest.skip(f"CPU torch has no {dtype} nextafter to compare with")
    x, y = _pairs(dtype)
    if name in ("atan2", "xlogy", "xlog1py") and _flushes_subnormals(mojo_gpu):
        # These are Metal kernels in torch MPS too, which flush the same way.
        keep = ~(_subnormal(x) | _subnormal(y))
        x, y = x[keep], y[keep]
    expected = fn(x, y)
    if name == "fmod":
        # CPU's vectorized fmod is x - trunc(x / y) * y, NaN once the quotient
        # overflows (1e30 by 1e-30); CUDA's ::fmod, and ours, is exact, which
        # is what float64 computes for these operands.
        expected = fn(x.double(), y.double()).to(dtype)
    with ran(*ops):
        actual = fn(x.to(mojo_gpu), y.to(mojo_gpu))
    assert actual.dtype == expected.dtype
    if name == "nextafter":
        # One ulp away: any tolerance would accept the input unchanged.
        if _flushes_subnormals(mojo_gpu):
            # Apple GPUs flush subnormal operands and results to zero.
            keep = ~(_subnormal(x) | _subnormal(y) | _subnormal(expected))
            actual, expected = actual.cpu()[keep], expected[keep]
        _close(actual, expected, rtol=0.0, atol=0.0)
    else:
        _close(actual, expected, **_tol(dtype, ulps))


def _cpu_has_nextafter(dtype: torch.dtype) -> bool:
    """Whether this CPU torch has a `dtype` nextafter (half types since 2.x)."""
    try:
        torch.nextafter(torch.ones(1, dtype=dtype), torch.zeros(1, dtype=dtype))
    except RuntimeError:
        return False
    return True


def test_nextafter_half_bits(mojo_gpu):
    """No CPU half kernel: check the 16-bit patterns against the float32
    ones rounded, which is exact for one ulp steps away from zero."""
    for dtype in (torch.float16, torch.bfloat16):
        x = torch.tensor([1.0, 1.0, -2.0, 0.0, 0.0, 3.0, float("nan")], dtype=dtype)
        y = torch.tensor([2.0, 0.0, 0.0, 1.0, -1.0, 3.0, 1.0], dtype=dtype)
        got = torch.nextafter(x.to(mojo_gpu), y.to(mojo_gpu)).cpu()
        u = torch.int16
        bits = x.view(u)
        step = torch.where((y > x) ^ (x < 0), 1, -1).to(u)
        want = (bits + step).view(dtype)
        want[3] = torch.tensor(0x0001, dtype=u).view(dtype)
        want[4] = torch.tensor(-32767, dtype=u).view(dtype)
        want[5] = 3.0
        want[6] = float("nan")
        if dtype == torch.bfloat16 and _flushes_subnormals(mojo_gpu):
            # The bfloat16 smallest subnormals (from +-0) are flushed to
            # zero on Apple GPUs.
            got, want = got[[0, 1, 2, 5, 6]], want[[0, 1, 2, 5, 6]]
        _close(got, want, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_binary_math_broadcast_scalar_strided_out(mojo_gpu, dtype):
    torch.manual_seed(1)
    a_cpu = torch.randn(6, 5).to(dtype)
    b_cpu = torch.randn(5).to(dtype)
    a, b = a_cpu.to(mojo_gpu), b_cpu.to(mojo_gpu)
    for fn in (
        torch.atan2,
        torch.hypot,
        torch.copysign,
        torch.fmod,
        torch.fmax,
        torch.logaddexp,
        torch.logaddexp2,
        torch.xlogy,
        torch.special.xlog1py,
    ):
        _close(fn(a, b), fn(a_cpu, b_cpu), **_tol(dtype, 2))
        _close(fn(a.t(), a.t()), fn(a_cpu.t(), a_cpu.t()), **_tol(dtype, 2))
        col = a_cpu[:, :1]
        _close(fn(a, col.to(mojo_gpu)), fn(a_cpu, col), **_tol(dtype, 2))
        zero_d = torch.tensor(0.75, dtype=dtype)
        _close(fn(a, zero_d.to(mojo_gpu)), fn(a_cpu, zero_d), **_tol(dtype, 2))
        out = torch.empty(5, 6, dtype=dtype, device=mojo_gpu).t()
        fn(a, b, out=out)
        _close(out, fn(a_cpu, b_cpu), **_tol(dtype, 2))
    # Python scalars: copysign.Scalar, fmod.Scalar, xlogy's wrapped number
    _close(torch.copysign(a, -1.0), torch.copysign(a_cpu, -1.0))
    _close(torch.fmod(a, 0.7), torch.fmod(a_cpu, 0.7), **_tol(dtype))
    _close(torch.xlogy(a, 2.0), torch.xlogy(a_cpu, 2.0), **_tol(dtype, 2))
    _close(torch.xlogy(2.0, b.abs()), torch.xlogy(2.0, b_cpu.abs()), **_tol(dtype, 2))


@pytest.mark.parametrize(
    "fn", [torch.logaddexp, torch.logaddexp2, torch.xlogy, torch.special.xlog1py]
)
def test_log_family_out_rejections(mojo_gpu, fn):
    a = torch.rand(3, device=mojo_gpu)
    b = torch.rand(3, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="single memory location"):
        fn(a, b, out=torch.empty(1, device=mojo_gpu).expand(3))
    with pytest.raises(RuntimeError, match="can't be cast"):
        fn(a, b, out=torch.empty(3, dtype=torch.int64, device=mojo_gpu))


def test_binary_math_in_place(mojo_gpu):
    a_cpu = torch.randn(4, 7)
    b_cpu = torch.randn(4, 7)
    for name in ("atan2_", "hypot_", "copysign_", "fmod_", "xlogy_", "nextafter_"):
        x, x_cpu = a_cpu.clone().to(mojo_gpu), a_cpu.clone()
        getattr(x, name)(b_cpu.to(mojo_gpu))
        getattr(x_cpu, name)(b_cpu)
        if name == "nextafter_":
            _close(x, x_cpu, rtol=0.0, atol=0.0)
        else:
            _close(x, x_cpu, **_tol(torch.float32, 2))


def test_binary_math_int_promotion(mojo_gpu):
    i = torch.arange(-6, 6)
    j = torch.arange(1, 13)
    for fn in (torch.atan2, torch.copysign, torch.xlogy):
        expected = fn(i, j)
        actual = fn(i.to(mojo_gpu), j.to(mojo_gpu))
        assert actual.dtype == expected.dtype == torch.float32
        _close(actual, expected, **_tol(torch.float32, 2))
    for fn in (torch.fmax, torch.fmin, torch.fmod):
        _close(fn(i.to(mojo_gpu), j.to(mojo_gpu)), fn(i, j))
    f = torch.randn(12)
    _close(torch.atan2(i.to(mojo_gpu), f.to(mojo_gpu)), torch.atan2(i, f))


def test_heaviside_cpu_scalar_values(mojo_gpu):
    # A CPU 0-d `values` is a tensor of its own dtype too, not a number.
    with pytest.raises(RuntimeError, match="different dtypes"):
        torch.heaviside(
            torch.ones(3, dtype=torch.int64, device=mojo_gpu),
            torch.tensor(2.0, dtype=torch.float64),
        )
    x = torch.tensor([-1.0, 0.0, 2.0])
    half = torch.tensor(0.5)
    _close(torch.heaviside(x.to(mojo_gpu), half), torch.heaviside(x, half))


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float16])
def test_heaviside_fmax_fmin_dtypes(mojo_gpu, dtype):
    if dtype == torch.bool:
        x = torch.tensor([True, False, True, False])
        y = torch.tensor([True, True, False, False])
    else:
        x = torch.tensor([-2, 0, 0, 3, 5, -1]).to(dtype)
        y = torch.tensor([7, 1, 0, -2, 5, 4]).to(dtype)
    for fn in (torch.heaviside, torch.fmax, torch.fmin):
        _close(fn(x.to(mojo_gpu), y.to(mojo_gpu)), fn(x, y))


def test_heaviside_rejects_mixed_dtypes(mojo_gpu):
    with pytest.raises(RuntimeError, match="different dtypes"):
        torch.heaviside(
            torch.ones(3, device=mojo_gpu),
            torch.ones(3, dtype=torch.int64, device=mojo_gpu),
        )


@pytest.mark.parametrize("dtype", INTS)
def test_gcd_lcm_shifts_fmod_int(mojo_gpu, dtype):
    torch.manual_seed(2)
    signed = dtype != torch.uint8
    lo = -60 if signed else 0
    a = torch.randint(lo, 60, (200,), dtype=dtype)
    b = torch.randint(lo, 60, (200,), dtype=dtype)
    a[:4] = torch.tensor([0, 0, 12, 7], dtype=dtype)
    b[:4] = torch.tensor([0, 5, 0, 7], dtype=dtype)
    with ran("aten::gcd"):
        _close(torch.gcd(a.to(mojo_gpu), b.to(mojo_gpu)), torch.gcd(a, b))
    _close(torch.lcm(a.to(mojo_gpu), b.to(mojo_gpu)), torch.lcm(a, b))
    nonzero = torch.where(b == 0, torch.ones_like(b), b)
    _close(torch.fmod(a.to(mojo_gpu), nonzero.to(mojo_gpu)), torch.fmod(a, nonzero))
    bits = torch.iinfo(dtype).bits
    shift = torch.randint(0, bits + 4, (200,)).to(dtype)
    if signed:
        shift[:3] = torch.tensor([-1, bits, bits - 1], dtype=dtype)
    with ran("aten::bitwise_left_shift.Tensor", "aten::__lshift__.Tensor"):
        _close(a.to(mojo_gpu) << shift.to(mojo_gpu), a << shift)
    _close(a.to(mojo_gpu) >> shift.to(mojo_gpu), a >> shift)
    _close(
        torch.bitwise_left_shift(a.to(mojo_gpu), shift.to(mojo_gpu)),
        torch.bitwise_left_shift(a, shift),
    )
    _close(a.to(mojo_gpu) << 3, a << 3)
    _close(a.to(mojo_gpu) >> 2, a >> 2)
    x, x_cpu = a.to(mojo_gpu), a.clone()
    x <<= 1
    x_cpu <<= 1
    x >>= shift.to(mojo_gpu)
    x_cpu >>= shift
    _close(x, x_cpu)


@pytest.mark.parametrize("dtype", FLOATS)
def test_frexp(mojo_gpu, dtype):
    x = torch.tensor(_SPECIAL + [3.0, 0.25, -1000.0, 1e-38, 6e-39]).to(dtype)
    x = torch.cat([x, torch.randn(100).to(dtype) * 100])
    m_cpu, e_cpu = torch.frexp(x)
    with ran("aten::frexp.Tensor", "aten::frexp.Tensor_out"):
        m, e = torch.frexp(x.to(mojo_gpu))
    finite = torch.isfinite(x)
    _close(m.cpu()[finite], m_cpu[finite], rtol=0.0, atol=0.0)
    _close(e.cpu()[finite], e_cpu[finite])
    assert e.dtype == torch.int32
    mo = torch.empty(0, dtype=dtype, device=mojo_gpu)
    eo = torch.empty(0, dtype=torch.int32, device=mojo_gpu)
    torch.frexp(x.to(mojo_gpu), out=(mo, eo))
    _close(mo.cpu()[finite], m_cpu[finite], rtol=0.0, atol=0.0)


def test_zeta(mojo_gpu):
    x = torch.tensor([1.0, 0.5, 2.0, 3.5, 2.0, 4.0, 2.0, 1.5, 10.0])
    q = torch.tensor([1.0, 1.0, 1.0, 2.0, -1.0, -2.5, 0.25, 30.0, 0.5])
    torch.manual_seed(3)
    x = torch.cat([x, torch.rand(100) * 6 + 1.01])
    q = torch.cat([q, torch.rand(100) * 5 + 0.1])
    with ran("aten::special_zeta"):
        actual = torch.special.zeta(x.to(mojo_gpu), q.to(mojo_gpu))
    _close(actual, torch.special.zeta(x, q), rtol=4e-6, atol=1e-6)
    _close(
        torch.special.zeta(x.to(mojo_gpu), 2.0),
        torch.special.zeta(x, 2.0),
        rtol=4e-6,
        atol=1e-6,
    )


_POLYS = [
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


@pytest.mark.parametrize("name", _POLYS)
def test_special_polynomials(mojo_gpu, name):
    fn = getattr(torch.special, name)
    torch.manual_seed(4)
    x = torch.cat(
        [torch.tensor([-1.0, 1.0, 0.0, 0.5, 2.0]), torch.rand(120) * 2.2 - 1.1]
    )
    n = torch.randint(-1, 14, (x.numel(),)).float()
    n[:5] = torch.tensor([3.0, 4.0, 5.0, 9.0, 3.0])
    with ran(f"aten::special_{name}"):
        actual = fn(x.to(mojo_gpu), n.to(mojo_gpu))
    # n > 6-8 takes cos(n acos x): a few float32 ulps of the argument.
    _close(actual, fn(x, n), rtol=2e-5, atol=2e-5)
    _close(fn(x.to(mojo_gpu), 3), fn(x, 3), rtol=2e-5, atol=2e-5)
    # NaN x: CPU's loop reads garbage there; CUDA (and ours) gives the
    # degree-0 constant, 0 for a negative degree, NaN otherwise.
    nan_x = torch.full((4,), float("nan"), device=mojo_gpu)
    got = fn(nan_x, torch.tensor([0.0, -1.0, 2.0, 7.0], device=mojo_gpu)).cpu()
    _close(
        got, torch.tensor([1.0, 0.0, float("nan"), float("nan")]), rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize(
    "name",
    [
        "hermite_polynomial_h",
        "hermite_polynomial_he",
        "laguerre_polynomial_l",
        "legendre_polynomial_p",
    ],
)
def test_special_polynomials_out(mojo_gpu, name):
    """The out= overloads: resize, strided out, a cast into a float16 out,
    overlap refused."""
    fn = getattr(torch.special, name)
    torch.manual_seed(6)
    x_cpu, n_cpu = torch.rand(3, 4) * 2 - 1, torch.randint(0, 6, (3, 4)).float()
    x, n = x_cpu.to(mojo_gpu), n_cpu.to(mojo_gpu)
    want = fn(x_cpu, n_cpu)
    out = torch.empty(0, device=mojo_gpu)
    with ran(f"aten::special_{name}.out"):
        fn(x, n, out=out)
    _close(out, want, rtol=2e-5, atol=2e-5)
    strided = torch.zeros(4, 3, device=mojo_gpu).t()
    fn(x, n, out=strided)
    _close(strided, want, rtol=2e-5, atol=2e-5)
    half = torch.empty(3, 4, dtype=torch.float16, device=mojo_gpu)
    fn(x, n, out=half)
    _close(half, want.half(), **_tol(torch.float16, 2))
    with pytest.raises(RuntimeError, match="single memory location"):
        fn(x, n, out=torch.empty(1, device=mojo_gpu).expand(3, 4))


@pytest.mark.parametrize("name", ["hermite_polynomial_h", "legendre_polynomial_p"])
def test_special_polynomials_cpu_scalar_degree(mojo_gpu, name):
    """An explicit CPU 0-d float64 degree promotes as a 0-dim tensor (an
    int64 x gives float64), not as a wrapped Python number."""
    skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    fn = getattr(torch.special, name)
    i = torch.tensor([0, 1, 2])
    deg = torch.tensor(2.0, dtype=torch.float64)
    got = fn(i.to(mojo_gpu), deg)
    assert got.dtype == torch.float64
    _close(got, fn(i, deg))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("name", "x"),
    [
        ("hermite_polynomial_h", 0.5),
        ("hermite_polynomial_he", 0.5),
        ("laguerre_polynomial_l", 0.0),
        ("legendre_polynomial_p", 1.0),
        ("legendre_polynomial_p", -1.0),
    ],
)
def test_special_polynomials_huge_degree(mojo_gpu, name, x, dtype):
    """Degrees near int64's edge: 9.21e18 is below 2^63, so it converts
    (1 for legendre at +-1 and laguerre at 0, NaN past hermite's limit), and
    the negatives give 0. The x values return before any recurrence loop.
    A degree of 2^63 or more is UB in C++ (`static_cast<int64_t>`): x86
    CPU torch gives INT64_MIN (0 here) but ARM and CUDA saturate, so it is
    not compared."""
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    fn = getattr(torch.special, name)
    n = torch.tensor([9.21e18, 2.0**62, -9.21e18, -(2.0**63)], dtype=dtype)
    xs = torch.full_like(n, x)
    got = fn(xs.to(mojo_gpu), n.to(mojo_gpu))
    _close(got, fn(xs, n), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("name", _POLYS)
def test_special_polynomials_float64(mojo_gpu, name):
    """float64 (CUDA dispatches it): the cos / acos branch has no std.math
    lowering on NVIDIA or AMD, so it takes fdlibm's double kernels."""
    skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    fn = getattr(torch.special, name)
    torch.manual_seed(4)
    x = torch.cat(
        [torch.tensor([-1.0, 1.0, 0.0, 0.5, 2.0]), torch.rand(120) * 2.2 - 1.1]
    ).double()
    n = torch.randint(-1, 30, (x.numel(),)).double()
    with ran(f"aten::special_{name}"):
        actual = fn(x.to(mojo_gpu), n.to(mojo_gpu))
    _close(actual, fn(x, n), rtol=1e-12, atol=1e-12)


def test_float64_binary_math(mojo_gpu):
    """atan2, hypot, logaddexp2 and zeta on float64 against CPU torch, edges
    included: C99's signed zeros and infinities, 2^-1070 through logaddexp2's
    exp2, zeta's NaN comparisons (zeta(1, NaN) = inf, zeta(NaN, 0) = inf)."""
    skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    x, y = _pairs(torch.float64)
    for fn in (torch.atan2, torch.hypot):
        _close(fn(x.to(mojo_gpu), y.to(mojo_gpu)), fn(x, y), rtol=4.5e-16, atol=0.0)
    a = torch.tensor([0.0, -1070.0, 3.0, float("inf"), -float("inf"), 5.5])
    b = torch.tensor([-1070.0, 0.0, 3.0, float("inf"), -float("inf"), -2.25])
    a, b = a.double(), b.double()
    _close(
        torch.logaddexp2(a.to(mojo_gpu), b.to(mojo_gpu)),
        torch.logaddexp2(a, b),
        rtol=4.5e-16,
        atol=0.0,
    )
    zx = torch.tensor([1.0, float("nan"), float("nan"), float("inf"), 2.0, 3.0])
    zq = torch.tensor([float("nan"), 0.0, -1.0, 1.0, 1.0, -2.5])
    zx, zq = zx.double(), zq.double()
    _close(
        torch.special.zeta(zx.to(mojo_gpu), zq.to(mojo_gpu)),
        torch.special.zeta(zx, zq),
        rtol=1e-15,
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# deg2rad / rad2deg / ldexp: ATen composes them (empty + mul, pow + mul); one
# pointwise kernel each here.
# ---------------------------------------------------------------------------


def _only(op: str):
    """Exactly one native op, `op`, ran inside the block."""

    @contextlib.contextmanager
    def check():
        native.op_counting(True)
        native.op_counts_reset()
        yield
        assert native.op_counts() == {op: 1}, native.op_counts()

    return check()


@pytest.mark.parametrize("name", ["deg2rad", "rad2deg"])
@pytest.mark.parametrize(
    "dtype", [*FLOATS, torch.float64, torch.int64, torch.int32, torch.bool]
)
def test_deg2rad_rad2deg(mojo_gpu, name, dtype):
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    fn = getattr(torch, name)
    if dtype.is_floating_point:
        x = torch.cat([torch.tensor(_SPECIAL), torch.randn(200) * 400]).to(dtype)
    elif dtype == torch.bool:
        x = torch.tensor([True, False, True])
    else:
        x = torch.arange(-360, 360, 7).to(dtype)
    xd = x.to(mojo_gpu)
    with _only(f"aten::{name}"):
        got = fn(xd)
    want = fn(x)
    assert got.dtype == want.dtype
    _close(
        got, want, **_tol(want.dtype if want.dtype != torch.float64 else torch.float32)
    )
    # strided, out= and in-place
    m = x[: (x.numel() // 2) * 2].reshape(2, -1)
    _close(fn(m.to(mojo_gpu).t()), fn(m.t()), **_tol(want.dtype))
    if dtype.is_floating_point:
        out = torch.empty(0, dtype=dtype, device=mojo_gpu)
        fn(x.to(mojo_gpu), out=out)
        _close(out, want, **_tol(want.dtype))
        y = x.clone().to(mojo_gpu)
        getattr(y, name + "_")()
        _close(y, want, **_tol(want.dtype))


_LDEXP_X = [
    0.0,
    -0.0,
    1.0,
    -1.5,
    3.25,
    1e-30,
    -7e30,
    1e-40,
    float("inf"),
    float("-inf"),
    float("nan"),
]
_LDEXP_E = [
    0,
    1,
    -1,
    5,
    -5,
    60,
    -60,
    127,
    -126,
    -149,
    -150,
    128,
    200,
    -200,
    1000,
    -1000,
    100000,
    -100000,
]


@pytest.mark.parametrize("dtype", [*FLOATS, torch.float64])
@pytest.mark.parametrize("edtype", [torch.int64, torch.int32, torch.int8])
def test_ldexp_int_exponent(mojo_gpu, dtype, edtype):
    """The `_ldexp_int_exponent` route: ::ldexp(x, exp), exact (including
    subnormal results, rounded once) and saturating."""
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    info = torch.iinfo(edtype)
    xs, es = zip(
        *[(x, e) for x in _LDEXP_X for e in _LDEXP_E if info.min <= e <= info.max]
    )
    x = torch.tensor(xs, dtype=torch.float64).to(dtype)
    e = torch.tensor(es, dtype=edtype)
    # CPU's ldexp kernel is std::ldexp(double(x), exp) rounded once: exactly
    # what ::ldexp gives.
    want = torch.ldexp(x, e)
    if _flushes_subnormals(mojo_gpu):
        # A subnormal operand or result is flushed on Apple GPUs (torch MPS,
        # which multiplies by pow(2, e) instead, flushes it too).
        keep = ~(_subnormal(x) | _subnormal(want))
        x, e, want = x[keep], e[keep], want[keep]
    xd, ed = x.to(mojo_gpu), e.to(mojo_gpu)
    with _only("aten::ldexp.Tensor"):
        got = torch.ldexp(xd, ed)
    assert got.dtype == dtype
    _close(got, want, rtol=0.0, atol=0.0)
    # broadcasting, out= and in-place
    x2 = torch.randn(4, 1).to(dtype)
    e2 = torch.tensor([-3, 0, 7], dtype=edtype)
    want2 = torch.ldexp(x2, e2)
    _close(torch.ldexp(x2.to(mojo_gpu), e2.to(mojo_gpu)), want2, rtol=0.0, atol=0.0)
    out = torch.empty(4, 3, dtype=dtype, device=mojo_gpu)
    torch.ldexp(x2.to(mojo_gpu), e2.to(mojo_gpu), out=out)
    _close(out, want2, rtol=0.0, atol=0.0)
    y = x2.expand(4, 3).clone().to(mojo_gpu)
    y.ldexp_(e2.to(mojo_gpu))
    _close(y, want2, rtol=0.0, atol=0.0)


def test_ldexp_general_route(mojo_gpu):
    """Float exponents and integral self: self * pow(2, other), promoted."""
    torch.manual_seed(3)
    x = torch.randn(50)
    e = (torch.randn(50) * 10).round()
    for xs, es in [
        (x, e),
        (x, e.half()),
        (x.half(), e.half()),
        (x.bfloat16(), e),
        (x.half(), e.bfloat16()),
        (torch.arange(-5, 5), torch.arange(10)),
        (torch.arange(-5, 5), torch.randn(10)),
        (x, e * 0.3),
    ]:
        want = torch.ldexp(xs, es)
        got = torch.ldexp(xs.to(mojo_gpu), es.to(mojo_gpu))
        assert got.dtype == want.dtype, (xs.dtype, es.dtype)
        _close(got, want, **_tol(want.dtype, 2))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("edtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("where", ["mojo", "cpu"])
def test_ldexp_zero_dim_float_exponent(mojo_gpu, dtype, edtype, where):
    """A 0-d exponent wider than a half self: pow(2, e) runs in e's dtype
    (15.999 is not read back as float16 16.0, which overflowed to inf) and
    is rounded to the half dtype before the product, as TensorIterator's
    common-dtype cast does. An explicit CPU 0-d exponent is accepted too."""
    if where == "mojo" and edtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    x = torch.tensor([1.0, -0.5, 3.0, 1e-3]).to(dtype)
    e = torch.tensor(15.999, dtype=edtype)
    got = torch.ldexp(x.to(mojo_gpu), e.to(mojo_gpu) if where == "mojo" else e)
    _close(got, torch.ldexp(x, e), rtol=0.0, atol=0.0)


def test_ldexp_pow_rounds_to_mul_dtype(mojo_gpu):
    """_pow2 runs pow in its own dtype and mul casts that result to its
    rank-aware common dtype: a 0-d float64 2^127.999999 stays finite in
    float32 (the exponent is not read back as float32 128.0), a 0-d float64
    2^128 is inf against a float32 self, in place too, and a float16
    exponent of 16 is inf even against a float64 self."""
    skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    one = torch.tensor([1.0, -1.0])
    near = torch.tensor(127.999999, dtype=torch.float64)
    _close(
        torch.ldexp(one.to(mojo_gpu), near.to(mojo_gpu)),
        torch.ldexp(one, near),
        rtol=0.0,
        atol=0.0,
    )
    x = torch.tensor([0.5, 0.0])
    e = torch.tensor(128.0, dtype=torch.float64)
    want = torch.ldexp(x, e)
    _close(torch.ldexp(x.to(mojo_gpu), e.to(mojo_gpu)), want, rtol=0.0, atol=0.0)
    inplace = x.to(mojo_gpu)
    inplace.ldexp_(e.to(mojo_gpu))
    _close(inplace, x.clone().ldexp_(e), rtol=0.0, atol=0.0)
    d = torch.tensor([0.5, 0.0], dtype=torch.float64)
    h = torch.tensor([16.0, 16.0], dtype=torch.float16)
    want = torch.ldexp(d, h)
    got = torch.ldexp(d.to(mojo_gpu), h.to(mojo_gpu))
    assert got.dtype == want.dtype
    _close(got, want, rtol=0.0, atol=0.0)


def test_ldexp_cpu_float64_scalar_exponent(mojo_gpu):
    """A CPU 0-d float64 exponent: pow(2, e) in double, rounded to float32
    only for the product (Apple GPUs run that pow on the host, like MPS)."""
    for x_cpu, e in [
        (torch.tensor([1.0, -1.0]), 127.999999),
        (torch.tensor([0.5, 0.0]), 128.0),
        (torch.tensor([3.0, 0.25]), -1.5),
        (torch.tensor([3.0, 0.0]), math.inf),
        (torch.tensor([3.0, -0.25]), -math.inf),
        (torch.tensor([3.0, 0.25]), math.nan),
        (torch.tensor([1.0, -1.0]), 2.0**32),
        (torch.tensor([1.0, -1.0]), -(2.0**32)),
        (torch.tensor([1.0, -1.0]), 1100.5),
    ]:
        e_cpu = torch.tensor(e, dtype=torch.float64)
        _close(
            torch.ldexp(x_cpu.to(mojo_gpu), e_cpu),
            torch.ldexp(x_cpu, e_cpu),
            rtol=0.0,
            atol=0.0,
        )
    h = torch.tensor([1.0, -0.5]).half()
    e_cpu = torch.tensor(15.999, dtype=torch.float64)
    _close(
        torch.ldexp(h.to(mojo_gpu), e_cpu), torch.ldexp(h, e_cpu), rtol=0.0, atol=0.0
    )


def test_ldexp_large_float_exponent_tensor(mojo_gpu):
    """Huge finite float exponents are inf / 0, never wrapped like ints."""
    x = torch.tensor([1.0, 1.0, -2.0, 0.5])
    e = torch.tensor([2.0**32, -(2.0**32), 1e10, 1030.0])
    _close(
        torch.ldexp(x.to(mojo_gpu), e.to(mojo_gpu)),
        torch.ldexp(x, e),
        rtol=0.0,
        atol=0.0,
    )


def test_ldexp_int64_exponent_wraps_to_int(mojo_gpu):
    """An int64 exponent is converted to ::ldexp's int first (2**32 is 0),
    for a device tensor, a CPU 0-d tensor and in place."""
    x = torch.tensor([1.0, 0.5, -3.0])
    e = torch.tensor([2**32, -(2**32), 2**32 + 3])
    _close(
        torch.ldexp(x.to(mojo_gpu), e.to(mojo_gpu)),
        torch.ldexp(x, e),
        rtol=0.0,
        atol=0.0,
    )
    for v in (2**32, 2**32 + 3, -(2**33) - 1):
        e0 = torch.tensor(v)
        _close(torch.ldexp(x.to(mojo_gpu), e0), torch.ldexp(x, e0), rtol=0.0, atol=0.0)
    y = x.to(mojo_gpu)
    y.ldexp_(e.to(mojo_gpu))
    _close(y, x.clone().ldexp_(e), rtol=0.0, atol=0.0)


def test_ldexp_inplace_cpu_scalar_exponent(mojo_gpu):
    """ldexp_ takes an explicit CPU 0-d exponent, float or integral."""
    for e_cpu in (
        torch.tensor(3.5, dtype=torch.float64),
        torch.tensor(3),
        torch.tensor(-2.25),
    ):
        x_cpu = torch.tensor([1.0, 0.5, -3.0])
        x = x_cpu.to(mojo_gpu)
        x.ldexp_(e_cpu)
        _close(x, x_cpu.clone().ldexp_(e_cpu), rtol=0.0, atol=0.0)


def test_frexp_out_partially_overlapping_input(mojo_gpu):
    """An output that partially overlaps the input raises, as on CPU; the
    same view is fine."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=mojo_gpu)
    e = torch.empty(3, dtype=torch.int32, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="single memory location"):
        torch.frexp(x[:-1], out=(x[1:], e))
    x_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
    want = torch.frexp(x_cpu)
    torch.frexp(x, out=(x, torch.empty(4, dtype=torch.int32, device=mojo_gpu)))
    _close(x, want.mantissa)


def test_frexp_rejects_overlapping_out(mojo_gpu):
    x = torch.rand(3, device=mojo_gpu)
    exponent = torch.empty(3, dtype=torch.int32, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="single memory location"):
        torch.frexp(x, out=(torch.empty(1, device=mojo_gpu).expand(3), exponent))


def test_ldexp_autograd(mojo_gpu):
    x_cpu = torch.randn(6, requires_grad=True)
    e_cpu = torch.randn(6, requires_grad=True)
    x = x_cpu.detach().to(mojo_gpu).requires_grad_()
    e = e_cpu.detach().to(mojo_gpu).requires_grad_()
    torch.ldexp(x, e).sum().backward()
    torch.ldexp(x_cpu, e_cpu).sum().backward()
    _close(x.grad, x_cpu.grad, **_tol(torch.float32, 4))
    _close(e.grad, e_cpu.grad, **_tol(torch.float32, 4))


# --------------------------------------------------------------------------
# shrink / mish backward promotion, softshrink's lambd range, the forward
# activations' out= overloads
# --------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["hardshrink_backward", "softshrink_backward"])
def test_shrink_backward_promotes(mojo_gpu, op):
    fn = getattr(torch.ops.aten, op)
    g = torch.tensor([1.0, 1.0, 1.0])
    for x in (
        torch.tensor([0.3, 0.1, -0.3]).half(),
        torch.tensor([0.7, 0.1, -0.6]).double(),
    ):
        if x.dtype == torch.float64:
            skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
        want = fn(g, x, 0.3)
        got = fn(g.to(mojo_gpu), x.to(mojo_gpu), 0.3)
        assert got.dtype == want.dtype
        _close(got, want, rtol=0.0, atol=0.0)


def test_mish_backward_keeps_self_dtype(mojo_gpu):
    g = torch.tensor([1.0, -0.5, 2.0])
    x = torch.tensor([0.3, -1.0, 4.0]).half()
    want = torch.ops.aten.mish_backward(g, x)
    got = torch.ops.aten.mish_backward(g.to(mojo_gpu), x.to(mojo_gpu))
    assert got.dtype == want.dtype == torch.float16
    _close(got, want, **_tol(torch.float16))


@pytest.mark.parametrize("lambd", [float("nan"), float("inf"), -0.5, 1e6])
def test_softshrink_lambd_range(mojo_gpu, lambd):
    x = torch.tensor([1.0, -2.0]).half().to(mojo_gpu)
    with pytest.raises(RuntimeError, match="lambda must be in range"):
        F.softshrink(x, lambd)
    if lambd == 1e6:  # in range for float32
        _close(F.softshrink(x.float(), lambd), F.softshrink(x.cpu().float(), lambd))


@pytest.mark.parametrize(
    "name", ["hardshrink", "softshrink", "hardsigmoid", "hardswish", "mish"]
)
def test_activation_b_out(mojo_gpu, name):
    op = getattr(torch.ops.aten, name).out
    x_cpu = torch.randn(4, 5) * 3
    x = x_cpu.to(mojo_gpu)
    want = getattr(torch.ops.aten, name)(x_cpu)
    out = torch.empty(0, device=mojo_gpu)
    with ran(f"aten::{name}.out"):
        assert op(x, out=out) is out
    _close(out, want, **_tol(torch.float32, 2))
    strided = torch.zeros(5, 4, device=mojo_gpu).t()
    op(x, out=strided)
    _close(strided, want, **_tol(torch.float32, 2))
    with pytest.raises(RuntimeError):
        op(x, out=torch.empty(4, 5, dtype=torch.float16, device=mojo_gpu))


@pytest.mark.parametrize("dtype", FLOATS)
def test_pow_scalar_base(mojo_gpu, dtype):
    e = torch.cat([torch.tensor(_SPECIAL), torch.randn(50) * 3]).to(dtype)
    for base in (2.0, 0.5, 1.0, 10.0, 0.0, -2.0):
        with ran("aten::pow.Scalar"):
            actual = torch.pow(base, e.to(mojo_gpu))
        want = torch.pow(base, e)
        if _flushes_subnormals(mojo_gpu):
            # Apple GPUs flush subnormal operands and results (torch MPS's
            # Metal pow as well).
            keep = ~(_subnormal(e) | _subnormal(want))
            actual, want = actual.cpu()[keep], want[keep]
        _close(actual, want, **_tol(dtype, 2))
    out = torch.empty_like(e, device=mojo_gpu)
    torch.pow(3.0, e.to(mojo_gpu), out=out)
    _close(out, torch.pow(3.0, e), **_tol(dtype, 2))


# --------------------------------------------------------------------------
# activations
# --------------------------------------------------------------------------


def _in(x: torch.Tensor, value: float) -> float:
    """`value` rounded to x's dtype: CUDA's shrink/threshold kernels compare
    against `value.to<scalar_t>()`, CPU's reduced-float ones against the
    float value; passing the rounded value makes both agree."""
    return torch.tensor(value, dtype=x.dtype).item()


_ACT = [
    ("elu", lambda x: F.elu(x), "aten::elu"),
    ("elu_params", lambda x: torch.ops.aten.elu(x, 0.7, 1.3, 0.8), "aten::elu"),
    ("selu", lambda x: F.selu(x), "aten::elu"),
    ("celu", lambda x: F.celu(x, 1.5), "aten::elu"),
    ("hardtanh", lambda x: F.hardtanh(x, -0.4, 1.7), "aten::hardtanh"),
    ("relu6", lambda x: F.relu6(x), "aten::hardtanh"),
    ("leaky_relu", lambda x: F.leaky_relu(x, 0.2), "aten::leaky_relu"),
    ("softplus", lambda x: F.softplus(x), "aten::softplus"),
    ("softplus_params", lambda x: F.softplus(x, 2.0, 3.0), "aten::softplus"),
    ("hardshrink", lambda x: F.hardshrink(x, _in(x, 0.3)), "aten::hardshrink"),
    ("softshrink", lambda x: F.softshrink(x, _in(x, 0.3)), "aten::softshrink"),
    ("hardsigmoid", lambda x: F.hardsigmoid(x), "aten::hardsigmoid"),
    ("hardswish", lambda x: F.hardswish(x), "aten::hardswish"),
    ("mish", lambda x: F.mish(x), "aten::mish"),
    ("threshold", lambda x: F.threshold(x, _in(x, 0.3), -2.0), "aten::threshold"),
    ("logsigmoid", lambda x: F.logsigmoid(x), "aten::log_sigmoid_forward"),
    (
        "rrelu_eval",
        lambda x: F.rrelu(x, 0.1, 0.3, training=False),
        "aten::rrelu_with_noise",
    ),
]


def _act_input(dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(7)
    edges = torch.tensor(
        [
            0.0,
            -0.0,
            3.0,
            -3.0,
            0.3,
            -0.3,
            20.0,
            25.0,
            -20.0,
            88.0,
            -88.0,
            1e-30,
            float("inf"),
            float("-inf"),
            float("nan"),
        ]
    )
    return torch.cat([edges, torch.randn(200) * 4]).to(dtype)


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("name,fn,op", _ACT, ids=[a[0] for a in _ACT])
def test_activation_forward(mojo_gpu, name, fn, op, dtype):
    x = _act_input(dtype)
    with ran(op):
        actual = fn(x.to(mojo_gpu))
    _close(actual, fn(x), **_tol(dtype, 3))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("name,fn,op", _ACT, ids=[a[0] for a in _ACT])
def test_activation_autograd(mojo_gpu, name, fn, op, dtype):
    """forward + backward against CPU: the backward kernels (elu_backward,
    hardtanh_backward, ...) run natively."""
    x_cpu = _act_input(dtype)
    x_cpu = x_cpu[torch.isfinite(x_cpu)].clone().requires_grad_()
    x = x_cpu.detach().to(mojo_gpu).requires_grad_()
    g = torch.randn(x_cpu.shape).to(dtype)
    y_cpu = fn(x_cpu)
    y_cpu.backward(g)
    y = fn(x)
    y.backward(g.to(mojo_gpu))
    _close(y.detach(), y_cpu.detach(), **_tol(dtype, 3))
    _close(x.grad, x_cpu.grad, **_tol(dtype, 3))


def test_activation_single_native_ops(mojo_gpu):
    """relu6 -> hardtanh, selu / celu -> elu: one native op each, forward
    and backward."""
    x = torch.randn(33, device=mojo_gpu, requires_grad=True)
    cases = [
        (F.relu6, "aten::hardtanh", "aten::hardtanh_backward"),
        (F.selu, "aten::elu", "aten::elu_backward"),
        (lambda t: F.celu(t, 0.5), "aten::elu", "aten::elu_backward"),
    ]
    for fn, fwd, bwd in cases:
        native.op_counting(True)
        f0, b0 = native.op_count(fwd), native.op_count(bwd)
        y = fn(x)
        assert native.op_count(fwd) == f0 + 1
        y.sum().backward()
        assert native.op_count(bwd) == b0 + 1
        x.grad = None


def test_activation_in_place_and_out(mojo_gpu):
    x_cpu = _act_input(torch.float32)
    for fn in (
        lambda t: F.elu(t, inplace=True),
        lambda t: F.hardtanh(t, inplace=True),
        lambda t: F.leaky_relu(t, 0.1, inplace=True),
        lambda t: F.hardsigmoid(t, inplace=True),
        lambda t: F.hardswish(t, inplace=True),
        lambda t: F.threshold(t, 0.5, 1.0, inplace=True),
        lambda t: F.mish(t, inplace=True),
        lambda t: F.rrelu(t, inplace=True),
    ):
        x, xc = x_cpu.clone().to(mojo_gpu), x_cpu.clone()
        fn(x)
        fn(xc)
        _close(x, xc, **_tol(torch.float32, 3))
    out = torch.empty(0, device=mojo_gpu)
    torch.ops.aten.softplus.out(x_cpu.to(mojo_gpu), 1.0, 20.0, out=out)
    _close(out, F.softplus(x_cpu), **_tol(torch.float32, 3))


@pytest.mark.parametrize(
    "bounds", [(math.nan, 1.0), (-1.0, math.nan), (math.nan, math.nan)]
)
def test_hardtanh_nan_bound(mojo_gpu, bounds):
    """hardtanh is two-bound clamp: a NaN bound fills NaN, as CPU torch."""
    x = torch.tensor([0.0, 2.0, -2.0, math.nan])
    want = F.hardtanh(x, *bounds)
    _close(F.hardtanh(x.to(mojo_gpu), *bounds), want)
    inplace = x.to(mojo_gpu)
    F.hardtanh(inplace, *bounds, inplace=True)
    _close(inplace, want)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_leaky_relu_slope_in_opmath(mojo_gpu, dtype):
    # Stock MPS holds the slope in the input dtype (ActivationKernel.metal:
    # REGISTER_UNARY_ALPHA_OP(leaky_relu, half, half, half)), as Apple GPUs do.
    skip_if_metal(mojo_gpu, "MPS keeps leaky_relu's slope in the input dtype")
    for x, slope in [([-1e-4, -2.0, 3.0], 1e5), ([-1e4, -3e4, 3.0], 1e-8)]:
        x_cpu = torch.tensor(x).to(dtype)
        _close(F.leaky_relu(x_cpu.to(mojo_gpu), slope), F.leaky_relu(x_cpu, slope))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_activation_scalar_overflow_raises(mojo_gpu, dtype):
    """A Scalar past float32's range raises as `Scalar::to<opmath_t>()`
    does on CPU torch; float64 takes it."""
    x = torch.tensor([-1.0, 2.0]).to(dtype)
    for fn in (
        lambda a: torch.ops.aten.elu(a, 1e40, 1.0, 1.0),
        lambda a: torch.ops.aten.elu(a, 1.0, 1.0, 1e40),
        lambda a: F.leaky_relu(a, 1e40),
        lambda a: F.leaky_relu(a.clone(), 1e40, inplace=True),
        lambda a: torch.ops.aten.softplus(a, 1e40, 20.0),
        lambda a: torch.ops.aten.softplus(a, 1.0, 1e40),
        lambda a: torch.ops.aten.softplus_backward(a, a, 1e40, 20.0),
        lambda a: torch.ops.aten.elu_backward(a, 1e40, 1.0, 1.0, False, a),
        lambda a: torch.ops.aten.leaky_relu_backward(a, a, 1e40, False),
    ):
        with pytest.raises(RuntimeError, match="without overflow"):
            fn(x)
        with pytest.raises(RuntimeError, match="without overflow"):
            fn(x.to(mojo_gpu))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_elu_scalars_in_opmath(mojo_gpu, dtype):
    if dtype == torch.float16:
        # Stock MPS reads elu's scalars as half (ActivationKernel.metal:
        # REGISTER_UNARY_ALPHA_OP(elu, T, ELUParams_##T, T)), as Apple GPUs do.
        skip_if_metal(mojo_gpu, "MPS keeps elu's scalars in the input dtype")
    x = torch.tensor([-1.0, -0.5, 2.0]).to(dtype)
    want = torch.ops.aten.elu(x, 1e-46, 1e38, -10.0)
    _close(torch.ops.aten.elu(x.to(mojo_gpu), 1e-46, 1e38, -10.0), want)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_elu_intermediates(mojo_gpu, dtype):
    """elu's float opmath for the half dtypes and float64's large expm1."""
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
        x = torch.tensor([-705.0, -700.0, -1.0], dtype=dtype)
        args = (1.0, 1.0, -1.0)
    else:
        x = torch.tensor([-1e-4, -3e-4, -2.0]).to(dtype)
        args = (1e4, 1.0, 1e-4)
    want = torch.ops.aten.elu(x, *args)
    _close(torch.ops.aten.elu(x.to(mojo_gpu), *args), want)


def _gelu_grad_input(
    g: torch.Tensor, x: torch.Tensor, approximate: str
) -> torch.Tensor:
    """gelu_backward.grad_input: the pointwise route (the functional
    gelu_backward is the dedicated activation_backward kernel)."""
    out = torch.empty_like(x)
    return torch.ops.aten.gelu_backward.grad_input(
        g, x, approximate=approximate, grad_input=out
    )


_BACKWARD = [
    ("silu_backward", lambda g, x: torch.ops.aten.silu_backward(g, x)),
    ("mish_backward", lambda g, x: torch.ops.aten.mish_backward(g, x)),
    ("hardswish_backward", lambda g, x: torch.ops.aten.hardswish_backward(g, x)),
    ("hardsigmoid_backward", lambda g, x: torch.ops.aten.hardsigmoid_backward(g, x)),
    ("logit_backward", lambda g, x: torch.ops.aten.logit_backward(g, x.sigmoid())),
    (
        "logit_backward_eps",
        lambda g, x: torch.ops.aten.logit_backward(g, x.sigmoid(), 0.2),
    ),
    ("gelu_backward_none", lambda g, x: _gelu_grad_input(g, x, "none")),
    ("gelu_backward_tanh", lambda g, x: _gelu_grad_input(g, x, "tanh")),
    (
        "log_sigmoid_backward",
        lambda g, x: torch.ops.aten.log_sigmoid_backward(
            g, x, torch.ops.aten.log_sigmoid_forward(x)[1]
        ),
    ),
    (
        "elu_backward_result",
        lambda g, x: torch.ops.aten.elu_backward(g, 1.0, 1.0, 1.0, True, F.elu(x)),
    ),
    ("softshrink_backward", lambda g, x: torch.ops.aten.softshrink_backward(g, x, 0.4)),
    ("hardshrink_backward", lambda g, x: torch.ops.aten.hardshrink_backward(g, x, 0.4)),
]


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("name,fn", _BACKWARD, ids=[b[0] for b in _BACKWARD])
def test_backward_ops(mojo_gpu, name, fn, dtype):
    torch.manual_seed(8)
    x = torch.randn(301).to(dtype) * 4
    g = torch.randn(301).to(dtype)
    _close(fn(g.to(mojo_gpu), x.to(mojo_gpu)), fn(g, x), **_tol(dtype, 4))


_F64_TOL = {"rtol": 1e-12, "atol": 1e-14}


@pytest.mark.parametrize("name,fn,op", _ACT, ids=[a[0] for a in _ACT])
def test_activation_float64(mojo_gpu, name, fn, op):
    """float64 computes in double (opmath), with CUDA's constants: float's
    1/6 in hardsigmoid / hardswish, as on CPU."""
    skip_if_metal(mojo_gpu, "Metal has no float64")
    x = _act_input(torch.float64)
    with ran(op):
        actual = fn(x.to(mojo_gpu))
    want = fn(x)
    if name in ("hardsigmoid", "hardswish"):
        # CPU divides by 6; CUDA multiplies by float(1/6), which we follow.
        one_sixth = torch.tensor(1.0 / 6.0, dtype=torch.float32).double()
        r = (x + 3).clamp(0, 6) * one_sixth
        want = r if name == "hardsigmoid" else x * (x + 3).clamp(0, 6) * one_sixth
    _close(actual, want, **_F64_TOL)


@pytest.mark.parametrize("name,fn", _BACKWARD, ids=[b[0] for b in _BACKWARD])
def test_backward_ops_float64(mojo_gpu, name, fn):
    skip_if_metal(mojo_gpu, "Metal has no float64")
    torch.manual_seed(8)
    x = torch.randn(301, dtype=torch.float64) * 4
    g = torch.randn(301, dtype=torch.float64)
    if name.startswith("logit"):
        # The sigmoid of the case is taken on CPU (not an op of this test).
        eps = 0.2 if name.endswith("eps") else None
        s = x.sigmoid()
        got = torch.ops.aten.logit_backward(g.to(mojo_gpu), s.to(mojo_gpu), eps)
        _close(got, torch.ops.aten.logit_backward(g, s, eps), **_F64_TOL)
        return
    _close(fn(g.to(mojo_gpu), x.to(mojo_gpu)), fn(g, x), **_F64_TOL)


def test_rrelu_training_float64(mojo_gpu):
    skip_if_metal(mojo_gpu, "Metal has no float64")
    x_cpu = torch.randn(1001, dtype=torch.float64) * 3
    x = x_cpu.to(mojo_gpu)
    noise = torch.empty_like(x)
    y = torch.ops.aten.rrelu_with_noise(x, noise, 0.1, 0.4, True)
    n = noise.cpu()
    neg = x_cpu <= 0
    assert (n[~neg] == 1).all()
    assert ((n[neg] >= 0.1) & (n[neg] <= 0.4)).all()
    _close(y, x_cpu * n, **_F64_TOL)


def test_backward_grad_input_out_forms(mojo_gpu):
    x = torch.randn(40) * 3
    g = torch.randn(40)
    gi = torch.empty(40, device=mojo_gpu)
    torch.ops.aten.silu_backward.grad_input(
        g.to(mojo_gpu), x.to(mojo_gpu), grad_input=gi
    )
    _close(gi, torch.ops.aten.silu_backward(g, x), **_tol(torch.float32, 3))
    torch.ops.aten.gelu_backward.grad_input(
        g.to(mojo_gpu), x.to(mojo_gpu), grad_input=gi
    )
    _close(gi, torch.ops.aten.gelu_backward(g, x), **_tol(torch.float32, 3))
    torch.ops.aten.logit_backward.grad_input(
        g.to(mojo_gpu), x.sigmoid().to(mojo_gpu), grad_input=gi
    )
    _close(gi, torch.ops.aten.logit_backward(g, x.sigmoid()), **_tol(torch.float32, 3))


@pytest.mark.parametrize("dtype", FLOATS)
def test_rrelu_training(mojo_gpu, dtype):
    """Training draws one slope per negative element into `noise`; the
    output is x * noise, the backward grad * noise, and the draws follow the
    device generator (same seed, same slopes)."""
    lower, upper = 0.1, 0.4
    x_cpu = (torch.randn(4097) * 3).to(dtype)
    x_cpu[:3] = torch.tensor([0.0, -0.0, float("nan")])
    torch.manual_seed(11)
    x = x_cpu.to(mojo_gpu).requires_grad_()
    noise = torch.empty_like(x).detach()
    with ran("aten::rrelu_with_noise"):
        y = torch.ops.aten.rrelu_with_noise(x, noise, lower, upper, True)
    n = noise.cpu().float()
    neg = x_cpu.float() <= 0
    assert (n[~neg] == 1).all()
    assert ((n[neg] >= lower - 1e-2) & (n[neg] <= upper + 1e-2)).all()
    assert n[neg].std() > 0.05  # really random
    _close(y.detach(), x_cpu * n.to(dtype), **_tol(dtype, 2))
    y.backward(torch.ones_like(y))
    _close(x.grad, n.to(dtype), **_tol(dtype, 2))
    torch.manual_seed(11)
    noise2 = torch.empty_like(noise)
    torch.ops.aten.rrelu_with_noise(x.detach(), noise2, lower, upper, True)
    _close(noise2, n.to(dtype), rtol=0.0, atol=0.0)
    z = x.detach().clone()
    F.rrelu(z, lower, upper, training=True, inplace=True)
    assert z.dtype == dtype
    # In place with an explicit noise: self becomes x * noise, noise the slopes.
    z = x.detach().clone()
    noise3 = torch.empty_like(noise)
    with ran("aten::rrelu_with_noise_"):
        torch.ops.aten.rrelu_with_noise_(z, noise3, lower, upper, True)
    n3 = noise3.cpu().float()
    assert (n3[~neg] == 1).all()
    assert ((n3[neg] >= lower - 1e-2) & (n3[neg] <= upper + 1e-2)).all()
    _close(z, x_cpu * n3.to(dtype), **_tol(dtype, 2))


def test_rrelu_noise_aliasing_self(mojo_gpu):
    """rrelu_with_noise(x, x, ...): each element is read before its noise is
    written, as in the CUDA kernel (lower == upper pins the slope)."""
    x = torch.tensor([-2.0, 3.0, -0.5], device=mojo_gpu)
    y = torch.ops.aten.rrelu_with_noise(x, x, 0.25, 0.25, True)
    _close(y, torch.tensor([-0.5, 3.0, -0.125]), rtol=0.0, atol=0.0)
    _close(x, torch.tensor([0.25, 1.0, 0.25]), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", FLOATS)
def test_rrelu_in_place_negative_slope(mojo_gpu, dtype):
    """In place, every slope reads the input as it was: a negative slope
    turns x = -2 into 1, and the noise must still hold -0.5 (it read the
    output's sign and saved 1). With `out` aliasing `noise`, the noise is
    written last, as in the CUDA kernel."""
    x_cpu = torch.tensor([-2.0, 3.0, -0.5])
    want_x, want_n = x_cpu.clone(), torch.empty_like(x_cpu)
    # CPU torch has no half rrelu; lower == upper makes the slope exact.
    torch.ops.aten.rrelu_with_noise_(want_x, want_n, -0.5, -0.5, True)
    x = x_cpu.to(dtype).to(mojo_gpu)
    noise = torch.empty(3, dtype=dtype, device=mojo_gpu)
    torch.ops.aten.rrelu_with_noise_(x, noise, -0.5, -0.5, True)
    _close(x, want_x.to(dtype), rtol=0.0, atol=0.0)
    _close(noise, want_n.to(dtype), rtol=0.0, atol=0.0)
    shared = torch.empty(3, dtype=dtype, device=mojo_gpu)
    torch.ops.aten.rrelu_with_noise.out(
        x_cpu.to(dtype).to(mojo_gpu), shared, -0.5, -0.5, True, out=shared
    )
    _close(shared, want_n.to(dtype), rtol=0.0, atol=0.0)


def test_rrelu_training_out_dtype_and_log_sigmoid_backward_dtypes(mojo_gpu):
    """Training rrelu_with_noise.out writes scalar_t (no cast into another
    dtype), and log_sigmoid_backward takes one dtype (no promotion), both
    raising as CPU torch does."""
    aten = torch.ops.aten
    x_cpu = torch.tensor([-2.0, 3.0])
    x = x_cpu.to(mojo_gpu)
    out = torch.empty(2, dtype=torch.float16, device=mojo_gpu)
    with pytest.raises(RuntimeError):
        aten.rrelu_with_noise.out(
            x_cpu, torch.empty(2), 0.1, 0.3, True, out=torch.empty(2).half()
        )
    with pytest.raises(RuntimeError):
        aten.rrelu_with_noise.out(x, torch.empty_like(x), 0.1, 0.3, True, out=out)
    buf_cpu = aten.log_sigmoid_forward(x_cpu)[1]
    g_cpu = torch.ones(2, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="Found dtype"):
        aten.log_sigmoid_backward(g_cpu, x_cpu, buf_cpu)
    buf = torch.empty(0, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="Found dtype"):
        aten.log_sigmoid_backward(g_cpu.to(mojo_gpu), x, buf)
    with pytest.raises(RuntimeError, match="Found dtype"):
        aten.log_sigmoid_backward.grad_input(
            g_cpu.to(mojo_gpu), x, buf, grad_input=torch.empty(2, device=mojo_gpu)
        )


def test_activation_c_out_overloads(mojo_gpu):
    torch.manual_seed(4)
    g_cpu, x_cpu = torch.randn(3, 5), torch.randn(3, 5) * 3
    g, x = g_cpu.to(mojo_gpu), x_cpu.to(mojo_gpu)
    aten = torch.ops.aten
    tol = _tol(torch.float32, 4)
    out = torch.empty(0, device=mojo_gpu)
    buf = torch.empty(0, device=mojo_gpu)
    with ran("aten::log_sigmoid_forward.output"):
        aten.log_sigmoid_forward.output(x, output=out, buffer=buf)
    _close(out, F.logsigmoid(x_cpu), **tol)
    gi = torch.zeros(5, 3, device=mojo_gpu).t()  # strided destination
    with ran("aten::log_sigmoid_backward.grad_input"):
        aten.log_sigmoid_backward.grad_input(g, x, buf, grad_input=gi)
    want = aten.log_sigmoid_backward(g_cpu, x_cpu, aten.log_sigmoid_forward(x_cpu)[1])
    _close(gi, want, **tol)
    gi = torch.empty(0, device=mojo_gpu)
    with ran("aten::silu_backward.grad_input"):
        aten.silu_backward.grad_input(g, x, grad_input=gi)
    _close(gi, aten.silu_backward(g_cpu, x_cpu), **tol)
    gi = torch.empty(0, device=mojo_gpu)
    with ran("aten::gelu_backward.grad_input"):
        aten.gelu_backward.grad_input(g, x, grad_input=gi)
    _close(gi, aten.gelu_backward(g_cpu, x_cpu), **tol)
    out = torch.empty(0, device=mojo_gpu)
    noise = torch.empty_like(x)
    with ran("aten::rrelu_with_noise.out"):
        aten.rrelu_with_noise.out(x, noise, 0.2, 0.2, True, out=out)
    _close(out, torch.where(x_cpu <= 0, x_cpu * 0.2, x_cpu), **tol)
    # log_sigmoid: out must have self's dtype (no cast).
    with pytest.raises(RuntimeError):
        aten.log_sigmoid_forward.output(
            x,
            output=torch.empty(3, 5, dtype=torch.float16, device=mojo_gpu),
            buffer=buf,
        )


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
def test_hardtanh_threshold_integers(mojo_gpu, dtype):
    x = torch.arange(-8, 9, dtype=dtype)
    _close(F.hardtanh(x.to(mojo_gpu), -3, 5), F.hardtanh(x, -3, 5))
    _close(F.relu6(x.to(mojo_gpu)), F.relu6(x))
    _close(F.threshold(x.to(mojo_gpu), 2, 7), F.threshold(x, 2, 7))


# ---------------------------------------------------------------------------
# elementwise losses: forward and backward for every reduction
# ---------------------------------------------------------------------------

_REDUCTIONS = ["none", "mean", "sum"]


def _loss_tol(dtype: torch.dtype) -> dict[str, float]:
    if dtype == torch.float32:
        return {"rtol": 2e-5, "atol": 1e-5}
    if dtype == torch.float16:
        return {"rtol": 2e-3, "atol": 1e-3}
    return {"rtol": 2e-2, "atol": 1e-2}


def _check_loss(mojo_gpu, fn, inputs, dtype, op, grads=(0,), tol=None):
    """Forward (one native `op`) and backward of `fn(*inputs)` against CPU."""
    tol = tol or _loss_tol(dtype)
    cpu = [t.detach().clone().to(dtype) if t is not None else None for t in inputs]
    dev = [t.to(mojo_gpu) if t is not None else None for t in cpu]
    for i in grads:
        c, d = cpu[i], dev[i]
        assert c is not None and d is not None
        c.requires_grad_()
        d.requires_grad_()
    want = fn(*cpu)
    native.op_counting(True)
    native.op_counts_reset()
    got = fn(*dev)
    # A weight F.* expands to the input's shape is a view (as_strided), not
    # a kernel: the loss itself must be one native op.
    counts = {k: v for k, v in native.op_counts().items() if k != "aten::as_strided"}
    assert counts == {op: 1}, native.op_counts()
    assert got.shape == want.shape and got.dtype == want.dtype
    _close(got, want.detach(), **tol)
    if not grads:
        return
    seed = torch.randn(want.shape).to(dtype)
    want.backward(seed)
    got.backward(seed.to(mojo_gpu))
    for i in grads:
        c, d = cpu[i], dev[i]
        assert c is not None and d is not None
        _close(d.grad, c.grad, **tol)


@pytest.mark.parametrize("reduction", _REDUCTIONS)
@pytest.mark.parametrize("dtype", FLOATS)
def test_mse_loss(mojo_gpu, reduction, dtype):
    torch.manual_seed(0)
    x, y = torch.randn(7, 33), torch.randn(7, 33)
    fn = lambda a, b: F.mse_loss(a, b, reduction=reduction)  # noqa: E731
    _check_loss(mojo_gpu, fn, [x, y], dtype, "aten::mse_loss", grads=(0, 1))


@pytest.mark.parametrize("beta", [1.0, 0.25, 0.0])
@pytest.mark.parametrize("reduction", _REDUCTIONS)
@pytest.mark.parametrize("dtype", FLOATS)
def test_smooth_l1_loss(mojo_gpu, beta, reduction, dtype):
    torch.manual_seed(1)
    x, y = torch.randn(5, 41), torch.randn(5, 41)
    op = "aten::smooth_l1_loss"

    def fn(a, b):
        return torch.ops.aten.smooth_l1_loss(
            a, b, ["none", "mean", "sum"].index(reduction), beta
        )

    if beta == 0.0:
        # F.smooth_l1_loss sends beta=0 to l1_loss; the aten op itself takes
        # it (its backward is 0/0 = NaN where x == target, like CUDA).
        grads = ()
    else:
        grads = (0, 1)
    _check_loss(mojo_gpu, fn, [x, y], dtype, op, grads=grads)


@pytest.mark.parametrize("delta", [1.0, 0.3, 2.5])
@pytest.mark.parametrize("reduction", _REDUCTIONS)
@pytest.mark.parametrize("dtype", FLOATS)
def test_huber_loss(mojo_gpu, delta, reduction, dtype):
    torch.manual_seed(2)
    x, y = torch.randn(6, 29) * 2, torch.randn(6, 29)
    fn = lambda a, b: F.huber_loss(a, b, reduction=reduction, delta=delta)  # noqa: E731
    _check_loss(mojo_gpu, fn, [x, y], dtype, "aten::huber_loss", grads=(0, 1))


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("reduction", _REDUCTIONS)
@pytest.mark.parametrize("dtype", FLOATS)
def test_binary_cross_entropy(mojo_gpu, weighted, reduction, dtype):
    torch.manual_seed(3)
    x = torch.rand(4, 37).clamp(1e-3, 1 - 1e-3)
    x[0, :4] = torch.tensor([0.0, 1.0, 1e-30, 1.0 - 1e-7])  # the -100 clamps
    t = torch.rand(4, 37)
    w = torch.rand(37) + 0.5 if weighted else None
    fn = lambda a, b, c: F.binary_cross_entropy(a, b, weight=c, reduction=reduction)  # noqa: E731
    _check_loss(
        mojo_gpu,
        fn,
        [x, t, w],
        dtype,
        "aten::binary_cross_entropy",
        grads=(0,),
        tol=_loss_tol(dtype)
        if dtype != torch.float32
        else {"rtol": 1e-4, "atol": 1e-4},
    )


@pytest.mark.parametrize("pos_weighted", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("reduction", _REDUCTIONS)
@pytest.mark.parametrize("dtype", FLOATS)
def test_binary_cross_entropy_with_logits(
    mojo_gpu, pos_weighted, weighted, reduction, dtype
):
    torch.manual_seed(4)
    x = torch.randn(3, 5, 11) * 4
    x[0, 0, :4] = torch.tensor([30.0, -30.0, 0.0, 100.0])
    t = torch.rand(3, 5, 11)
    w = torch.rand(5, 11) + 0.5 if weighted else None
    pw = torch.rand(11) * 3 if pos_weighted else None
    fn = lambda a, b, c, d: F.binary_cross_entropy_with_logits(  # noqa: E731
        a, b, weight=c, pos_weight=d, reduction=reduction
    )
    _check_loss(
        mojo_gpu,
        fn,
        [x, t, w, pw],
        dtype,
        "aten::binary_cross_entropy_with_logits",
        grads=(0, 1),
    )


def test_bce_mixed_dtypes_declined(mojo_gpu):
    """Mixed-dtype operands CPU torch accepts are declined (NotImplementedError)
    rather than approximated: a weight / pos_weight of another dtype than
    the input, logits input and target of different dtypes, and a backward
    weight of another dtype."""
    aten = torch.ops.aten
    d = mojo_gpu
    h = torch.tensor([0.3, 0.6]).half().to(d)
    f = torch.tensor([2.0, 1.0], device=d)
    calls = [
        lambda: aten.binary_cross_entropy(h, h, f, 1),
        lambda: aten.binary_cross_entropy_with_logits(h, f, None, None, 1),
        lambda: aten.binary_cross_entropy_with_logits(f, h, None, None, 1),
        lambda: aten.binary_cross_entropy_with_logits(h, h, f, None, 1),
        lambda: aten.binary_cross_entropy_with_logits(h, h, None, f, 1),
        lambda: aten.binary_cross_entropy_backward(torch.ones_like(h), h, h, f, 1),
    ]
    for call in calls:
        with pytest.raises(NotImplementedError):
            call()


@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_bce_weighted_same_dtype_rounding(mojo_gpu, reduction):
    """Same-dtype weights round like ATen's separate steps: the backward's
    `grad_input.mul_(weight)` is float16 inf before the mean divides it."""
    aten = torch.ops.aten
    g = torch.ones(2).half()
    x = torch.tensor([0.5, 0.5]).half()
    t = torch.zeros(2).half()
    w = torch.tensor([40000.0, 40000.0]).half()
    want = aten.binary_cross_entropy_backward(g, x, t, w, reduction)
    got = aten.binary_cross_entropy_backward(
        *[a.to(mojo_gpu) for a in (g, x, t, w)], reduction
    )
    _close(got, want, rtol=0.0, atol=0.0)
    xs = torch.tensor([0.99951171875, 0.25]).half()
    ts = torch.tensor([1.0, 0.0]).half()
    ws = torch.tensor([60000.0, 2.0]).half()
    want = aten.binary_cross_entropy(xs, ts, ws, reduction)
    got = aten.binary_cross_entropy(*[a.to(mojo_gpu) for a in (xs, ts, ws)], reduction)
    _close(got, want, **_loss_tol(torch.float16))


def test_bce_pairs_squeezed_operands(mojo_gpu):
    """ATen's BCE iterators run over squeeze(input) and squeeze(target): a
    [2, 1] input against a [2] target pairs element by element (no
    broadcast to [2, 2]), in input's shape; other shape mismatches are
    declined."""
    aten = torch.ops.aten
    d = mojo_gpu
    x = torch.tensor([[0.25], [0.5]])
    t = torch.tensor([0.0, 1.0])
    g = torch.tensor([[1.0], [0.5]])
    for reduction in (0, 1, 2):
        want = aten.binary_cross_entropy(x, t, None, reduction)
        got = aten.binary_cross_entropy(x.to(d), t.to(d), None, reduction)
        assert got.shape == want.shape
        _close(got, want, **_loss_tol(torch.float32))
        want = aten.binary_cross_entropy(t, x, None, reduction)
        _close(aten.binary_cross_entropy(t.to(d), x.to(d), None, reduction), want)
    want = aten.binary_cross_entropy_backward(g, x, t, None, 0)
    got = aten.binary_cross_entropy_backward(g.to(d), x.to(d), t.to(d), None, 0)
    assert got.shape == want.shape
    _close(got, want, **_loss_tol(torch.float32))
    with pytest.raises(NotImplementedError):
        aten.binary_cross_entropy(torch.tensor([0.3], device=d), t.to(d), None, 0)


def test_bce_out_is_weight(mojo_gpu):
    """`out` / `grad_input` that is the weight: ATen writes the loss there
    first, so `mul_(weight)` reads it back (loss^2, grad^2)."""
    aten = torch.ops.aten
    d = mojo_gpu
    x, t = torch.tensor([0.5, 0.25]), torch.tensor([0.0, 1.0])
    w_cpu = torch.tensor([3.0, 2.0])
    want = aten.binary_cross_entropy.out(x, t, w_cpu, 0, out=w_cpu)
    w = torch.tensor([3.0, 2.0], device=d)
    aten.binary_cross_entropy.out(x.to(d), t.to(d), w, 0, out=w)
    _close(w, want, **_loss_tol(torch.float32))
    g = torch.ones(2)
    for reduction in (0, 1):
        w_cpu = torch.tensor([3.0, 2.0])
        want = aten.binary_cross_entropy_backward.grad_input(
            g, x, t, w_cpu, reduction, grad_input=w_cpu
        )
        w = torch.tensor([3.0, 2.0], device=d)
        aten.binary_cross_entropy_backward.grad_input(
            g.to(d), x.to(d), t.to(d), w, reduction, grad_input=w
        )
        _close(w, want, **_loss_tol(torch.float32))


def test_bce_backward_weight_overlapping_grad_input(mojo_gpu):
    """`grad_input.mul_(weight)` refuses a weight partially overlapping
    grad_input, as CPU torch does."""
    aten = torch.ops.aten
    s = torch.rand(5, device=mojo_gpu)
    ones = torch.ones(4, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="single memory location"):
        aten.binary_cross_entropy_backward.grad_input(
            ones, ones * 0.5, ones * 0, s[1:], 1, grad_input=s[:-1]
        )


def test_bce_logits_weight_overlapping_out(mojo_gpu):
    """A weight sharing memory with `out` is read before `out` is written,
    as the composite's temporary guarantees."""
    aten = torch.ops.aten
    x_cpu = torch.tensor([0.5, -1.0, 2.0])
    t_cpu = torch.tensor([1.0, 0.0, 1.0])
    storage_cpu = torch.tensor([0.3, 2.0, 0.5, 4.0])
    want = aten.binary_cross_entropy_with_logits(
        x_cpu, t_cpu, storage_cpu[1:].clone(), None, 0
    )
    storage = storage_cpu.to(mojo_gpu)
    aten.binary_cross_entropy_with_logits.out(
        x_cpu.to(mojo_gpu), t_cpu.to(mojo_gpu), storage[1:], None, 0, out=storage[:-1]
    )
    _close(storage[:-1], want, **_loss_tol(torch.float32))


def test_bce_dtype_and_shape_rules(mojo_gpu):
    """binary_cross_entropy: input and target of one dtype, the loss in it;
    with logits the loss has target's dtype. No weight may grow the loss's
    shape."""
    aten = torch.ops.aten
    d = mojo_gpu
    p = torch.tensor([0.3, 0.9, 0.5, 0.01])
    y = torch.tensor([1.0, 0.0, 0.5, 1.0])
    w = torch.tensor([2.0, 1.0, 0.5, 3.0])
    ph, yh, wh = p.half(), y.half(), w.half()
    got = aten.binary_cross_entropy(ph.to(d), yh.to(d), wh.to(d), 0)
    assert got.dtype == torch.float16
    _close(got, aten.binary_cross_entropy(ph, yh, wh, 0), **_loss_tol(torch.float16))
    with pytest.raises(RuntimeError, match="Found dtype"):
        aten.binary_cross_entropy(ph.to(d), y.to(d), None, 1)
    for x_, t_, pw in ((ph, yh, None), (ph, yh, wh), (p, y, w)):
        want = aten.binary_cross_entropy_with_logits(x_, t_, None, pw, 1)
        got = aten.binary_cross_entropy_with_logits(
            x_.to(d), t_.to(d), None, None if pw is None else pw.to(d), 1
        )
        assert got.dtype == want.dtype == t_.dtype
        _close(got, want, **_loss_tol(t_.dtype))
    half = torch.full((2, 1), 0.5, device=d)
    wide = torch.ones(2, 3, device=d)
    with pytest.raises(RuntimeError, match="doesn't match the broadcast shape"):
        aten.binary_cross_entropy(half, half, wide, 0)
    with pytest.raises(RuntimeError, match="doesn't match the broadcast shape"):
        aten.binary_cross_entropy_backward(half, half, half, wide, 0)
    for w, pw in ((wide, None), (None, wide)):
        with pytest.raises(RuntimeError, match="doesn't match the broadcast shape"):
            aten.binary_cross_entropy_with_logits(half, half, w, pw, 0)


def test_bce_out_overloads(mojo_gpu):
    """The out= forms: resized from empty, written into a strided out, and
    an out of another dtype refused (no cast)."""
    aten = torch.ops.aten
    d = mojo_gpu
    torch.manual_seed(5)
    p, y = torch.rand(3, 4) * 0.9 + 0.05, torch.rand(3, 4)
    g = torch.randn(3, 4)
    dev = [t.to(d) for t in (p, y)]
    for reduction in (0, 1, 2):
        out = torch.empty(0, device=d)
        aten.binary_cross_entropy.out(*dev, None, reduction, out=out)
        _close(out, aten.binary_cross_entropy(p, y, None, reduction))
        out = torch.empty(0, device=d)
        aten.binary_cross_entropy_with_logits.out(*dev, None, None, reduction, out=out)
        _close(out, aten.binary_cross_entropy_with_logits(p, y, None, None, reduction))
    strided = torch.zeros(4, 3, device=d).t()
    aten.binary_cross_entropy.out(*dev, None, 0, out=strided)
    _close(strided, aten.binary_cross_entropy(p, y, None, 0))
    gi = torch.zeros(4, 3, device=d).t()
    aten.binary_cross_entropy_backward.grad_input(g.to(d), *dev, None, 0, grad_input=gi)
    _close(
        gi,
        aten.binary_cross_entropy_backward(g, p, y, None, 0),
        **_tol(torch.float32, 4),
    )
    narrow = torch.empty(3, 4, dtype=torch.float16, device=d)
    with pytest.raises(RuntimeError):
        aten.binary_cross_entropy.out(*dev, None, 0, out=narrow)
    with pytest.raises(RuntimeError):
        aten.binary_cross_entropy_backward.grad_input(
            g.to(d), *dev, None, 0, grad_input=narrow
        )
    with pytest.raises(RuntimeError):
        aten.binary_cross_entropy_with_logits.out(*dev, None, None, 0, out=narrow)


def test_loss_out_and_edge_shapes(mojo_gpu):
    x, y = torch.randn(3, 4), torch.randn(3, 4)
    for reduction in (0, 1, 2):
        out = torch.empty(0, device=mojo_gpu)
        torch.ops.aten.mse_loss.out(x.to(mojo_gpu), y.to(mojo_gpu), reduction, out=out)
        _close(
            out, torch.ops.aten.mse_loss(x, y, reduction), **_loss_tol(torch.float32)
        )
        out = torch.empty(0, device=mojo_gpu)
        torch.ops.aten.huber_loss.out(
            x.to(mojo_gpu), y.to(mojo_gpu), reduction, 0.5, out=out
        )
        _close(
            out,
            torch.ops.aten.huber_loss(x, y, reduction, 0.5),
            **_loss_tol(torch.float32),
        )
    # 0-d operands, broadcasting target, non-contiguous input
    a, b = torch.tensor(1.5), torch.tensor(-0.25)
    _close(F.mse_loss(a.to(mojo_gpu), b.to(mojo_gpu)), F.mse_loss(a, b))
    xt = torch.randn(4, 3).t()
    _close(
        F.smooth_l1_loss(
            xt.to(mojo_gpu), y[:1].to(mojo_gpu).expand(3, 4), reduction="none"
        ),
        F.smooth_l1_loss(xt, y[:1].expand(3, 4), reduction="none"),
    )
    with pytest.raises(RuntimeError, match="non-positive"):
        F.huber_loss(x.to(mojo_gpu), y.to(mojo_gpu), delta=0.0)


@pytest.mark.parametrize("reduction", [1, 2])
def test_huber_out_reduced_computes_in_out_dtype(mojo_gpu, reduction):
    """huber_loss.out runs its kernel on `out` for every reduction, then
    reduces it: float16 400 against a float32 target with delta 1000 is
    80000 in a float32 `out`, as CPU torch computes, not float16 inf."""
    aten = torch.ops.aten
    h = torch.tensor([400.0, -300.0]).half()
    f = torch.tensor([0.0, 0.0])
    want = aten.huber_loss.out(h, f, reduction, 1000.0, out=torch.empty(()))
    out = torch.empty((), device=mojo_gpu)
    aten.huber_loss.out(h.to(mojo_gpu), f.to(mojo_gpu), reduction, 1000.0, out=out)
    _close(out, want, **_loss_tol(torch.float32))


def test_loss_dtype_rules(mojo_gpu):
    """The loss kernels compute in their output's dtype (`iter.dtype()`):
    huber's is `empty_like(input)`, an out='s its own, and a reduction
    reduces straight into the out's dtype; beta is `scalar_t` of that."""
    aten = torch.ops.aten
    h = torch.tensor([2.0, -3.0, 0.25, 100.0]).half()
    f = torch.tensor([0.0, 1.0, 0.5, -100.0])
    got = aten.huber_loss(h.to(mojo_gpu), f.to(mojo_gpu), 0, 1.0)
    assert got.dtype == torch.float16
    _close(got, aten.huber_loss(h, f, 0, 1.0), **_loss_tol(torch.float16))
    # sum of ten float16 squares of 100 into a float32 out: 1e5, not inf
    big = torch.full((10,), 100.0).half()
    out = torch.empty((), device=mojo_gpu)
    aten.mse_loss.out(big.to(mojo_gpu), big.to(mojo_gpu) * 0, 2, out=out)
    assert out.item() == 100000.0
    # beta = 1e5 in float32 (the promoted dtype), not float16 inf
    for red in (0, 1, 2):
        want = aten.smooth_l1_loss(h, f, red, 1e5)
        got = aten.smooth_l1_loss(h.to(mojo_gpu), f.to(mojo_gpu), red, 1e5)
        _close(got, want, **_loss_tol(torch.float32))
        for od in (torch.float32, torch.float16):
            if red == 0 and od == torch.float16:
                # CUDA (and the device) compute in the float16 out (beta is
                # float16 inf there); CPU torch keeps the promoted float32.
                continue
            o = torch.empty(4 if red == 0 else (), dtype=od, device=mojo_gpu)
            want = aten.smooth_l1_loss.out(
                h, f, red, 1e5, out=torch.empty(o.shape, dtype=od)
            )
            aten.smooth_l1_loss.out(h.to(mojo_gpu), f.to(mojo_gpu), red, 1e5, out=o)
            _close(o, want, **_loss_tol(od))
    # smooth_l1 backward into a float32 grad_input from a float16 self
    g = torch.tensor(100000.0)
    gi = torch.empty(4, device=mojo_gpu)
    aten.smooth_l1_loss_backward.grad_input(
        g.to(mojo_gpu), h.to(mojo_gpu), f.to(mojo_gpu), 0, 1.0, grad_input=gi
    )
    want = aten.smooth_l1_loss_backward.grad_input(
        g, h, f, 0, 1.0, grad_input=torch.empty(4)
    )
    _close(gi, want)
    # mse / huber backward: one dtype for every operand
    for op, extra in ((aten.mse_loss_backward, ()), (aten.huber_loss_backward, (1.0,))):
        with pytest.raises(RuntimeError, match="Found dtype"):
            op(g.to(mojo_gpu), h.to(mojo_gpu), f.to(mojo_gpu), 0, *extra)
    with pytest.raises(RuntimeError, match="negative values for beta"):
        F.smooth_l1_loss(f.to(mojo_gpu), f.to(mojo_gpu), beta=float("nan"))


@pytest.mark.parametrize(
    "op,extra", [("mse_loss", ()), ("smooth_l1_loss", (0.5,)), ("huber_loss", (0.5,))]
)
def test_loss_backward_grad_input(mojo_gpu, op, extra):
    """The backward out= overloads: resized from empty, or written where a
    strided grad_input lives."""
    aten = torch.ops.aten
    fn = getattr(aten, f"{op}_backward")
    out_fn = fn.grad_input if hasattr(fn, "grad_input") else fn.out
    torch.manual_seed(3)
    x, y = torch.randn(3, 4), torch.randn(3, 4)
    for reduction in (0, 1):
        g = torch.randn(3, 4) if reduction == 0 else torch.tensor(1.5)
        want = fn(g, x, y, reduction, *extra)
        dev = [t.to(mojo_gpu) for t in (g, x, y)]
        gi = torch.empty(0, device=mojo_gpu)
        out_fn(*dev, reduction, *extra, grad_input=gi)
        _close(gi, want, **_loss_tol(torch.float32))
        strided = torch.zeros(4, 3, device=mojo_gpu).t()
        out_fn(*dev, reduction, *extra, grad_input=strided)
        _close(strided, want, **_loss_tol(torch.float32))
    out = torch.empty(0, device=mojo_gpu)
    aten.smooth_l1_loss.out(x.to(mojo_gpu), y.to(mojo_gpu), 0, 0.5, out=out)
    _close(out, aten.smooth_l1_loss(x, y, 0, 0.5), **_loss_tol(torch.float32))
