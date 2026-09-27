"""Pointwise math on the native mojo device (tmb/ops/pointwise.mojo): the
pointwise routes of pow, lerp.Scalar and gelu_backward.

Everything is compared with the same computation on CPU torch through the
public API, over edge values (signed zeros, infinities, NaN, huge, tiny,
denormals), broadcasting, strided operands, out= and in-place forms.
"""

import contextlib
import math

import pytest
import torch

from tests.native.conftest import skip_if_metal
from torch_mojo_backend import native

FLOATS = [torch.float32, torch.float16, torch.bfloat16]


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
    special = ~torch.isfinite(want) | (want == 0)
    assert torch.equal(got[special], want[special])
    finite = ~special
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
