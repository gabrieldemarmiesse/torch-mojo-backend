"""Pointwise math on the native mojo device (tmb/ops/pointwise.mojo): pow,
lerp (Scalar and Tensor), gelu_backward, clamp.Tensor, rsub and the binary
math family (atan2, hypot, copysign, fmod, fmax, fmin, heaviside, nextafter,
gcd, lcm, bitwise_left_shift.Tensor, bitwise_right_shift.Tensor so far).

Everything is compared with the same computation on CPU torch through the
public API, over edge values (signed zeros, infinities, NaN, huge, tiny,
denormals), broadcasting, strided operands, out= and in-place forms.
"""

import contextlib
import itertools
import math

import pytest
import torch

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
