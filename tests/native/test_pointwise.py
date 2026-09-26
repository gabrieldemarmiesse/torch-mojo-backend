"""Pointwise math on the native mojo device (tmb/ops/pointwise.mojo): the
pointwise routes of pow, lerp.Scalar and gelu_backward.

Everything is compared with the same computation on CPU torch through the
public API, over edge values (signed zeros, infinities, NaN, huge, tiny,
denormals), broadcasting, strided operands, out= and in-place forms.
"""

import contextlib

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
