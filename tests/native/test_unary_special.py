"""Tests for the unary math and special functions of the native `unary` op
group (torch_mojo_backend/mojo/tmb/ops/unary.mojo): asin/atan/erfc/erfinv/
exp2/expm1/log10/sinc/angle/frac/trunc/round/sgn/signbit/nan_to_num and the
torch.special
Bessel functions (bessel_j0/j1/y0/y1, modified_bessel_i0/i1/k0/k1,
scaled_modified_bessel_k0/k1, spherical_bessel_j0), i0e/i1/i1e, airy_ai,
entr, erfcx, log_ndtr and ndtri so far, whose kernels port the CUDA
routines stock torch runs.

Public torch API only, compared against CPU torch.
"""

from collections.abc import Callable

import pytest
import torch

from tests.native.conftest import is_metal, skip_if_metal
from torch_mojo_backend import native

S = torch.special

_EDGES = [
    0.0,
    -0.0,
    float("inf"),
    -float("inf"),
    float("nan"),
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.0,
    5.0,
    8.0,
    -8.0,
    1e-30,
    -1e-30,
    1e30,
    -1e30,
]

# (native op name, torch callable, sampled interval). Every interval crosses
# the regime switches of its algorithm (bessel at 5 and 8, erfcx at 50 and
# -6.1, lgamma at 0.7/1.5/3/7.8, digamma's reflection below 0, ...).
_CASES: list[tuple[str, Callable[[torch.Tensor], torch.Tensor], float, float]] = [
    ("asin", torch.asin, -1.0, 1.0),
    ("atan", torch.atan, -50.0, 50.0),
    ("erfc", torch.erfc, -5.0, 12.0),
    ("erfinv", torch.erfinv, -1.0, 1.0),
    ("exp2", torch.exp2, -130.0, 130.0),
    ("expm1", torch.expm1, -20.0, 90.0),
    ("log10", torch.log10, 0.0, 1e6),
    ("sinc", torch.sinc, -20.0, 20.0),
    ("special_airy_ai", S.airy_ai, -20.0, 20.0),
    ("special_bessel_j0", S.bessel_j0, -30.0, 30.0),
    ("special_bessel_j1", S.bessel_j1, -30.0, 30.0),
    ("special_bessel_y0", S.bessel_y0, 0.0, 30.0),
    ("special_bessel_y1", S.bessel_y1, 0.0, 30.0),
    ("special_entr", S.entr, -1.0, 20.0),
    ("special_erfcx", S.erfcx, -10.0, 60.0),
    ("special_i0e", S.i0e, -30.0, 30.0),
    ("special_i1", S.i1, -30.0, 30.0),
    ("special_i1e", S.i1e, -30.0, 30.0),
    ("special_log_ndtr", S.log_ndtr, -30.0, 10.0),
    ("special_modified_bessel_i0", S.modified_bessel_i0, -30.0, 30.0),
    ("special_modified_bessel_i1", S.modified_bessel_i1, -30.0, 30.0),
    ("special_modified_bessel_k0", S.modified_bessel_k0, 0.0, 30.0),
    ("special_modified_bessel_k1", S.modified_bessel_k1, 0.0, 30.0),
    ("special_ndtri", S.ndtri, 0.0, 1.0),
    ("special_scaled_modified_bessel_k0", S.scaled_modified_bessel_k0, 0.0, 30.0),
    ("special_scaled_modified_bessel_k1", S.scaled_modified_bessel_k1, 0.0, 30.0),
    ("special_spherical_bessel_j0", S.spherical_bessel_j0, -30.0, 30.0),
    ("angle", torch.angle, -5.0, 5.0),
    ("frac", torch.frac, -100.0, 100.0),
    ("trunc", torch.trunc, -100.0, 100.0),
    ("round", torch.round, -100.0, 100.0),
    ("sgn", torch.sgn, -5.0, 5.0),
]
_IDS = [case[0] for case in _CASES]


def _counted(name: str) -> bool:
    return native.op_count(f"aten::{name}") > 0


def _reset_counts():
    native.op_counting(True)
    native.op_counts_reset()


def _sample(lo: float, hi: float, dtype: torch.dtype, n: int = 2000) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n, generator=g, dtype=torch.float64) * (hi - lo) + lo
    return torch.cat([x, torch.tensor(_EDGES, dtype=torch.float64)]).to(dtype)


def _expected(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor):
    """CPU torch in the input dtype, or through float32 for the special
    functions CPU torch does not implement for half types (the device
    computes those in float32 and rounds once)."""
    try:
        return fn(x)
    except (RuntimeError, NotImplementedError):
        return fn(x.float()).to(x.dtype)


# On Apple GPUs these ops port torch's MPS kernels, whose results differ
# from CPU torch beyond the test tolerance, so the reference there is stock
# torch on the same machine (the `mps` device): round_decimals'
# rint(exp10(n) x) exp10(-n) in float.
_MPS_SEMANTICS: set[str] = set()


def _reference(
    device: str, op_name: str, fn: Callable[..., torch.Tensor], x: torch.Tensor
) -> torch.Tensor:
    if op_name in _MPS_SEMANTICS and is_metal(device):
        return fn(x.to("mps")).cpu()
    return _expected(fn, x)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("op_name,fn,lo,hi", _CASES, ids=_IDS)
def test_matches_cpu(
    mojo_gpu: str,
    op_name: str,
    fn: Callable[[torch.Tensor], torch.Tensor],
    lo: float,
    hi: float,
    dtype: torch.dtype,
):
    x = _sample(lo, hi, dtype)
    _reset_counts()
    actual = fn(x.to(mojo_gpu))
    assert _counted(op_name), f"aten::{op_name} did not run natively"
    expected = _reference(mojo_gpu, op_name, fn, x)
    torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)


@pytest.mark.parametrize("op_name,fn,lo,hi", _CASES, ids=_IDS)
def test_layouts_out_and_inplace(
    mojo_gpu: str, op_name: str, fn: Callable[..., torch.Tensor], lo: float, hi: float
):
    """A transposed input, an unaligned (offset 1) input, out= into a
    strided view and the in-place method (which ATen routes to .out)."""
    x = _sample(lo, hi, torch.float32, n=15 * 17 - len(_EDGES)).reshape(15, 17)
    expected = _reference(mojo_gpu, op_name, fn, x)
    device = x.to(mojo_gpu)
    torch.testing.assert_close(fn(device.t()).cpu(), expected.t(), equal_nan=True)
    storage = torch.cat((torch.zeros(1), x.flatten())).to(mojo_gpu)
    torch.testing.assert_close(
        fn(storage[1:].view(15, 17)).cpu(), expected, equal_nan=True
    )
    out_base = torch.full((17, 15), -7.0).to(mojo_gpu)
    out = out_base.t()
    fn(device, out=out)
    torch.testing.assert_close(out.cpu(), expected, equal_nan=True)
    method = getattr(torch.Tensor, getattr(fn, "__name__", "") + "_", None)
    if method is not None and not op_name.startswith("special_"):
        inplace = device.clone()
        method(inplace)
        torch.testing.assert_close(inplace.cpu(), expected, equal_nan=True)


@pytest.mark.parametrize("fn", (torch.angle, torch.frac, torch.trunc, torch.round))
def test_float64(mojo_gpu: str, fn: Callable[[torch.Tensor], torch.Tensor]):
    skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    x = _sample(-100.0, 100.0, torch.float64)
    torch.testing.assert_close(fn(x.to(mojo_gpu)).cpu(), fn(x), equal_nan=True)


def test_special_functions_decline_float64(mojo_gpu: str):
    x = torch.rand(8, dtype=torch.float64).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        S.i0e(x)


@pytest.mark.parametrize("dtype", (torch.int8, torch.int32, torch.int64, torch.uint8))
@pytest.mark.parametrize(
    "fn", (torch.trunc, torch.round, torch.sgn, torch.signbit, torch.nan_to_num)
)
def test_integer_inputs(
    mojo_gpu: str, fn: Callable[[torch.Tensor], torch.Tensor], dtype: torch.dtype
):
    lo = 0 if dtype == torch.uint8 else -100
    x = torch.randint(lo, 100, (5, 7), dtype=dtype)
    actual = fn(x.to(mojo_gpu))
    assert actual.dtype == fn(x).dtype
    assert torch.equal(actual.cpu(), fn(x))


@pytest.mark.parametrize("fn", (torch.sgn, torch.signbit, torch.nan_to_num))
def test_bool_inputs(mojo_gpu: str, fn: Callable[..., torch.Tensor]):
    x = torch.tensor([True, False, True])
    assert torch.equal(fn(x.to(mojo_gpu)).cpu(), fn(x))
    out = torch.ones(5, dtype=fn(x).dtype).to(mojo_gpu)
    fn(x.to(mojo_gpu), out=out)
    assert torch.equal(out.cpu(), fn(x))


@pytest.mark.parametrize(
    "dtype", (torch.float32, torch.float16, torch.bfloat16, torch.float64)
)
def test_signbit(mojo_gpu: str, dtype: torch.dtype):
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    x = torch.tensor([-0.0, 0.0, -1.5, 2.0, float("inf"), -float("inf")], dtype=dtype)
    x = torch.cat([x, -torch.tensor([float("nan")], dtype=dtype)])
    _reset_counts()
    actual = torch.signbit(x.to(mojo_gpu))
    assert _counted("signbit")
    assert torch.equal(actual.cpu(), torch.signbit(x))


def _round_decimals_reference(x: torch.Tensor, decimals: int) -> torch.Tensor:
    """round_decimals_kernel_cuda: CPU torch for float32/float64; for the
    half types CUDA's scalar_t arithmetic (each product and quotient rounds
    to the dtype, 10^|decimals| too), which CPU torch does not reproduce."""
    if x.dtype in (torch.float32, torch.float64):
        return torch.round(x, decimals=decimals)
    ten_pow = torch.tensor(10.0 ** abs(decimals)).to(x.dtype).float()
    if decimals < 0:
        q = (x.float() / ten_pow).to(x.dtype).float()
        return (torch.round(q) * ten_pow).to(x.dtype)
    q = (x.float() * ten_pow).to(x.dtype).float()
    return (torch.round(q) / ten_pow).to(x.dtype)


@pytest.mark.parametrize(
    "dtype", (torch.float32, torch.float16, torch.bfloat16, torch.float64)
)
@pytest.mark.parametrize("decimals", (0, 1, 3, -1, -2))
def test_round_decimals(mojo_gpu: str, dtype: torch.dtype, decimals: int):
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    x = _sample(-500.0, 500.0, dtype)
    if is_metal(mojo_gpu):  # MPS's round_decimals_functor, see _MPS_SEMANTICS
        expected = torch.round(x.to("mps"), decimals=decimals).cpu()
    else:
        expected = _round_decimals_reference(x, decimals)
    actual = torch.round(x.to(mojo_gpu), decimals=decimals)
    torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)
    out = torch.empty_like(x).to(mojo_gpu)
    torch.round(x.to(mojo_gpu), decimals=decimals, out=out)
    torch.testing.assert_close(out.cpu(), expected, equal_nan=True)
    inplace = x.to(mojo_gpu)
    inplace.round_(decimals=decimals)
    torch.testing.assert_close(inplace.cpu(), expected, equal_nan=True)


@pytest.mark.parametrize(
    "dtype", (torch.float32, torch.float16, torch.bfloat16, torch.float64)
)
@pytest.mark.parametrize(
    "nan,posinf,neginf", ((None, None, None), (1.5, None, None), (0.0, 7.0, -9.0))
)
def test_nan_to_num(
    mojo_gpu: str,
    dtype: torch.dtype,
    nan: float | None,
    posinf: float | None,
    neginf: float | None,
):
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    x = _sample(-10.0, 10.0, dtype)
    expected = torch.nan_to_num(x, nan, posinf, neginf)
    _reset_counts()
    actual = torch.nan_to_num(x.to(mojo_gpu), nan, posinf, neginf)
    assert _counted("nan_to_num")
    torch.testing.assert_close(actual.cpu(), expected)
    out = torch.empty(x.numel() * 2, dtype=dtype).to(mojo_gpu)[::2]
    torch.nan_to_num(x.to(mojo_gpu), nan, posinf, neginf, out=out)
    torch.testing.assert_close(out.cpu(), expected)


def test_nan_to_num_overflowing_replacement(mojo_gpu: str):
    """A replacement beyond the dtype's range becomes inf, as the C++ cast does."""
    x = torch.tensor([float("nan"), float("inf")], dtype=torch.float16)
    actual = torch.nan_to_num(x.to(mojo_gpu), nan=1e6, posinf=1e10)
    assert torch.equal(actual.cpu(), torch.nan_to_num(x, nan=1e6, posinf=1e10))
