"""Tests for the complex arithmetic and math of the mojo device
(torch_mojo_backend/mojo/tmb/ops/complex_math.mojo): add / sub / mul / div /
pow, the unary math, complex() / polar(), sum / mean / cumsum.

Public torch API only, compared against CPU torch (or values recorded from
stock CUDA torch where CPU and CUDA differ). complex128 needs float64, which
Apple GPUs do not have.
"""

import pathlib
import warnings

import pytest
import torch

from tests.native.conftest import skip_if_metal
from tests.native.test_complex import CDTYPES, _close, _put, _rand, _skip_c128
from torch_mojo_backend import native

BINARY = [torch.add, torch.sub, torch.mul, torch.div]


@pytest.mark.parametrize("dtype", CDTYPES)
@pytest.mark.parametrize("op", BINARY, ids=lambda f: f.__name__)
def test_binary(mojo_gpu: str, dtype: torch.dtype, op):
    _skip_c128(mojo_gpu, dtype)
    x = _rand(4, 5, dtype=dtype)
    y = _rand(4, 5, dtype=dtype)
    d, e = x.to(mojo_gpu), y.to(mojo_gpu)
    _close(op(d, e), op(x, y))
    _close(op(d, 2.5), op(x, 2.5))
    _close(op(d, 1 - 2j), op(x, 1 - 2j))
    _close(op(d, 3), op(x, 3))
    # broadcasting, 0-d and real operands
    r = torch.randn(5)
    _close(op(d, r.to(mojo_gpu)), op(x, r))
    _close(op(r.to(mojo_gpu), d), op(r, x))
    s = _rand(dtype=dtype)
    _close(op(d, s.to(mojo_gpu)), op(x, s))
    _close(op(d[:, :1], e), op(x[:, :1], y))
    # non-contiguous
    _close(op(d.t(), e.t()), op(x.t(), y.t()))
    # a real tensor with a complex scalar promotes to complex
    _close(op(r.to(mojo_gpu), 2j), op(r, 2j))
    # out= and in-place
    out = torch.empty(0, dtype=dtype, device=mojo_gpu)
    op(d, e, out=out)
    _close(out, op(x, y))
    g = d.clone()
    getattr(g, op.__name__ + "_")(e)
    _close(g, op(x, y))


def test_binary_promotion(mojo_gpu: str):
    skip_if_metal(mojo_gpu, "complex128 needs float64")
    x = _rand(3)
    d = x.to(mojo_gpu)
    r64 = torch.randn(3, dtype=torch.float64)
    assert (d + r64.to(mojo_gpu)).dtype == torch.complex128
    _close(d * r64.to(mojo_gpu), x * r64)
    # a 0-d float64 meets a dimensioned complex64 in complex64
    z = torch.tensor(2.0, dtype=torch.float64)
    assert (d * z.to(mojo_gpu)).dtype == torch.complex64
    c128 = torch.randn(3, dtype=torch.complex128)
    _close(d + c128.to(mojo_gpu), x + c128)


def test_add_alpha(mojo_gpu: str):
    x, y = _rand(6), _rand(6)
    d, e = x.to(mojo_gpu), y.to(mojo_gpu)
    _close(torch.add(d, e, alpha=2), torch.add(x, y, alpha=2))
    _close(torch.add(d, e, alpha=1j), torch.add(x, y, alpha=1j))
    _close(torch.sub(d, e, alpha=-0.5), torch.sub(x, y, alpha=-0.5))


def test_inplace_errors(mojo_gpu: str):
    r = torch.randn(3).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="can't be cast"):
        r.add_(_rand(3).to(mojo_gpu))
    d = _rand(3).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="doesn't match the broadcast shape"):
        d.add_(_rand(2, 3).to(mojo_gpu))


def test_division_special_values(mojo_gpu: str):
    num = torch.tensor(
        [1 + 1j, 1e30 + 1e30j, 1e-30 + 1e-30j, 3 - 4j], dtype=torch.complex64
    )
    den = torch.tensor(
        [0 + 0j, 1e30 - 1e30j, 1e-30 + 2e-30j, 1e-20 + 1e20j], dtype=torch.complex64
    )
    _close(num.to(mojo_gpu) / den.to(mojo_gpu), num / den, equal_nan=True)


UNARY = [
    torch.exp,
    torch.log,
    torch.sqrt,
    torch.rsqrt,
    torch.reciprocal,
    torch.sgn,
    torch.neg,
]


@pytest.mark.parametrize("dtype", CDTYPES)
@pytest.mark.parametrize("op", UNARY, ids=lambda f: f.__name__)
def test_unary(mojo_gpu: str, dtype: torch.dtype, op):
    _skip_c128(mojo_gpu, dtype)
    x = _rand(6, 7, dtype=dtype) * 3
    _put(x, (0, 0), 0)
    _put(x, (0, 1), 1)
    _put(x, (0, 2), -1)
    _put(x, (0, 3), 1j)
    _put(x, (0, 4), -2 + 0j)
    d = x.to(mojo_gpu)
    _close(op(d), op(x), equal_nan=True)
    _close(op(d.t()), op(x.t()), equal_nan=True)
    out = torch.empty(0, dtype=dtype, device=mojo_gpu)
    op(d, out=out)
    _close(out, op(x), equal_nan=True)


# sgn and rsqrt: stock CUDA (the reference) and CPU disagree on infinite
# inputs; checked against CUDA's own values below.
@pytest.mark.parametrize(
    "op",
    [f for f in UNARY if f not in (torch.sgn, torch.rsqrt)],
    ids=lambda f: f.__name__,
)
def test_unary_special_values(mojo_gpu: str, op):
    inf, nan = float("inf"), float("nan")
    vals = [
        complex(a, b)
        for a in (0.0, -0.0, 1.0, -1.0, inf, -inf, nan)
        for b in (0.0, -0.0, 2.0, inf, -inf, nan)
    ]
    x = torch.tensor(vals, dtype=torch.complex64)
    got = op(x.to(mojo_gpu)).cpu()
    want = op(x)
    # Annex G special values (as glibc and thrust define them); NaN signs and
    # the sign of a NaN's partner part are unspecified, so compare NaN-ness
    # and the finite / infinite values.
    for g, w, v in zip(got.tolist(), want.tolist(), vals, strict=True):
        if op is torch.reciprocal and v == 0:
            # 1 / (+-0 + +-0i): c10 divides by std::abs on gcc builds and by
            # `x < 0 ? -x : x` on clang ones (macOS torch), which flips the
            # infinity's sign; CUDA's is +inf.
            assert g.real == float("inf"), (v, g)
            continue
        for gp, wp in ((g.real, w.real), (g.imag, w.imag)):
            if wp != wp:
                assert gp != gp, (op, v, g, w)
            else:
                assert gp == pytest.approx(wp, rel=1e-5, abs=1e-6), (op, v, g, w)


def _same(got: complex, want: complex) -> bool:
    for g, w in ((got.real, want.real), (got.imag, want.imag)):
        if w != w:
            if g == g:
                return False
        elif g != pytest.approx(w, rel=1e-5, abs=1e-30):
            return False
    return True


def test_sgn_rsqrt_match_cuda(mojo_gpu: str):
    """Values stock torch computes on CUDA (its jiterated libc++ complex
    math), where CPU torch gives NaNs."""
    inf, nan = float("inf"), float("nan")
    x = torch.tensor(
        [complex(-inf, 2), complex(2, -inf), complex(inf, inf), 0j],
        dtype=torch.complex64,
    )
    got = torch.sgn(x.to(mojo_gpu)).cpu().tolist()
    want = [complex(nan, 0), complex(0, nan), complex(nan, nan), 0j]
    assert all(_same(g, w) for g, w in zip(got, want, strict=True)), got
    y = torch.tensor(
        [complex(0, inf), complex(inf, 0), complex(-inf, 1), complex(1e30, 1e30), 0j],
        dtype=torch.complex64,
    )
    got = torch.rsqrt(y.to(mojo_gpu)).cpu().tolist()
    want = [
        -0j,
        0j,
        -0j,
        complex(7.768870254581787e-16, -3.217971340243285e-16),
        complex(inf, nan),
    ]
    assert all(_same(g, w) for g, w in zip(got, want, strict=True)), got


@pytest.mark.parametrize("dtype", CDTYPES)
def test_to_real(mojo_gpu: str, dtype: torch.dtype):
    _skip_c128(mojo_gpu, dtype)
    x = _rand(5, 6, dtype=dtype)
    _put(x, (0, 0), 0)
    _put(x, (0, 1), -3)
    _put(x, (0, 2), 4j)
    _put(x, (1, 0), complex(float("inf"), 1))
    _put(x, (1, 1), complex(1, float("nan")))
    d = x.to(mojo_gpu)
    _close(torch.abs(d), torch.abs(x), equal_nan=True)
    _close(torch.angle(d), torch.angle(x), equal_nan=True)
    for f in (torch.isnan, torch.isinf, torch.isfinite):
        assert torch.equal(f(d).cpu(), f(x))
    out = torch.empty(0, dtype=x.real.dtype, device=mojo_gpu)
    torch.abs(d, out=out)
    _close(out, torch.abs(x), equal_nan=True)


@pytest.mark.parametrize("dtype", CDTYPES)
def test_pow(mojo_gpu: str, dtype: torch.dtype):
    _skip_c128(mojo_gpu, dtype)
    x = _rand(4, 4, dtype=dtype) + 0.5
    y = _rand(4, 4, dtype=dtype) * 0.5
    d, e = x.to(mojo_gpu), y.to(mojo_gpu)
    rtol, atol = (1e-4, 1e-5) if dtype == torch.complex64 else (1e-10, 1e-12)
    for p in (2, 3, 0.5, -0.5, -1, 0, 1, 1.7, 0.5 + 1j):
        _close(d**p, x**p, rtol=rtol, atol=atol)
    _close(d**e, x**y, rtol=rtol, atol=atol)
    _close(2**d, 2**x, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_complex_and_polar(mojo_gpu: str, dtype: torch.dtype):
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "float64")
    re = torch.randn(4, 5, dtype=dtype)
    im = torch.randn(5, dtype=dtype)
    _close(torch.complex(re.to(mojo_gpu), im.to(mojo_gpu)), torch.complex(re, im))
    a = torch.rand(4, 5, dtype=dtype)
    t = torch.randn(4, 5, dtype=dtype) * 4
    _close(torch.polar(a.to(mojo_gpu), t.to(mojo_gpu)), torch.polar(a, t))
    out = torch.empty(
        0,
        dtype=re.to(
            torch.complex64 if dtype == torch.float32 else torch.complex128
        ).dtype,
        device=mojo_gpu,
    )
    torch.complex(re.to(mojo_gpu), im.to(mojo_gpu), out=out)
    _close(out, torch.complex(re, im))
    with pytest.raises(RuntimeError):
        torch.complex(re.to(mojo_gpu), im.to(mojo_gpu).half())


@pytest.mark.parametrize("dtype", CDTYPES)
def test_sum_mean(mojo_gpu: str, dtype: torch.dtype):
    _skip_c128(mojo_gpu, dtype)
    x = _rand(3, 4, 5, dtype=dtype)
    d = x.to(mojo_gpu)
    _close(d.sum(), x.sum())
    _close(d.sum(1), x.sum(1))
    _close(d.sum((0, -1), keepdim=True), x.sum((0, -1), keepdim=True))
    _close(d.mean(), x.mean())
    _close(d.mean(-1), x.mean(-1))
    _close(d[0, 0, 0].sum(), x[0, 0, 0].sum())
    out = torch.empty(0, dtype=dtype, device=mojo_gpu)
    torch.sum(d, 2, out=out)
    _close(out, x.sum(2))


def test_ops_run_natively(mojo_gpu: str):
    native.op_counting(True)
    native.op_counts_reset()
    d = _rand(3).to(mojo_gpu)
    _ = (d * d).abs().sum()
    assert native.op_count("aten::mul.Tensor") == 1
    assert native.op_count("aten::abs") + native.op_count("aten::abs.out") >= 1


def _special_grid() -> tuple[torch.Tensor, torch.Tensor]:
    inf, nan = float("inf"), float("nan")
    vals = [
        complex(a, b)
        for a in (0.0, -0.0, 1.0, inf, -inf, nan)
        for b in (0.0, 3.0, inf, -inf, nan)
    ]
    x = torch.tensor([v for v in vals for _ in vals], dtype=torch.complex64)
    y = torch.tensor([w for _ in vals for w in vals], dtype=torch.complex64)
    return x, y


def _same_values(got: torch.Tensor, want: torch.Tensor):
    """Equal parts, NaN where the reference has NaN (its sign aside)."""
    torch.testing.assert_close(
        torch.view_as_real(got.cpu()),
        torch.view_as_real(want),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "op",
    [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: torch.add(a, b, alpha=2),
        lambda a, b: torch.add(a, b, alpha=1j),
        lambda a, b: torch.rsub(a, b),
        lambda a, b: a - (1 - 2j),
        lambda a, b: 1 - a,
    ],
    ids=["add", "sub", "alpha2", "alpha1j", "rsub", "sub_scalar", "rsub_scalar"],
)
def test_add_sub_alpha_special_values(mojo_gpu: str, op):
    """a + alpha * b with the product always formed (CUDA's AddFunctor and
    CPU's kernel agree on every pair of this grid), so e.g. 1 - inf i is
    NaN - inf i."""
    x, y = _special_grid()
    _same_values(op(x.to(mojo_gpu), y.to(mojo_gpu)), op(x, y))


def test_division_by_a_cpu_scalar_is_a_reciprocal_multiply(mojo_gpu: str):
    """div_true_kernel_cuda turns a CPU-scalar divisor into a multiply by
    its reciprocal, computed once on the host: x / 0j is NaN everywhere on
    CUDA (CPU divides per element and differs)."""
    x, _ = _special_grid()
    got = (x.to(mojo_gpu) / 0j).cpu()
    assert torch.isnan(got.real).all() and torch.isnan(got.imag).all()
    z = torch.randn(64, dtype=torch.complex64)
    _close(z.to(mojo_gpu) / (2 - 1j), z * (1 / torch.tensor(2 - 1j)))


@pytest.mark.parametrize(
    "exponent",
    [0j, 1 + 0j, 2 + 0j, 0.5 + 0j, -1 + 0j, 0.5, -0.5, -1, 2, 0, 1, True, 1.5 - 2j],
)
def test_pow_scalar_exponents(mojo_gpu: str, exponent: complex):
    """Pow.cpp's `Scalar::equal` special cases match complex Scalars too."""
    x = torch.tensor([0j, 1 + 1j, -2 + 0.5j, 3j, 0.25 - 4j], dtype=torch.complex64)
    got = (x.to(mojo_gpu) ** exponent).cpu()
    want = x**exponent
    torch.testing.assert_close(got, want, equal_nan=True, rtol=1e-5, atol=1e-6)


def test_pow_special_cases_on_special_values(mojo_gpu: str):
    """The special cases are taken for exponents EQUAL to 0 / 1 (complex
    ones too): 0j ** 0j is 1 and (inf + 1j) ** (1 + 0j) is a copy (stock
    CUDA's values; the general exp(log(x) * e) gives NaNs there)."""
    inf = float("inf")
    x = torch.tensor([0j, complex(inf, 1)], dtype=torch.complex64).to(mojo_gpu)
    assert (x**0j).cpu().tolist() == [1 + 0j, 1 + 0j]
    assert (x ** (1 + 0j)).cpu().tolist() == [0j, complex(inf, 1)]


def test_pow_scalar_base_one(mojo_gpu: str):
    x = _rand(4)
    assert torch.equal(((1 + 0j) ** x.to(mojo_gpu)).cpu(), (1 + 0j) ** x)


def test_promotion_with_scalars(mojo_gpu: str):
    x = _rand(3)
    d = x.to(mojo_gpu)
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert (d ** (2 + 0j)).dtype == torch.complex64
        assert (d * 1j).dtype == torch.complex64
        assert (torch.ones(3, dtype=torch.float32).to(mojo_gpu) * 1j).dtype == (
            torch.complex64
        )
        assert (
            torch.ones(3, dtype=torch.int64).to(mojo_gpu) * 1j
        ).dtype == torch.complex128
    finally:
        torch.set_default_dtype(old)
    b = torch.ones(3, dtype=torch.bfloat16)
    got = b.to(mojo_gpu) * (1 + 2j)
    assert got.dtype == torch.complex64
    _close(got, b * (1 + 2j))
    with pytest.raises(NotImplementedError):  # complex32 is not held
        torch.ones(3, dtype=torch.half).to(mojo_gpu) * 1j


def test_reduction_out_and_dtype(mojo_gpu: str):
    x = torch.tensor([1e8, 1, -1e8], dtype=torch.complex64)
    out = torch.empty(0, dtype=torch.complex128, device=mojo_gpu)
    torch.sum(x.to(mojo_gpu), 0, out=out)
    assert out.item() == 1  # accumulated in the out's complex128
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
        torch.sum(x.to(mojo_gpu), 0, dtype=torch.complex64, out=out)
    y = _rand(3, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _close(y.to(mojo_gpu).sum(dtype=torch.float32), y.sum(dtype=torch.float32))
        _close(
            y.to(mojo_gpu).mean(0, dtype=torch.float32), y.mean(0, dtype=torch.float32)
        )
        o = torch.empty(0, device=mojo_gpu)
        torch.sum(y.to(mojo_gpu), 0, out=o)
        _close(o, torch.sum(y, 0, out=torch.empty(0)))
    big = torch.empty(1, dtype=torch.complex64, device=mojo_gpu).expand(3)
    got = torch.sum(y.to(mojo_gpu), (0, 1), out=big)
    _close(got, y.sum())


def test_elementwise_out_checks(mojo_gpu: str):
    x = _rand(5)
    with pytest.raises(RuntimeError, match="same device"):
        torch.exp(x.to(mojo_gpu), out=torch.empty(0, dtype=torch.complex64))
    d = x.to(mojo_gpu)
    with pytest.raises(RuntimeError, match="single memory location"):
        torch.add(d[:-1], 1, out=d[1:])
    with pytest.raises(RuntimeError, match="single memory location"):
        torch.exp(d[:-1], out=d[1:])
    e = d.clone()
    torch.exp(e, out=e)
    _close(e, torch.exp(x))
    with pytest.raises(RuntimeError, match="can't be cast"):
        torch.add(d, 1, out=torch.empty(0, device=mojo_gpu))


def test_real_inputs_complex_out(mojo_gpu: str):
    r = torch.rand(4) + 0.5
    d = r.to(mojo_gpu)
    for f in (
        lambda a, o: torch.add(a, a, out=o),
        lambda a, o: torch.mul(a, 2, out=o),
        lambda a, o: torch.div(a, 3, out=o),
        lambda a, o: torch.exp(a, out=o),
        lambda a, o: torch.sqrt(a, out=o),
        lambda a, o: torch.pow(a, 2, out=o),
        lambda a, o: torch.eq(a, a, out=o),
    ):
        o = torch.empty(0, dtype=torch.complex64, device=mojo_gpu)
        f(d, o)
        _close(o, f(r, torch.empty(0, dtype=torch.complex64)))


def test_conj_operand(mojo_gpu: str):
    x, y = _rand(4, 5), _rand(4, 5)
    _close(x.to(mojo_gpu).conj() + y.to(mojo_gpu), x.conj() + y)


def test_scans_and_rsub(mojo_gpu: str):
    x, y = _rand(3, 4), _rand(3, 4)
    d, e = x.to(mojo_gpu), y.to(mojo_gpu)
    _close(torch.rsub(d, e, alpha=2), torch.rsub(x, y, alpha=2))
    _close(1 - d, 1 - x)
    _close(torch.cumsum(d, 1), torch.cumsum(x, 1))


def test_zero_dim_reductions(mojo_gpu: str):
    d = torch.tensor(1 + 2j).to(mojo_gpu)
    for f in (lambda t: t.sum(5), lambda t: t.sum(1), lambda t: t.mean(1)):
        with pytest.raises(IndexError):
            f(d)
    assert d.sum(0).item() == 1 + 2j and d.cumsum(0).item() == 1 + 2j
    with pytest.raises(IndexError):
        d.cumsum(1)


# --- overflow, precision and argument checks ------------------------------


def test_pow_of_a_huge_base_stays_finite(mojo_gpu: str):
    # |z| overflows float32 though z is finite: CUDA (thrust's clog) scales
    x = torch.tensor([3e38 + 3e38j])
    got = (x.to(mojo_gpu) ** 0.1).cpu()
    torch.testing.assert_close(got, torch.tensor([7268.1353 + 572.0147j]))
    got = torch.pow(x.to(mojo_gpu), torch.tensor([0.5 + 0.1j], device=mojo_gpu))
    torch.testing.assert_close(
        got.cpu(), torch.tensor([-1.8861e19 + 2.6145e18j]), rtol=1e-4, atol=0
    )
    # torch.log keeps libc++'s log(abs(z)): +inf on CUDA too
    assert torch.log(x.to(mojo_gpu)).cpu().real.isinf().all()


def test_pow_with_a_cpu_scalar_tensor_exponent(mojo_gpu: str):
    # pow_tensor_tensor_kernel: only `equal(2.0)` (square) is special-cased
    d = torch.tensor([1j], device=mojo_gpu)
    assert torch.equal(torch.pow(d, torch.tensor(2.0)).cpu(), torch.tensor([-1 + 0j]))
    big = torch.tensor([1e20j], device=mojo_gpu)
    got = torch.pow(big, torch.tensor(2)).cpu()
    assert got.real.item() == float("-inf") and got.imag.item() == 0.0
    # no 0 / 1 shortcuts on this path: 0 ** 0. is exp(log 0 * 0) = NaN
    z = torch.tensor([0j, 2 + 1j], device=mojo_gpu)
    got = torch.pow(z, torch.tensor(0.0)).cpu()
    assert got[0].real.isnan() and got[1] == 1


def test_complex_and_polar_validation(mojo_gpu: str):
    ones = torch.ones(3)
    for fn in (torch.complex, torch.polar):
        with pytest.raises(RuntimeError, match="same device"):
            fn(ones.to(mojo_gpu), ones)
        with pytest.raises(RuntimeError, match="same device"):
            fn(ones.to(mojo_gpu), torch.tensor(1.0))
        with pytest.raises(RuntimeError, match="same device"):
            fn(torch.tensor(1.0), ones.to(mojo_gpu))
    b = torch.ones(3, dtype=torch.bfloat16, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="Half, Float or Double"):
        torch.complex(b, b)
    h = torch.ones(3, dtype=torch.half, device=mojo_gpu)
    with pytest.raises(NotImplementedError):
        torch.complex(h, h)  # complex32: declined, never garbage


def test_cumsum_dtypes_and_outs(mojo_gpu: str):
    x = _rand(4)
    d = x.to(mojo_gpu)
    out = torch.empty(0, dtype=torch.complex128, device=mojo_gpu)
    torch.cumsum(d, 0, out=out)
    assert out.dtype == torch.complex128
    _close(out, torch.cumsum(x.to(torch.complex128), 0))
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
        torch.cumsum(
            d,
            0,
            dtype=torch.complex128,
            out=torch.empty(0, dtype=torch.complex64, device=mojo_gpu),
        )
    y = torch.tensor([1 + 1j, 2 + 2j])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _close(
            torch.cumsum(y.to(mojo_gpu), 0, dtype=torch.float32),
            torch.tensor([1.0, 3.0]),
        )
        rout = torch.empty(0, device=mojo_gpu)
        torch.cumsum(y.to(mojo_gpu), 0, out=rout)
        _close(rout, torch.tensor([1.0, 3.0]))
    # an expanded out is resized before any overlap check
    ex = torch.empty(1, dtype=torch.complex64, device=mojo_gpu).expand(5)
    torch.cumsum(d, 0, out=ex)
    _close(ex, torch.cumsum(x, 0))
    # a real self with a complex dtype
    got = torch.ones(3, device=mojo_gpu).cumsum(0, dtype=torch.cfloat)
    _close(got, torch.ones(3).cumsum(0, dtype=torch.cfloat))


def test_same_dtype_unary_outs(mojo_gpu: str):
    d = _rand(4).to(mojo_gpu)
    for fn in (torch.neg, torch.sgn):
        with pytest.raises(RuntimeError, match="Found dtype"):
            fn(d, out=torch.empty(0, dtype=torch.complex128, device=mojo_gpu))
    with pytest.raises(RuntimeError, match="Found dtype"):
        torch.neg(
            torch.ones(3, device=mojo_gpu),
            out=torch.empty(0, dtype=torch.complex64, device=mojo_gpu),
        )
    r = torch.tensor([-1.0, 2.0])
    out = torch.empty(0, dtype=torch.complex64, device=mojo_gpu)
    torch.angle(r.to(mojo_gpu), out=out)
    _close(out, torch.angle(r).to(torch.complex64))


def test_reduction_dims_and_routing(mojo_gpu: str):
    x = _rand(3, 4)
    d = x.to(mojo_gpu)
    with pytest.raises(RuntimeError, match="multiple times"):
        d.sum((0, 0))
    with pytest.raises(RuntimeError, match="multiple times"):
        d[0].mean((0, -1))  # the same dim of a 1-d tensor
    with pytest.raises(RuntimeError, match="multiple times"):
        torch.tensor(1 + 2j, device=mojo_gpu).sum((0, 0))
    empty = torch.empty(0, dtype=torch.complex64, device=mojo_gpu).mean().cpu()
    assert empty.real.isnan() and empty.imag == 0
    _close(
        torch.ones(3, device=mojo_gpu).sum(dtype=torch.cfloat),
        torch.ones(3).sum(dtype=torch.cfloat),
    )
    _close(
        torch.arange(4, device=mojo_gpu).mean(dtype=torch.cfloat),
        torch.arange(4).mean(dtype=torch.cfloat),
    )
    r = torch.randn(3, 2)
    for fn in (torch.sum, torch.mean):
        out = torch.empty(0, dtype=torch.complex64, device=mojo_gpu)
        fn(r.to(mojo_gpu), 0, out=out)
        _close(out, fn(r, 0).to(torch.complex64))


def test_arithmetic_argument_checks(mojo_gpu: str):
    b = torch.tensor([True, False], device=mojo_gpu)
    with pytest.raises(RuntimeError, match="Subtraction"):
        _ = b - 1j
    with pytest.raises(RuntimeError, match="Subtraction"):
        _ = 1j - b
    d = _rand(4).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="Boolean alpha"):
        torch.add(d, d, alpha=True)
    z = _rand(4)
    _close(torch.pow(True, z.to(mojo_gpu)), torch.pow(True, z))


# --- fused multiply-adds, rounding and out dtypes ----------------------------


def _cuda_cmul64(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """complex64 a * b as stock CUDA computes it (nvcc's contraction of
    c10's operator*, measured bit-exact on 200k pairs): re = fma(a, c,
    -(b d)), im = fma(a, d, b c), each fma in float64 then rounded."""
    ar, ai = a.real.double(), a.imag.double()
    br, bi = b.real.double(), b.imag.double()
    bd = (ai * bi).float().double()
    bc = (ai * br).float().double()
    re = (ar * br - bd).float()
    im = (ar * bi + bc).float()
    return torch.complex(re, im)


def test_mul_overflow_matches_cuda(mojo_gpu: str):
    # recorded on stock CUDA (torch 2.11, H100)
    a = torch.tensor([1e30 + 1e30j, 1e20 + 1e20j])
    b = torch.tensor([1e30 - 1e30j, 1e20 - 1e20j])
    got = (a.to(mojo_gpu) * b.to(mojo_gpu)).cpu()
    assert got.tolist() == [complex(float("inf"), float("inf"))] * 2
    vals = [1e20, -1e20, 1e30, -1e30, 3e38, -3e38, 2.5, -7e-3]
    zs = torch.tensor([complex(x, y) for x in vals for y in vals])
    x = zs.repeat_interleave(len(zs))
    y = zs.repeat(len(zs))
    got = torch.view_as_real((x.to(mojo_gpu) * y.to(mojo_gpu)).cpu())
    want = torch.view_as_real(_cuda_cmul64(x, y))
    # which parts overflow, and to which sign, is the contraction's doing;
    # the finite parts agree up to the model's double rounding
    assert torch.equal(got.isinf(), want.isinf())
    assert torch.equal(got[got.isinf()], want[want.isinf()])
    fin = got.isfinite()
    torch.testing.assert_close(got[fin], want[fin], rtol=1e-6, atol=0)


def test_pow_cpu_exponent_is_rounded_first(mojo_gpu: str):
    z = torch.tensor([1e20j], device=mojo_gpu)
    got = (z ** torch.tensor(2.00000001, dtype=torch.float64)).cpu()
    assert got.real.item() == float("-inf") and got.imag.item() == 0.0


def test_div_rounding_mode_into_a_complex_out(mojo_gpu: str):
    out = torch.empty(0, dtype=torch.complex64, device=mojo_gpu)
    torch.div(
        torch.tensor([-3.0, 3.0], device=mojo_gpu), 2, rounding_mode="floor", out=out
    )
    assert out.cpu().tolist() == [-2 + 0j, 1 + 0j]


def test_mean_out_dtype_checks(mojo_gpu: str):
    cout = torch.empty(0, dtype=torch.complex64, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="could not infer output dtype"):
        torch.mean(torch.tensor([1, 2, 3], device=mojo_gpu), 0, out=cout)
    with pytest.raises(RuntimeError, match="can't be cast"):
        torch.ops.aten.mean.dtype_out(
            torch.tensor([1 + 2j], device=mojo_gpu), out=torch.empty(0, device=mojo_gpu)
        )


def test_abs_out_dtypes(mojo_gpu: str):
    z = torch.tensor([3 + 4j], device=mojo_gpu)
    with pytest.raises(
        RuntimeError, match="Found dtype ComplexDouble but expected ComplexFloat"
    ):
        torch.abs(z, out=torch.empty(0, dtype=torch.complex128, device=mojo_gpu))
    with pytest.raises(
        RuntimeError, match="Found dtype ComplexFloat but expected Float"
    ):
        torch.abs(
            torch.tensor([-1.0], device=mojo_gpu),
            out=torch.empty(0, dtype=torch.complex64, device=mojo_gpu),
        )
    out = torch.empty(0, dtype=torch.complex64, device=mojo_gpu)
    torch.abs(z, out=out)
    assert out.cpu().tolist() == [5 + 0j]


def test_pow_of_one_fills_any_out(mojo_gpu: str):
    out = torch.empty(0, device=mojo_gpu)
    torch.pow(1, torch.tensor([1 + 2j, 3j], device=mojo_gpu), out=out)
    assert out.cpu().tolist() == [1.0, 1.0]
    with pytest.raises(RuntimeError, match="can't be cast"):
        torch.pow(2, torch.tensor([1 + 2j], device=mojo_gpu), out=out)


# --- thrust's pow, Scalar conversion, real outs -----------------------------


def test_pow_follows_thrust_log_and_exp(mojo_gpu: str):
    # recorded on stock CUDA (torch 2.11+cu130): thrust's clogf takes
    # log1p(1)/2 for |re| == 1, so the phase of y * log(x) matches
    x = torch.tensor([1 + 1j, 2.5 + 0.9999j, 1 + 0.5j])
    y = torch.tensor([-1e20j, -1j, 1 + 0j])
    got = (x.to(mojo_gpu) ** y.to(mojo_gpu)).cpu()
    assert got[0].real.item() == float("-inf") and got[0].imag.item() == float("-inf")
    assert got[1].item() == complex(0.8021237254142761, -1.223475694656372)
    assert got[2].item() == complex(1.0, 0.5000000596046448)


def test_scalar_conversion_overflow(mojo_gpu: str):
    z = _rand(3).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="without overflow"):
        torch.add(z, z, alpha=1e100)
    with pytest.raises(RuntimeError, match="without overflow"):
        torch.sub(z, z, alpha=1e100)
    with pytest.raises(RuntimeError, match="without overflow"):
        z**1e100
    # complex128 holds any double; a wrapped operand is not checked
    w = z.to(torch.complex128)
    assert torch.add(w, w, alpha=1e100).cpu().real.abs().max() > 1e99
    got = (torch.tensor([1 + 1j], device=mojo_gpu) + 1e100).cpu()
    assert got.real.item() == float("inf") and got.imag.isnan()


def test_pow_scalar_base_resizes_its_out_first(mojo_gpu: str):
    x = _rand(2)
    out = torch.empty(1, dtype=torch.complex64, device=mojo_gpu).expand(3)
    torch.pow(2, x.to(mojo_gpu), out=out)
    assert out.shape == (2,)
    _close(out, torch.pow(2, x))


@pytest.mark.parametrize("dtype", CDTYPES)
def test_signed_zeros_and_reciprocal_specials(mojo_gpu: str, dtype: torch.dtype):
    _skip_c128(mojo_gpu, dtype)
    z = torch.tensor([complex(1.0, -0.0), complex(0.0, -0.0)], dtype=dtype)
    got = torch.sqrt(z.to(mojo_gpu)).cpu()
    assert torch.equal(
        torch.view_as_real(got).signbit(), torch.view_as_real(torch.sqrt(z)).signbit()
    )
    inf = float("inf")
    r = torch.tensor(
        [
            complex(inf, 1),
            complex(1, -inf),
            complex(inf, inf),
            complex(float("nan"), 0),
            2j,
        ],
        dtype=dtype,
    )
    got = torch.reciprocal(r.to(mojo_gpu)).cpu()
    assert got[:2].tolist() == [0j, 0j]
    assert got[2:4].real.isnan().all() and got[2:4].imag.isnan().all()
    assert got[4].item() == -0.5j


def test_alpha_and_in_place_dtype_errors(mojo_gpu: str):
    r = torch.ones(2, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="argument alpha must not be a complex"):
        torch.add(r, r, alpha=1j)
    with pytest.raises(RuntimeError, match="Bad in-place call"):
        _rand(3).to(mojo_gpu).cumsum_(0, dtype=torch.complex128)


def test_outs_of_real_results(mojo_gpu: str):
    z = _rand(2, 3)
    d = z.to(mojo_gpu)
    # abs / angle of a complex self: resize then copy_ (any device)
    cpu_out = torch.empty(0)
    torch.abs(d, out=cpu_out)
    _close(cpu_out, torch.abs(z))
    ex = torch.empty(1, device=mojo_gpu).expand(7)
    torch.angle(d, out=ex)
    _close(ex, torch.angle(z))
    with pytest.raises(RuntimeError, match="can't be cast"):
        torch.abs(d, out=torch.empty(0, dtype=torch.int64, device=mojo_gpu))
    c128 = torch.empty(0, dtype=torch.complex128, device=mojo_gpu)
    torch.angle(d, out=c128)
    _close(c128, torch.angle(z).to(torch.complex128))
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype bool"):
        torch.ops.aten.isnan.out(d, out=torch.empty(0, device=mojo_gpu))
    # rsub's generated out: exact dtype, no overlap check with the inputs
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
        torch.ops.aten.rsub.Tensor_out(d, d, out=c128)
    # a scan's expanded out is refused, a reduction's is written
    with pytest.raises(RuntimeError, match="unsupported operation"):
        torch.cumsum(
            d,
            1,
            out=torch.empty(1, dtype=torch.complex64, device=mojo_gpu).expand(2, 3),
        )


# --- mean dtypes, out casts, broadcast messages, notices ----------------------


def test_mean_with_a_non_floating_dtype_raises(mojo_gpu: str):
    z = _rand(2, 3).to(mojo_gpu)
    for dt in (torch.int64, torch.bool, torch.int8, torch.int32):
        with pytest.raises(RuntimeError, match="Optional dtype must be either"):
            z.mean(dtype=dt)
        with pytest.raises(RuntimeError, match="Optional dtype must be either"):
            z.mean(0, dtype=dt)


def test_pow_scalar_base_out_resize_and_bool_base(mojo_gpu: str):
    out = torch.empty(1, dtype=torch.complex64, device=mojo_gpu).expand(3)
    torch.pow(2, torch.tensor([1.0, 2.0], device=mojo_gpu), out=out)
    assert out.cpu().tolist() == [2 + 0j, 4 + 0j]
    # True is not Scalar::equal(1.0): the power is computed
    got = torch.pow(True, torch.tensor([float("inf") + 0j], device=mojo_gpu)).cpu()
    assert got[0].real.isnan() and got[0].imag.isnan()


def test_out_cast_is_checked_before_the_resize(mojo_gpu: str):
    z = _rand(2).to(mojo_gpu)
    for op in (torch.add, torch.mul, torch.pow):
        out = torch.zeros(5, device=mojo_gpu)
        with pytest.raises(RuntimeError, match="can't be cast"):
            op(z, z, out=out)
        assert out.shape == (5,)


def test_broadcast_error_names_the_sizes(mojo_gpu: str):
    with pytest.raises(
        RuntimeError,
        match=r"tensor a \(3\) must match the size of tensor b \(2\) at non-singleton dimension 2",
    ):
        torch.ones(2, 3, device=mojo_gpu) * torch.ones(
            4, 1, 2, dtype=torch.complex64, device=mojo_gpu
        )
    with pytest.raises(
        RuntimeError,
        match=r"tensor a \(2\) must match the size of tensor b \(3\) at non-singleton dimension 0",
    ):
        torch.complex(torch.ones(2, device=mojo_gpu), torch.ones(3, device=mojo_gpu))


def test_third_party_notices_cover_the_ports():
    root = pathlib.Path(__file__).resolve().parents[2]
    notices = (root / "THIRD_PARTY_NOTICES.txt").read_text()
    for needle in (
        "David Schultz",
        "Stephen Montgomery-Smith",
        "NVIDIA Corporation",
        "LLVM Exceptions",
        "Apache License",
    ):
        assert needle in notices
    src = (
        root / "torch_mojo_backend/mojo/tmb/kernels/common/complex_math.mojo"
    ).read_text()
    assert "SPDX-License-Identifier: BSD-2-Clause" in src
    assert "SPDX-License-Identifier: Apache-2.0 WITH" in src


# --- host-side scalar math, packaging -----------------------------------------


def test_host_scalar_reciprocal_is_taken_in_double(mojo_gpu: str):
    got = (torch.tensor([1 + 1j], device=mojo_gpu) / 1e-40).cpu()
    assert got.real.item() == float("inf") and got.imag.item() == float("inf")


def test_notices_ship_with_the_package():
    root = pathlib.Path(__file__).resolve().parents[2]
    pyproject = (root / "pyproject.toml").read_text()
    assert '"THIRD_PARTY_NOTICES.txt"' in pyproject
    notices = (root / "THIRD_PARTY_NOTICES.txt").read_text()
    assert "From PyTorch:" in notices and "Neither the names of Facebook" in notices


# --- complex reductions and scans into bool / integer dtypes are declined -----


@pytest.mark.parametrize(
    "out_dtype", [torch.bool, torch.int64, torch.int32, torch.int8]
)
def test_non_floating_reduction_outs_are_declined_untouched(
    mojo_gpu: str, out_dtype: torch.dtype
):
    x = torch.ones(2, 3, dtype=torch.complex64, device=mojo_gpu)
    for call in (
        lambda o: torch.sum(x, 1, out=o),
        lambda o: torch.mean(x, 1, out=o),
        lambda o: torch.cumsum(x, 1, out=o),
        lambda o: torch.mean(
            torch.empty(3, 0, dtype=torch.complex64, device=mojo_gpu), 0, out=o
        ),
        lambda o: torch.cumsum(torch.tensor(1 + 0j, device=mojo_gpu), 0, out=o),
    ):
        out = torch.zeros(7, dtype=out_dtype, device=mojo_gpu)
        with pytest.raises(NotImplementedError):
            call(out)
        assert out.shape == (7,)  # declined before the out is touched
    with pytest.raises(NotImplementedError):
        x.sum(dtype=out_dtype)
    with pytest.raises(NotImplementedError):
        x.cumsum(0, dtype=out_dtype)


def test_declines_come_after_cudas_own_checks(mojo_gpu: str):
    e = torch.empty(0, dtype=torch.complex64, device=mojo_gpu)
    bout = torch.zeros(7, dtype=torch.bool, device=mojo_gpu)
    with pytest.raises(IndexError, match="Dimension out of range"):
        torch.cumsum(e, 9, out=bout)
    cout = torch.zeros(5, dtype=torch.complex64, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype bool"):
        torch.cumsum(
            torch.ones(3, dtype=torch.complex64, device=mojo_gpu),
            0,
            dtype=torch.bool,
            out=cout,
        )
    assert cout.shape == (5,)
    x = torch.ones(2, 3, dtype=torch.complex64, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="Optional dtype must be either"):
        torch.mean(x, 1, dtype=torch.bool, out=bout)
    assert bout.shape == (7,)


# --- host operands with device outs, and the order of the reduction checks -----


def test_abs_angle_of_a_host_tensor_into_a_device_out(mojo_gpu: str):
    x = torch.tensor([3 + 4j, -1 + 0j])
    for fn in (torch.abs, torch.angle):
        for src in (x, x[:1], x[0]):
            out = torch.empty(0, device=mojo_gpu)
            fn(src, out=out)
            ref = fn(src)
            assert out.device.type == "mojo" and torch.equal(out.cpu(), ref)
        with pytest.raises(RuntimeError, match="same device"):
            fn(x, out=torch.empty(0, dtype=torch.complex64, device=mojo_gpu))
    # abs: the out's dtype is checked before the devices
    with pytest.raises(
        RuntimeError, match="Found dtype ComplexDouble but expected ComplexFloat"
    ):
        torch.abs(x, out=torch.empty(0, dtype=torch.complex128, device=mojo_gpu))


def test_binary_outs_with_only_host_operands(mojo_gpu: str):
    x = torch.tensor([1 + 1j, 2j, 3 + 0j])
    out = torch.empty(0, dtype=torch.complex64, device=mojo_gpu)
    for op in (torch.add, torch.mul, torch.div, torch.pow):
        with pytest.raises(RuntimeError, match="same device"):
            op(x, x, out=out)
        with pytest.raises(RuntimeError, match="too many CPU scalars"):
            op(x[0], x[0], out=out)
        # a host 0-d operand with a device one is a scalar
        got = op(
            x[0],
            x[0].to(mojo_gpu),
            out=torch.empty(0, dtype=torch.complex64, device=mojo_gpu),
        )
        _close(got, op(x[0], x[0]))


def test_reduction_dim_is_checked_before_the_out_dtype(mojo_gpu: str):
    x = torch.ones(2, 3, dtype=torch.complex64, device=mojo_gpu)
    out = torch.empty(0, dtype=torch.complex64, device=mojo_gpu)
    for fn in (torch.sum, torch.mean):
        with pytest.raises(IndexError, match="Dimension out of range"):
            fn(x, 9, dtype=torch.float32, out=out)
    with pytest.raises(RuntimeError, match="Optional dtype must be either"):
        torch.mean(x, 9, dtype=torch.bool)


# --- integer scalars meet complex arithmetic in one rounding --------------------


@pytest.mark.parametrize(
    "v", [2**53 + 1, 2**62 + 2**38 + 1, 2**63 - 1, -(2**62 + 2**38 + 1)]
)
@pytest.mark.parametrize("dtype", CDTYPES)
def test_integer_scalars_round_once(mojo_gpu: str, v: int, dtype: torch.dtype):
    _skip_c128(mojo_gpu, dtype)
    one = torch.ones(2, dtype=dtype)
    d = one.to(mojo_gpu)
    for op in (torch.add, torch.sub, torch.mul):
        want = op(one.cuda() if torch.cuda.is_available() else one, v).cpu()
        assert torch.equal(op(d, v).cpu(), want)
        assert torch.equal(op(d, torch.tensor(v)).cpu(), want)
    want = torch.add(one, one, alpha=v)
    assert torch.equal(torch.add(d, d, alpha=v).cpu(), want)
