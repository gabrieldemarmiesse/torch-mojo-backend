"""The scans group of the native mojo backend (tmb/ops/scans.mojo): cumsum,
cumprod, logcumsumexp and cummax / cummin, through public torch APIs,
compared against CPU torch."""

import pytest
import torch

from tests.native.conftest import ran, skip_if_metal

SHAPES_DIMS = [
    ((7,), 0),
    ((5, 300), 1),
    ((5, 300), 0),
    ((3, 4, 5), 1),
    ((3, 4, 5), 0),
    ((2, 3, 70), -1),
    ((2, 1, 4), 1),
]


def _float_tol(dtype: torch.dtype) -> tuple[float | None, float | None]:
    if dtype in (torch.float16, torch.bfloat16):
        return 2e-2, 2e-2
    if dtype == torch.float32:
        # The block scans add in a tree, CPU in sequence: a few ulp of the
        # running sum, which near a cancellation is large relative to it.
        return 1e-4, 1e-5
    return None, None


@pytest.mark.parametrize(("shape", "dim"), SHAPES_DIMS)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.int32]
)
def test_cumsum_every_dim_matches_cpu(mojo_gpu, shape, dim, dtype):
    x = (torch.randn(shape) * 4).to(dtype)
    with ran("aten::cumsum"):
        got = torch.cumsum(x.to(mojo_gpu), dim)
    # float32 accumulation, rounded once: the float32 CPU result (half
    # dtypes against the float32 scan of their values).
    ref = torch.cumsum(x.float() if dtype.is_floating_point else x, dim).to(got.dtype)
    atol, rtol = _float_tol(dtype)
    torch.testing.assert_close(got.cpu(), ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize(("shape", "dim"), SHAPES_DIMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64])
def test_cumprod_matches_cpu(mojo_gpu, shape, dim, dtype):
    x = (torch.rand(shape) * 0.4 + 0.8).to(dtype)
    if dtype == torch.int64:
        x = torch.randint(-2, 3, shape)
    with ran("aten::cumprod"):
        got = torch.cumprod(x.to(mojo_gpu), dim)
    ref = torch.cumprod(x.float() if dtype.is_floating_point else x, dim).to(got.dtype)
    atol, rtol = _float_tol(dtype)
    torch.testing.assert_close(got.cpu(), ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("op", [torch.cumsum, torch.cumprod])
@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.int8, torch.int16])
def test_cum_integer_inputs_promote_to_int64(mojo_gpu, op, dtype):
    x = torch.randint(0, 2, (4, 6)).to(dtype)
    got = op(x.to(mojo_gpu), 1)
    assert got.dtype == torch.int64
    torch.testing.assert_close(got.cpu(), op(x, 1))


@pytest.mark.parametrize("op", [torch.cumsum, torch.cumprod])
def test_cum_dtype_kwarg_casts_first(mojo_gpu, op):
    x = torch.tensor([[1.7, 2.2, -0.6], [0.5, 1.5, 2.5]])
    got = op(x.to(mojo_gpu), 1, dtype=torch.int64)
    torch.testing.assert_close(got.cpu(), op(x, 1, dtype=torch.int64))
    got = op(x.to(mojo_gpu), 0, dtype=torch.float16)
    torch.testing.assert_close(got.cpu(), op(x, 0, dtype=torch.float16))


@pytest.mark.parametrize("op", [torch.cumsum, torch.cumprod])
def test_cum_out_and_inplace(mojo_gpu, op):
    x = torch.randn(4, 5)
    out = torch.empty(0, device=mojo_gpu)
    op(x.to(mojo_gpu), 0, out=out)
    torch.testing.assert_close(out.cpu(), op(x, 0))
    # A non-contiguous out of the right shape is written where it lives.
    base = torch.zeros(5, 4, device=mojo_gpu)
    op(x.to(mojo_gpu), 1, out=base.t())
    torch.testing.assert_close(base.t().cpu(), op(x, 1))
    # The out's dtype is the compute dtype when dtype= is absent.
    out_d = torch.empty(4, 5, dtype=torch.float64, device=mojo_gpu)
    skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    op(x.to(mojo_gpu), 1, out=out_d)
    torch.testing.assert_close(out_d.cpu(), op(x.double(), 1))


@pytest.mark.parametrize("op", ["cumsum_", "cumprod_"])
def test_cum_inplace_method(mojo_gpu, op):
    x = torch.randn(3, 6)
    y = x.to(mojo_gpu)
    r = getattr(y, op)(1)
    assert r.data_ptr() == y.data_ptr()
    torch.testing.assert_close(y.cpu(), getattr(x.clone(), op)(1))
    with pytest.raises(RuntimeError, match="Bad in-place call"):
        getattr(y, op)(1, dtype=torch.float16)


@pytest.mark.parametrize("op", [torch.cumsum, torch.cumprod])
def test_cum_zero_dim_and_empty(mojo_gpu, op):
    s = torch.tensor(3.5)
    torch.testing.assert_close(op(s.to(mojo_gpu), 0).cpu(), op(s, 0))
    torch.testing.assert_close(op(s.to(mojo_gpu), -1).cpu(), op(s, -1))
    e = torch.empty(0, 3)
    assert op(e.to(mojo_gpu), 1).shape == (0, 3)
    with pytest.raises(IndexError):
        op(torch.randn(2, 3, device=mojo_gpu), 2)


@pytest.mark.parametrize("op", [torch.cumsum, torch.cumprod])
def test_cum_noncontiguous_input(mojo_gpu, op):
    x = torch.randn(6, 5)
    xm = x.to(mojo_gpu)
    torch.testing.assert_close(op(xm.t(), 1).cpu(), op(x.t(), 1))
    torch.testing.assert_close(op(xm[:, ::2], 0).cpu(), op(x[:, ::2], 0))


def test_cum_nonfinite_propagates(mojo_gpu):
    x = torch.tensor([[1.0, float("nan"), 2.0], [float("inf"), -1.0, 3.0]])
    for op in (torch.cumsum, torch.cumprod):
        for d in (0, 1):
            torch.testing.assert_close(
                op(x.to(mojo_gpu), d).cpu(), op(x, d), equal_nan=True
            )


def test_cum_long_rows(mojo_gpu):
    """Rows longer than one 256-wide tile take the block-scan route."""
    x = torch.rand(3, 1000) * 0.02 + 0.995
    torch.testing.assert_close(
        torch.cumprod(x.to(mojo_gpu), 1).cpu(),
        torch.cumprod(x, 1),
        rtol=1e-4,
        atol=1e-5,
    )
    xi = torch.randint(-50, 50, (2, 3000))
    torch.testing.assert_close(
        torch.cumsum(xi.to(mojo_gpu), 1).cpu(), torch.cumsum(xi, 1)
    )


@pytest.mark.parametrize(("shape", "dim"), SHAPES_DIMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_logcumsumexp_matches_cpu(mojo_gpu, shape, dim, dtype):
    x = (torch.randn(shape) * 3).to(dtype)
    with ran("aten::_logcumsumexp"):
        got = torch.logcumsumexp(x.to(mojo_gpu), dim)
    ref = torch.logcumsumexp(x.double(), dim)
    atol, rtol = (1e-5, 1e-5) if dtype == torch.float32 else (3e-2, 2e-2)
    torch.testing.assert_close(got.cpu().double(), ref, atol=atol, rtol=rtol)


def test_logcumsumexp_nonfinite(mojo_gpu):
    inf = float("inf")
    x = torch.tensor(
        [
            [-inf, -inf, 1.0, -inf],
            [inf, inf, 1.0, -inf],
            [0.0, float("nan"), 1.0, 2.0],
            [inf, -inf, 3.0, inf],
        ]
    )
    torch.testing.assert_close(
        torch.logcumsumexp(x.to(mojo_gpu), 1).cpu(),
        torch.logcumsumexp(x, 1),
        equal_nan=True,
    )


def test_logcumsumexp_out_scalar_empty_and_errors(mojo_gpu):
    x = torch.randn(3, 4)
    out = torch.empty(0, device=mojo_gpu)
    torch.logcumsumexp(x.to(mojo_gpu), 1, out=out)
    torch.testing.assert_close(out.cpu(), torch.logcumsumexp(x, 1))
    s = torch.tensor(-2.0)
    torch.testing.assert_close(torch.logcumsumexp(s.to(mojo_gpu), 0).cpu(), s)
    assert torch.logcumsumexp(torch.empty(0, 2, device=mojo_gpu), 0).shape == (0, 2)
    with pytest.raises(RuntimeError, match="not implemented for 'Long'"):
        torch.logcumsumexp(torch.arange(4, device=mojo_gpu), 0)


@pytest.mark.parametrize("op", [torch.cummax, torch.cummin])
@pytest.mark.parametrize(("shape", "dim"), SHAPES_DIMS)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int64,
        torch.int32,
        torch.bool,
        torch.uint8,
        torch.int8,
    ],
)
def test_cummax_cummin_match_cpu(mojo_gpu, op, shape, dim, dtype):
    # Few distinct values, so ties (the LAST index wins) are common.
    x = torch.randint(-3, 4, shape).to(dtype)
    name = "aten::_cummax_helper" if op is torch.cummax else "aten::_cummin_helper"
    with ran(name):
        values, indices = op(x.to(mojo_gpu), dim)
    ref_v, ref_i = op(x, dim)
    torch.testing.assert_close(values.cpu(), ref_v)
    torch.testing.assert_close(indices.cpu(), ref_i)


@pytest.mark.parametrize("op", [torch.cummax, torch.cummin])
def test_cummax_cummin_nan_sticks(mojo_gpu, op):
    nan = float("nan")
    x = torch.tensor([[1.0, nan, 3.0, nan, -1.0], [2.0, 2.0, -5.0, 9.0, 9.0]])
    x = torch.cat([x, torch.randn(2, 300)], 1)
    values, indices = op(x.to(mojo_gpu), 1)
    ref_v, ref_i = op(x, 1)
    torch.testing.assert_close(values.cpu(), ref_v, equal_nan=True)
    torch.testing.assert_close(indices.cpu(), ref_i)


@pytest.mark.parametrize("op", [torch.cummax, torch.cummin])
def test_cummax_cummin_out_and_noncontiguous(mojo_gpu, op):
    x = torch.randn(5, 4)
    # Right-shaped outs: ATen's composite cummax_out resizes through
    # aten::resize_, which the mojo device does not implement.
    v = torch.empty(4, 5, device=mojo_gpu)
    i = torch.empty(5, 4, dtype=torch.int64, device=mojo_gpu).t()
    op(x.to(mojo_gpu).t(), 1, out=(v, i))
    ref_v, ref_i = op(x.t(), 1)
    torch.testing.assert_close(v.cpu(), ref_v)
    torch.testing.assert_close(i.cpu(), ref_i)
    s = torch.tensor(2.0)
    rv, ri = op(s.to(mojo_gpu), 0)
    assert rv.item() == 2.0 and ri.item() == 0
