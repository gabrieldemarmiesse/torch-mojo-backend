"""Pairwise distances on the native mojo device (tmb/ops/distance.mojo):
`_cdist_forward`, `_cdist_backward`, `_pdist_forward`, `_pdist_backward`,
reached through the public `torch.cdist` / `F.pdist` and their autograd, and
glu's forward-mode derivatives `glu_jvp` / `glu_backward_jvp`
(tmb/ops/pointwise.mojo).

Everything is compared with CPU torch through the public API. CUDA's
distance kernels dispatch float and double only, and so do ours; the half
types reach a distance only through the matrix-multiply route (p = 2).
"""

import pytest
import torch
import torch.nn.functional as F
from torch.func import jvp

from tests.native.conftest import ran, skip_if_metal

aten = torch.ops.aten

P_VALUES = [0.0, 1.0, 2.0, 3.0, 0.5, 1.5, 2.5, float("inf")]
DIST_DTYPES = [torch.float32, torch.float64]
# (x1 shape, x2 shape): plain, batched, broadcast batch dims, wide rows.
CDIST_SHAPES = [
    ((5, 5, 2), (5, 6, 2)),
    ((3, 5), (4, 5)),
    ((2, 1, 3, 4), (1, 2, 5, 4)),
    ((1, 1, 3), (2, 1, 3, 3)),
    ((3, 357), (7, 357)),
]


def _tol(dtype: torch.dtype, p: float) -> tuple[float, float]:
    """(rtol, atol)."""
    if dtype == torch.float64:
        return 1e-11, 1e-12
    # |diff|^(p - 1) for p < 1 is ill-conditioned near zero differences: a
    # sum of large terms of both signs, where the CPU's float32 gradient
    # is measured further from the float64 one than ours.
    if p < 1:
        return 1e-2, 1e-3
    return 1e-4, 1e-5


def _dtype_or_skip(device: str, dtype: torch.dtype):
    if dtype == torch.float64:
        skip_if_metal(device, "Apple GPUs have no float64")


@pytest.mark.parametrize("dtype", DIST_DTYPES)
@pytest.mark.parametrize("p", P_VALUES)
@pytest.mark.parametrize("shapes", CDIST_SHAPES, ids=str)
def test_cdist(mojo_gpu, dtype, p, shapes):
    _dtype_or_skip(mojo_gpu, dtype)
    torch.manual_seed(0)
    a = torch.randn(shapes[0], dtype=dtype, requires_grad=True)
    b = torch.randn(shapes[1], dtype=dtype, requires_grad=True)
    want = torch.cdist(a, b, p, "donot_use_mm_for_euclid_dist")
    grad = torch.randn_like(want)
    wa, wb = torch.autograd.grad(want, (a, b), grad)
    am = a.detach().to(mojo_gpu).requires_grad_()
    bm = b.detach().to(mojo_gpu).requires_grad_()
    with ran("aten::_cdist_forward"):
        got = torch.cdist(am, bm, p, "donot_use_mm_for_euclid_dist")
    with ran("aten::_cdist_backward"):
        ga, gb = torch.autograd.grad(got, (am, bm), grad.to(mojo_gpu))
    rtol, atol = _tol(dtype, p)
    torch.testing.assert_close(got.cpu(), want.detach(), rtol=rtol, atol=atol)
    torch.testing.assert_close(ga.cpu(), wa, rtol=rtol, atol=atol)
    torch.testing.assert_close(gb.cpu(), wb, rtol=rtol, atol=atol)


def test_cdist_nan_and_ties(mojo_gpu):
    """CUDA's `zero` norm returns the NaN difference itself, its `inf` norm
    skips a NaN (`diff > agg` is false); the inf backward splits the
    gradient over tied maxima."""
    a = torch.tensor([[1.0, float("nan"), 3.0], [0.0, 0.0, 0.0]])
    b = torch.tensor([[1.0, 2.0, 3.0], [2.0, -2.0, 1.0]])
    for p in (0.0, 1.0, 2.0, float("inf"), 3.0):
        want = torch.cdist(a, b, p)
        got = torch.cdist(a.to(mojo_gpu), b.to(mojo_gpu), p)
        torch.testing.assert_close(got.cpu(), want, equal_nan=True)
    x = torch.tensor([[0.0, 0.0]], requires_grad=True)
    y = torch.tensor([[1.0, -1.0], [0.0, 0.0]])
    (want,) = torch.autograd.grad(torch.cdist(x, y, float("inf")).sum(), x)
    xm = x.detach().to(mojo_gpu).requires_grad_()
    (got,) = torch.autograd.grad(
        torch.cdist(xm, y.to(mojo_gpu), float("inf")).sum(), xm
    )
    torch.testing.assert_close(got.cpu(), want)


@pytest.mark.parametrize("mode", [None, 1])
def test_cdist_forward_mm_route(mojo_gpu, mode):
    """p = 2 with more than 25 rows (or compute_mode 1): `_cdist_forward`
    runs ATen's `_euclidean_dist` composite, on this device's matmul."""
    torch.manual_seed(1)
    a = torch.randn(2, 30, 6)
    b = torch.randn(1, 7, 6)
    want = aten._cdist_forward(a, b, 2.0, mode)
    with ran("aten::_cdist_forward"):
        got = aten._cdist_forward(a.to(mojo_gpu), b.to(mojo_gpu), 2.0, mode)
    assert got.shape == want.shape
    torch.testing.assert_close(got.cpu(), want, rtol=1e-4, atol=1e-4)


def test_cdist_half_mm_route(mojo_gpu):
    """Half inputs: the matrix-multiply route works as on CUDA."""
    torch.manual_seed(2)
    a = torch.randn(30, 8)
    b = torch.randn(4, 8)
    want = torch.cdist(a, b)
    got = torch.cdist(a.half().to(mojo_gpu), b.half().to(mojo_gpu))
    assert got.dtype == torch.float16
    torch.testing.assert_close(got.cpu().float(), want, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "shapes",
    [
        ((0, 5), (4, 5)),
        ((4, 5), (0, 5)),
        ((0, 4, 5), (3, 5)),
        ((1, 4, 5), (0, 3, 5)),
        ((3, 0), (4, 0)),
    ],
    ids=str,
)
def test_cdist_empty(mojo_gpu, shapes):
    a = torch.randn(shapes[0], requires_grad=True)
    b = torch.randn(shapes[1], requires_grad=True)
    want = torch.cdist(a, b, 1.0)
    am = a.detach().to(mojo_gpu).requires_grad_()
    bm = b.detach().to(mojo_gpu).requires_grad_()
    got = torch.cdist(am, bm, 1.0)
    assert got.shape == want.shape
    torch.testing.assert_close(got.cpu(), want)
    if want.numel():
        (ga,) = torch.autograd.grad(got.sum(), am)
        (wa,) = torch.autograd.grad(want.sum(), a)
        torch.testing.assert_close(ga.cpu(), wa)


def test_cdist_errors(mojo_gpu):
    a = torch.randn(3, 4).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="only supports non-negative p values"):
        aten._cdist_forward(a, a, -1.0, None)
    with pytest.raises(RuntimeError, match="possible modes: 0, 1, 2, but was: 3"):
        aten._cdist_forward(a, a, 2.0, 3)
    with pytest.raises(RuntimeError, match="same number of columns"):
        aten._cdist_forward(a, torch.randn(3, 5).to(mojo_gpu), 2.0, None)
    with pytest.raises(RuntimeError, match="at least 2D tensors, X1 got: 1D"):
        aten._cdist_forward(a[0], a, 2.0, None)
    with pytest.raises(RuntimeError, match="floating-point dtypes, X1 got: Long"):
        aten._cdist_forward(a.long(), a, 2.0, None)
    with pytest.raises(RuntimeError, match="must match the size of tensor b"):
        aten._cdist_forward(
            torch.randn(2, 3, 4).to(mojo_gpu),
            torch.randn(3, 3, 4).to(mojo_gpu),
            1.0,
            None,
        )
    # CUDA's kernel dispatches float and double only.
    with pytest.raises(RuntimeError, match="\"cdist_cuda\" not implemented for 'Half'"):
        torch.cdist(a.half(), a.half(), 1.0)


def test_out_overloads(mojo_gpu):
    """The autogenerated `.out` overloads: the functional result resized
    into `out` and copied; `out` must already hold the result's dtype."""
    torch.manual_seed(6)
    a = torch.randn(3, 4)
    b = torch.randn(5, 4)
    d = lambda t: t.to(mojo_gpu)  # noqa: E731
    dist = aten._cdist_forward(a, b, 3.0, None)
    out = torch.empty(0, device=mojo_gpu)
    assert aten._cdist_forward.out(d(a), d(b), 3.0, None, out=out) is out
    torch.testing.assert_close(out.cpu(), dist)
    grad = torch.randn(3, 5)
    # The generated `copy_arg` takes the result's exact dtype: no cast.
    out = torch.empty(3, 4, dtype=torch.float16, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
        aten._cdist_backward.out(d(grad), d(a), d(b), 3.0, d(dist), out=out)
    out = torch.empty(3, 4, device=mojo_gpu)
    aten._cdist_backward.out(d(grad), d(a), d(b), 3.0, d(dist), out=out)
    torch.testing.assert_close(out.cpu(), aten._cdist_backward(grad, a, b, 3.0, dist))
    with pytest.raises(RuntimeError, match="unsupported operation"):
        aten._pdist_forward.out(
            d(b), 1.0, out=torch.empty(1, device=mojo_gpu).expand(10)
        )
    pd = aten._pdist_forward(b, 1.0)
    out = torch.empty(0, device=mojo_gpu)
    aten._pdist_forward.out(d(b), 1.0, out=out)
    torch.testing.assert_close(out.cpu(), pd)
    g = torch.randn(pd.shape)
    out = torch.empty(0, device=mojo_gpu)
    aten._pdist_backward.out(d(g), d(b), 1.0, d(pd), out=out)
    torch.testing.assert_close(out.cpu(), aten._pdist_backward(g, b, 1.0, pd))
    x = torch.randn(4, 6)
    glu = aten.glu(x, 1)
    out = torch.empty(0, device=mojo_gpu)
    aten.glu_jvp.out(d(glu), d(x), d(x), 1, out=out)
    torch.testing.assert_close(out.cpu(), aten.glu_jvp(glu, x, x, 1))
    gx = aten.glu_backward(glu, x, 1)
    out = torch.empty(0, device=mojo_gpu)
    aten.glu_backward_jvp.out(d(gx), d(glu), d(x), d(glu), d(x), 1, out=out)
    torch.testing.assert_close(out.cpu(), aten.glu_backward_jvp(gx, glu, x, glu, x, 1))


@pytest.mark.parametrize("dtype", DIST_DTYPES)
@pytest.mark.parametrize("p", [0.0, 1.0, 2.0, 10.0, 0.5, 1.5, float("inf")])
@pytest.mark.parametrize("n,m", [(1, 5), (5, 1), (5, 5), (37, 357), (300, 3)])
def test_pdist(mojo_gpu, dtype, p, n, m):
    _dtype_or_skip(mojo_gpu, dtype)
    torch.manual_seed(3)
    x = torch.randn(n, m, dtype=dtype, requires_grad=True)
    want = F.pdist(x, p)
    xm = x.detach().to(mojo_gpu).requires_grad_()
    with ran("aten::_pdist_forward"):
        got = F.pdist(xm, p)
    rtol, atol = _tol(dtype, p)
    torch.testing.assert_close(got.cpu(), want.detach(), rtol=rtol, atol=atol)
    if want.numel():
        grad = torch.randn_like(want)
        (wx,) = torch.autograd.grad(want, x, grad)
        with ran("aten::_pdist_backward"):
            (gx,) = torch.autograd.grad(got, xm, grad.to(mojo_gpu))
        torch.testing.assert_close(gx.cpu(), wx, rtol=rtol, atol=atol)


def test_pdist_zero_columns_and_grad_stride(mojo_gpu):
    x = torch.randn(4, 0)
    torch.testing.assert_close(F.pdist(x.to(mojo_gpu)).cpu(), F.pdist(x))
    # A strided grad: `_pdist_backward` reads grad[k * stride].
    x = torch.randn(6, 3)
    dist = aten._pdist_forward(x, 2.0)
    grad = torch.randn(2 * dist.numel())[::2]
    want = aten._pdist_backward(grad, x, 2.0, dist)
    got = aten._pdist_backward(
        grad.to(mojo_gpu), x.to(mojo_gpu), 2.0, dist.to(mojo_gpu)
    )
    torch.testing.assert_close(got.cpu(), want)


def test_pdist_errors(mojo_gpu):
    x = torch.randn(4, 3).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="requires contiguous input"):
        aten._pdist_forward(x.t(), 2.0)
    with pytest.raises(RuntimeError, match="\"pdist_cuda\" not implemented for 'Half'"):
        aten._pdist_forward(x.half(), 2.0)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64]
)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_glu_jvp(mojo_gpu, dtype, dim):
    """glu_jvp_kernel computes in float and rounds once: the reference is
    the float32 kernel over the same operands (glu's output already rounded
    to the dtype, as forward AD hands it over), rounded to the dtype."""
    _dtype_or_skip(mojo_gpu, dtype)
    torch.manual_seed(4)
    x = (torch.randn(4, 6, 8) * 3).to(dtype)
    t = torch.randn(4, 6, 8).to(dtype)
    wide = torch.float64 if dtype == torch.float64 else torch.float32
    glu = F.glu(x.to(wide), dim).to(dtype).to(wide)
    want = aten.glu_jvp(glu, x.to(wide), t.to(wide), dim)
    with ran("aten::glu_jvp"):
        got = jvp(lambda z: F.glu(z, dim), (x.to(mojo_gpu),), (t.to(mojo_gpu),))[1]
    assert got.dtype == dtype
    torch.testing.assert_close(got.cpu(), want.to(dtype))


def test_glu_jvp_errors(mojo_gpu):
    x = torch.randn(4, 6).to(mojo_gpu)
    glu = F.glu(x, 1)
    with pytest.raises(RuntimeError, match="Found dtype Half but expected Float"):
        aten.glu_jvp(glu, x, x.half(), 1)
    with pytest.raises(RuntimeError, match="exceeds dimension size"):
        aten.glu_jvp(glu, x[:, :4], x, 1)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64]
)
@pytest.mark.parametrize("dim", [0, 2, -1])
def test_glu_backward_jvp(mojo_gpu, dtype, dim):
    """The same composite as GatedLinearUnit.cpp's, op by op: float32 and
    float64 match the CPU's; the half types round every intermediate as the
    CUDA composite does, so they are held to the float math loosely."""
    _dtype_or_skip(mojo_gpu, dtype)
    if dtype == torch.float64:
        # ATen's composite runs sigmoid, which this device does not have in
        # float64 yet: the op declines exactly where its sigmoid does.
        try:
            torch.sigmoid(torch.zeros(1, dtype=dtype, device=mojo_gpu))
        except NotImplementedError:
            pytest.skip("sigmoid has no float64 kernel on this device")
    torch.manual_seed(5)
    x = torch.randn(4, 6, 8).to(dtype)
    dx = torch.randn(4, 6, 8).to(dtype)
    half_shape = list(aten.glu(x, dim).shape)
    grad_glu = torch.randn(half_shape).to(dtype)
    dgrad_glu = torch.randn(half_shape).to(dtype)
    grad_x = aten.glu_backward(grad_glu, x, dim)
    args = (grad_x, grad_glu, x, dgrad_glu, dx)
    with ran("aten::glu_backward_jvp"):
        got = aten.glu_backward_jvp(*(a.to(mojo_gpu) for a in args), dim)
    assert got.dtype == dtype and got.shape == x.shape
    if dtype in (torch.float32, torch.float64):
        want = aten.glu_backward_jvp(*args, dim)
        torch.testing.assert_close(got.cpu(), want, rtol=1e-5, atol=1e-5)
    else:
        want = aten.glu_backward_jvp(*(a.float() for a in args), dim)
        torch.testing.assert_close(got.cpu().float(), want, rtol=5e-2, atol=5e-2)


def test_cdist_matmul_route_multi_dim_batches(mojo_gpu):
    """p=2 with more than 25 rows takes `_euclidean_dist` over the batches
    broadcast THEN flattened, as Distance.cpp does."""
    torch.manual_seed(7)
    a = torch.randn(2, 1, 30, 4)
    b = torch.randn(1, 3, 27, 4)
    want = aten._cdist_forward(a, b, 2.0, None)
    with ran("aten::_cdist_forward"):
        got = aten._cdist_forward(a.to(mojo_gpu), b.to(mojo_gpu), 2.0, None)
    assert got.shape == (2, 3, 30, 27)
    torch.testing.assert_close(got.cpu(), want, atol=1e-4, rtol=1e-4)


def test_glu_jvp_broadcasts(mojo_gpu):
    """glu_jvp's result takes the broadcast shape of all its operands."""
    torch.manual_seed(8)
    glu, x, dx = torch.randn(1, 2), torch.randn(3, 4), torch.randn(3, 4)
    want = aten.glu_jvp(glu, x, dx, 1)
    got = aten.glu_jvp(glu.to(mojo_gpu), x.to(mojo_gpu), dx.to(mojo_gpu), 1)
    assert got.shape == want.shape == (3, 2)
    torch.testing.assert_close(got.cpu(), want)


def test_glu_backward_jvp_broadcasts(mojo_gpu):
    """grad_glu of length 1 along dim against a dgrad_glu of length 3: the
    halves take the broadcast length."""
    torch.manual_seed(9)
    gx, gg, x = torch.randn(2, 2), torch.randn(2, 1), torch.randn(2, 2)
    dgg, dx = torch.randn(2, 3), torch.randn(2, 2)
    want = aten.glu_backward_jvp(gx, gg, x, dgg, dx, 1)
    got = aten.glu_backward_jvp(*[t.to(mojo_gpu) for t in (gx, gg, x, dgg, dx)], 1)
    assert got.shape == want.shape
    torch.testing.assert_close(got.cpu(), want)
