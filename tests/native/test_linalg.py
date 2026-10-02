"""The linalg group of the native mojo backend (tmb/ops/linalg.mojo on the
`linalg` kernel family): factorizations, solves, decompositions, through
the public torch API, compared against CPU torch. Where the answer is not
unique (eigenvector and singular vector signs) the tests check the
invariants torch's own OpInfos check: reconstructions, orthonormality and
the absolute values."""

import pytest
import torch

from tests.native.conftest import is_metal, ran, skip_if_metal

DTYPES = (torch.float32, torch.float64)


def _tol(dtype: torch.dtype) -> float:
    return 1e-4 if dtype == torch.float32 else 1e-10


def _dtype_or_skip(device: str, dtype: torch.dtype):
    if dtype == torch.float64:
        skip_if_metal(device, "Apple GPUs have no float64")


def _close(actual, expected, dtype):
    if isinstance(expected, (tuple, list)):
        for a, e in zip(actual, expected, strict=True):
            _close(a, e, dtype)
        return
    torch.testing.assert_close(
        actual.cpu(), expected, atol=_tol(dtype), rtol=_tol(dtype)
    )


def _spd(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    a = torch.randn(*shape, dtype=torch.float64)
    n = shape[-1]
    return (a @ a.mT + n * torch.eye(n, dtype=torch.float64)).to(dtype)


def _sym(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    a = torch.randn(*shape, dtype=dtype)
    return a + a.mT


# --- Cholesky ----------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("shape", [(5, 5), (3, 4, 4), (2, 1, 7, 7), (0, 3, 3), (0, 0)])
def test_cholesky(mojo_gpu: str, dtype: torch.dtype, upper: bool, shape):
    _dtype_or_skip(mojo_gpu, dtype)
    torch.manual_seed(0)
    a = _spd(*shape, dtype=dtype)
    with ran("aten::linalg_cholesky_ex"):
        got = torch.linalg.cholesky(a.to(mojo_gpu), upper=upper)
    _close(got, torch.linalg.cholesky(a, upper=upper), dtype)
    assert got.mT.is_contiguous() or got.numel() == 0  # column-major, as CUDA
    _close(
        torch.cholesky(a.to(mojo_gpu), upper=upper),
        torch.cholesky(a, upper=upper),
        dtype,
    )


def test_cholesky_non_contiguous_and_reads_one_triangle(mojo_gpu: str):
    torch.manual_seed(1)
    a = _spd(6, 6)
    junk = a.clone()
    junk.triu_(1).fill_(1e6)  # only the lower triangle may be read
    lower_only = torch.tril(a) + torch.triu(junk, 1)
    big = torch.zeros(6, 12)
    big[:, ::2] = lower_only
    got = torch.linalg.cholesky(big.to(mojo_gpu)[:, ::2])
    _close(got, torch.linalg.cholesky(a), torch.float32)


def test_cholesky_not_positive_definite(mojo_gpu: str):
    a = torch.stack(
        [_spd(4, 4), -torch.eye(4), torch.diag(torch.tensor([1.0, 2.0, -1.0, 3.0]))]
    )
    L, info = torch.linalg.cholesky_ex(a.to(mojo_gpu))
    assert info.cpu().tolist() == [0, 1, 3]
    assert info.dtype == torch.int32
    with pytest.raises(torch.linalg.LinAlgError, match="not positive-definite"):
        torch.linalg.cholesky(a.to(mojo_gpu))
    # check_errors=True raises from inside the op, through the backend's
    # dispatcher bridge, which carries torch's message but not its
    # LinAlgError subclass (a RuntimeError).
    with pytest.raises(RuntimeError, match="not positive-definite"):
        torch.linalg.cholesky_ex(a.to(mojo_gpu), check_errors=True)


def test_cholesky_out(mojo_gpu: str):
    a = _spd(3, 4, 4)
    out = torch.empty(0, device=mojo_gpu)
    info = torch.empty(0, dtype=torch.int32, device=mojo_gpu)
    torch.linalg.cholesky_ex(a.to(mojo_gpu), out=(out, info))
    _close(out, torch.linalg.cholesky(a), torch.float32)
    assert info.cpu().tolist() == [0, 0, 0]
    with pytest.raises(RuntimeError, match="dtype"):
        torch.linalg.cholesky_ex(
            a.to(mojo_gpu),
            out=(torch.empty(0, dtype=torch.float64, device=mojo_gpu), info),
        )


@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_inverse_and_solve(mojo_gpu: str, upper: bool):
    torch.manual_seed(2)
    a = _spd(2, 5, 5)
    f = torch.linalg.cholesky(a, upper=upper)
    _close(
        torch.cholesky_inverse(f.to(mojo_gpu), upper=upper),
        torch.cholesky_inverse(f, upper=upper),
        torch.float32,
    )
    b = torch.randn(3, 1, 5, 2)  # broadcasts against f's batch
    _close(
        torch.cholesky_solve(b.to(mojo_gpu), f.to(mojo_gpu), upper=upper),
        torch.cholesky_solve(b, f, upper=upper),
        torch.float32,
    )


# --- LU ----------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape", [(5, 5), (4, 7), (7, 4), (3, 6, 6), (0, 4, 4), (0, 0)]
)
def test_lu_factor_matches_lapack(mojo_gpu: str, dtype: torch.dtype, shape):
    _dtype_or_skip(mojo_gpu, dtype)
    torch.manual_seed(3)
    a = torch.randn(*shape, dtype=dtype)
    with ran("aten::linalg_lu_factor_ex"):
        LU, piv, info = torch.linalg.lu_factor_ex(a.to(mojo_gpu))
    LU0, piv0, info0 = torch.linalg.lu_factor_ex(a)
    _close(LU, LU0, dtype)
    assert torch.equal(piv.cpu(), piv0) and piv.dtype == torch.int32
    assert torch.equal(info.cpu(), info0)
    assert LU.mT.is_contiguous() or LU.numel() == 0
    P, L, U = torch.linalg.lu(a.to(mojo_gpu))
    _close((P, L, U), torch.linalg.lu(a), dtype)


def test_lu_singular_info(mojo_gpu: str):
    a = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]])
    z = torch.zeros(3, 3)
    _, _, info = torch.linalg.lu_factor_ex(
        torch.stack([a, z, torch.eye(3)]).to(mojo_gpu)
    )
    _, _, info0 = torch.linalg.lu_factor_ex(torch.stack([a, z, torch.eye(3)]))
    assert info.cpu().tolist() == info0.tolist()
    with pytest.raises(RuntimeError, match=r"U\[1,1\] is zero"):
        torch.linalg.lu_factor(z.to(mojo_gpu))
    with pytest.raises(torch.linalg.LinAlgError, match="singular"):
        torch.linalg.inv(z.to(mojo_gpu))


def test_lu_without_pivoting(mojo_gpu: str):
    torch.manual_seed(4)
    a = torch.randn(2, 5, 5) + 10 * torch.eye(5)
    P, L, U = torch.linalg.lu(a.to(mojo_gpu), pivot=False)
    assert P.numel() == 0
    _close(L @ U, a, torch.float32)
    _, piv = torch.linalg.lu_factor(a.to(mojo_gpu), pivot=False)
    assert piv.cpu().tolist() == [[1, 2, 3, 4, 5]] * 2


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("adjoint", [True, False])
def test_lu_solve(mojo_gpu: str, left: bool, adjoint: bool):
    torch.manual_seed(5)
    a = torch.randn(2, 4, 4)
    LU, piv = torch.linalg.lu_factor(a)
    b = torch.randn(3, 1, 4, 2) if left else torch.randn(3, 1, 2, 4)
    got = torch.linalg.lu_solve(
        LU.to(mojo_gpu), piv.to(mojo_gpu), b.to(mojo_gpu), left=left, adjoint=adjoint
    )
    _close(
        got,
        torch.linalg.lu_solve(LU, piv, b, left=left, adjoint=adjoint),
        torch.float32,
    )
    _close(
        torch.lu_unpack(LU.to(mojo_gpu), piv.to(mojo_gpu)),
        torch.lu_unpack(LU, piv),
        torch.float32,
    )


# --- inv, det, slogdet, solve --------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_inv_det_slogdet(mojo_gpu: str, dtype: torch.dtype):
    _dtype_or_skip(mojo_gpu, dtype)
    torch.manual_seed(6)
    a = torch.randn(3, 5, 5, dtype=dtype)
    a_t = a.mT  # non-contiguous
    for x in (a, a_t):
        d = x.to(mojo_gpu)
        _close(torch.linalg.inv(d), torch.linalg.inv(x), dtype)
        _close(torch.linalg.det(d), torch.linalg.det(x), dtype)
        _close(torch.linalg.slogdet(d), torch.linalg.slogdet(x), dtype)
    singular = torch.zeros(2, 2, dtype=dtype)
    assert torch.linalg.det(singular.to(mojo_gpu)).item() == 0.0
    sgn, logabs = torch.linalg.slogdet(singular.to(mojo_gpu))
    assert sgn.item() == 0.0 and logabs.item() == float("-inf")
    _close(
        torch.linalg.det(torch.empty(0, 0, dtype=dtype).to(mojo_gpu)),
        torch.ones((), dtype=dtype),
        dtype,
    )


@pytest.mark.parametrize("left", [True, False])
def test_solve(mojo_gpu: str, left: bool):
    torch.manual_seed(7)
    a = torch.randn(2, 1, 4, 4)
    b = torch.randn(3, 4, 2) if left else torch.randn(3, 2, 4)
    _close(
        torch.linalg.solve(a.to(mojo_gpu), b.to(mojo_gpu), left=left),
        torch.linalg.solve(a, b, left=left),
        torch.float32,
    )
    v = torch.randn(2, 1, 4)  # the vector right-hand side
    _close(
        torch.linalg.solve(a.to(mojo_gpu), v.to(mojo_gpu)),
        torch.linalg.solve(a, v),
        torch.float32,
    )
    x, info = torch.linalg.solve_ex(a.to(mojo_gpu), v.to(mojo_gpu))
    assert info.cpu().eq(0).all()


# --- triangular solves -----------------------------------------------------------


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("left", [False, True])
@pytest.mark.parametrize("unit", [False, True])
def test_solve_triangular(mojo_gpu: str, upper: bool, left: bool, unit: bool):
    torch.manual_seed(8)
    a = torch.randn(2, 1, 5, 5) + 5 * torch.eye(5)
    b = torch.randn(3, 5, 7) if left else torch.randn(3, 7, 5)
    kw = {"upper": upper, "left": left, "unitriangular": unit}
    _close(
        torch.linalg.solve_triangular(a.to(mojo_gpu), b.to(mojo_gpu), **kw),
        torch.linalg.solve_triangular(a, b, **kw),
        torch.float32,
    )
    at = a.mT  # transposed A: the other triangle, read through strides
    _close(
        torch.linalg.solve_triangular(at.to(mojo_gpu), b.to(mojo_gpu), **kw),
        torch.linalg.solve_triangular(at, b, **kw),
        torch.float32,
    )


def test_solve_triangular_many_rhs_and_out(mojo_gpu: str):
    torch.manual_seed(9)
    a = torch.randn(40, 40).tril() + 40 * torch.eye(40)
    b = torch.randn(40, 300)  # more right-hand sides than one block holds
    out = torch.empty(0, device=mojo_gpu)
    torch.linalg.solve_triangular(a.to(mojo_gpu), b.to(mojo_gpu), upper=False, out=out)
    _close(out, torch.linalg.solve_triangular(a, b, upper=False), torch.float32)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("upper", [False, True])
def test_triangular_solve(mojo_gpu: str, transpose: bool, upper: bool):
    torch.manual_seed(10)
    a = torch.randn(2, 4, 4) + 4 * torch.eye(4)
    b = torch.randn(2, 4, 3)
    _close(
        torch.triangular_solve(
            b.to(mojo_gpu), a.to(mojo_gpu), upper=upper, transpose=transpose
        ),
        torch.triangular_solve(b, a, upper=upper, transpose=transpose),
        torch.float32,
    )


# --- QR --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape", [(5, 3), (3, 5), (4, 4), (2, 6, 2), (0, 3, 3), (3, 0)]
)
def test_qr(mojo_gpu: str, dtype: torch.dtype, shape):
    _dtype_or_skip(mojo_gpu, dtype)
    torch.manual_seed(11)
    a = torch.randn(*shape, dtype=dtype)
    _close(torch.geqrf(a.to(mojo_gpu)), torch.geqrf(a), dtype)
    for mode in ("reduced", "complete", "r"):
        _close(
            torch.linalg.qr(a.to(mojo_gpu), mode=mode),
            torch.linalg.qr(a, mode=mode),
            dtype,
        )


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("transpose", [True, False])
def test_householder_product_and_ormqr(mojo_gpu: str, left: bool, transpose: bool):
    torch.manual_seed(12)
    qr, tau = torch.geqrf(torch.randn(2, 6, 4))
    _close(
        torch.linalg.householder_product(qr.to(mojo_gpu), tau.to(mojo_gpu)),
        torch.linalg.householder_product(qr, tau),
        torch.float32,
    )
    c = torch.randn(2, 6, 3) if left else torch.randn(2, 3, 6)
    _close(
        torch.ormqr(qr.to(mojo_gpu), tau.to(mojo_gpu), c.to(mojo_gpu), left, transpose),
        torch.ormqr(qr, tau, c, left, transpose),
        torch.float32,
    )


# --- eigh ------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("uplo", ["L", "U"])
@pytest.mark.parametrize("shape", [(6, 6), (3, 5, 5), (1, 1), (33, 33), (0, 4, 4)])
def test_eigh(mojo_gpu: str, dtype: torch.dtype, uplo: str, shape):
    _dtype_or_skip(mojo_gpu, dtype)
    torch.manual_seed(13)
    a = _sym(*shape, dtype=dtype)
    # garbage in the triangle UPLO does not name
    junk = a.triu(1) if uplo == "L" else a.tril(-1)
    w, v = torch.linalg.eigh((a + 100 * junk).to(mojo_gpu), UPLO=uplo)
    w0 = torch.linalg.eigvalsh(a)
    _close(w, w0, dtype)
    vc = v.cpu()
    _close(vc @ torch.diag_embed(w.cpu()) @ vc.mT, a, dtype)
    _close(vc.mT @ vc, torch.eye(shape[-1], dtype=dtype).expand_as(a), dtype)
    _close(torch.linalg.eigvalsh((a + 100 * junk).to(mojo_gpu), UPLO=uplo), w0, dtype)


def test_eigh_repeated_eigenvalues(mojo_gpu: str):
    a = torch.diag(torch.tensor([3.0, 1.0, 3.0, -2.0, 1.0]))
    q, _ = torch.linalg.qr(torch.randn(5, 5))
    a = q @ a @ q.mT
    a = (a + a.mT) / 2
    w, v = torch.linalg.eigh(a.to(mojo_gpu))
    _close(w, torch.tensor([-2.0, 1.0, 1.0, 3.0, 3.0]), torch.float32)
    _close(v.cpu() @ torch.diag(w.cpu()) @ v.cpu().mT, a, torch.float32)


# --- SVD -------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize(
    "shape", [(5, 3), (3, 5), (4, 4), (2, 7, 3), (0, 3), (3, 0), (2, 0, 0)]
)
def test_svd(mojo_gpu: str, dtype: torch.dtype, full: bool, shape):
    _dtype_or_skip(mojo_gpu, dtype)
    torch.manual_seed(14)
    a = torch.randn(*shape, dtype=dtype)
    U, S, Vh = torch.linalg.svd(a.to(mojo_gpu), full_matrices=full)
    U0, S0, Vh0 = torch.linalg.svd(a, full_matrices=full)
    assert U.shape == U0.shape and S.shape == S0.shape and Vh.shape == Vh0.shape
    _close(S, S0, dtype)
    U, S, Vh = U.cpu(), S.cpu(), Vh.cpu()
    k = S.shape[-1]
    _close(U[..., :k] @ torch.diag_embed(S) @ Vh[..., :k, :], a, dtype)
    _close(
        U.mT @ U,
        torch.eye(U.shape[-1], dtype=dtype).expand(*U.shape[:-2], -1, -1),
        dtype,
    )
    _close(
        Vh @ Vh.mT,
        torch.eye(Vh.shape[-2], dtype=dtype).expand(*Vh.shape[:-2], -1, -1),
        dtype,
    )
    _close(torch.linalg.svdvals(a.to(mojo_gpu)), S0, dtype)


def test_svd_rank_deficient(mojo_gpu: str):
    torch.manual_seed(15)
    a = torch.randn(6, 2) @ torch.randn(2, 4)  # rank 2
    U, S, Vh = torch.linalg.svd(a.to(mojo_gpu), full_matrices=True)
    U, S, Vh = U.cpu(), S.cpu(), Vh.cpu()
    _close(S, torch.linalg.svdvals(a), torch.float32)
    _close(U.mT @ U, torch.eye(6), torch.float32)
    _close(U[:, :4] @ torch.diag(S) @ Vh, a, torch.float32)
    # matrix_rank / pinv compare against a float64 0-d tolerance torch's
    # composite builds: no float64 on an Apple GPU.
    skip_if_metal(mojo_gpu, "matrix_rank's float64 tolerance tensor")
    _close(torch.linalg.matrix_rank(a.to(mojo_gpu)), torch.tensor(2), torch.float32)
    _close(torch.linalg.pinv(a.to(mojo_gpu)), torch.linalg.pinv(a), torch.float32)


# --- LDL, lstsq, matrix_exp ----------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_ldl(mojo_gpu: str, dtype: torch.dtype):
    _dtype_or_skip(mojo_gpu, dtype)
    torch.manual_seed(16)
    a = _sym(3, 6, 6, dtype=dtype)
    a[1, 0, 0] = 0  # forces a 2x2 pivot at the start
    LD, piv, info = torch.linalg.ldl_factor_ex(a.to(mojo_gpu))
    LD0, piv0, info0 = torch.linalg.ldl_factor_ex(a)
    _close(LD, LD0, dtype)
    assert torch.equal(piv.cpu(), piv0) and torch.equal(info.cpu(), info0)
    b = torch.randn(3, 6, 2, dtype=dtype)
    _close(
        torch.linalg.ldl_solve(LD, piv, b.to(mojo_gpu)),
        torch.linalg.ldl_solve(LD0, piv0, b),
        dtype,
    )


@pytest.mark.parametrize("shape", [(6, 3), (3, 6), (4, 4)])
def test_lstsq(mojo_gpu: str, shape):
    """CUDA's lstsq: the 'gels' driver (the default) through QR."""
    torch.manual_seed(17)
    a = torch.randn(2, *shape)
    b = torch.randn(2, shape[0], 2)
    got = torch.linalg.lstsq(a.to(mojo_gpu), b.to(mojo_gpu))
    exp = torch.linalg.lstsq(a, b, driver="gels")
    if shape[0] < shape[1]:
        # the minimum-norm solution of the underdetermined system
        _close(got.solution, torch.linalg.pinv(a) @ b, torch.float32)
    else:
        _close(got.solution, exp.solution, torch.float32)
    _close(got.residuals, exp.residuals, torch.float32)
    assert got.rank.numel() == 0 and got.singular_values.numel() == 0


@pytest.mark.parametrize("driver", ["gelsy", "gelsd", "GELSS"])
def test_lstsq_rejects_other_drivers_as_cuda(mojo_gpu: str, driver: str):
    a = torch.randn(4, 3).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="other than `gels` is not supported"):
        torch.linalg.lstsq(a, a, driver=driver)


def test_lstsq_out_dtype_checked(mojo_gpu: str):
    a = torch.randn(4, 3).to(mojo_gpu)
    sol = torch.empty(0, dtype=torch.int64, device=mojo_gpu)
    res = torch.empty(0, device=mojo_gpu)
    rank = torch.empty(0, dtype=torch.int64, device=mojo_gpu)
    sv = torch.empty(0, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="Expected solution to be safely castable"):
        torch.linalg.lstsq(a, a, out=(sol, res, rank, sv))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale", [1e-9, 1e-3, 0.3, 1.0, 20.0])
def test_matrix_exp(mojo_gpu: str, dtype: torch.dtype, scale: float):
    _dtype_or_skip(mojo_gpu, dtype)
    if dtype == torch.float64:
        # composed of bmm, which has no float64 route on the device yet
        with pytest.raises(NotImplementedError):
            torch.linalg.matrix_exp(torch.ones(2, 3, 3, dtype=dtype).to(mojo_gpu))
        return
    torch.manual_seed(18)
    for shape in ((4, 4), (3, 4, 4), (1, 1), (0, 0)):
        a = torch.randn(*shape, dtype=dtype) * scale
        e = torch.linalg.matrix_exp(a.double()).to(dtype)
        got = torch.linalg.matrix_exp(a.to(mojo_gpu)).cpu()
        torch.testing.assert_close(
            got,
            e,
            atol=1e-5 * max(1, e.abs().max().item() if e.numel() else 1),
            rtol=1e-4,
        )


def test_complex_declines(mojo_gpu: str):
    a = torch.randn(3, 3, dtype=torch.complex64)
    with pytest.raises(NotImplementedError):
        torch.linalg.cholesky(a.to(mojo_gpu))


def test_float64_declines_on_metal(mojo_gpu: str):
    if not is_metal(mojo_gpu):
        pytest.skip("only Apple GPUs lack float64")
    with pytest.raises(NotImplementedError):
        torch.linalg.inv(torch.eye(3, dtype=torch.float64).to(mojo_gpu))


# --- review regressions ---------------------------------------------------------


def test_matrix_exp_non_finite(mojo_gpu: str):
    inf = torch.tensor([[float("inf"), 1.0], [0.0, 1.0]])
    assert torch.linalg.matrix_exp(inf.to(mojo_gpu)).cpu().isnan().all()
    torch.manual_seed(19)
    batch = torch.stack([torch.full((3, 3), float("nan")), 5 * torch.eye(3)])
    got = torch.linalg.matrix_exp(batch.to(mojo_gpu)).cpu()
    assert got[0].isnan().all()
    _close(got[1], torch.linalg.matrix_exp(batch[1]), torch.float32)
    big = torch.stack([torch.full((3, 3), float("nan")), 20 * torch.randn(3, 3)])
    exp = torch.linalg.matrix_exp(big[1].double()).float()
    torch.testing.assert_close(
        torch.linalg.matrix_exp(big.to(mojo_gpu)).cpu()[1],
        exp,
        rtol=1e-3,
        atol=1e-3 * exp.abs().max().item(),
    )


def test_matrix_exp_empty_batch(mojo_gpu: str):
    assert torch.linalg.matrix_exp(torch.empty(0, 3, 3).to(mojo_gpu)).shape == (0, 3, 3)


def test_operands_on_other_devices_are_refused(mojo_gpu: str):
    qr, tau = torch.geqrf(torch.randn(4, 3))
    c = torch.randn(4, 2)
    with pytest.raises(RuntimeError, match="same device"):
        torch.ormqr(qr.to(mojo_gpu), tau, c.to(mojo_gpu))
    LU, piv = torch.linalg.lu_factor(torch.randn(3, 3))
    with pytest.raises(RuntimeError, match="same device"):
        torch.linalg.lu_solve(LU.to(mojo_gpu), piv, torch.randn(3, 1).to(mojo_gpu))
    with pytest.raises(RuntimeError, match="same device"):
        torch.lu_unpack(LU.to(mojo_gpu), piv)
    a = torch.randn(3, 3).triu() + 3 * torch.eye(3)
    with pytest.raises(RuntimeError, match="same device"):
        torch.linalg.solve_triangular(
            a.to(mojo_gpu),
            torch.randn(3, 1).to(mojo_gpu),
            upper=True,
            out=torch.empty(3, 1),
        )


@pytest.mark.parametrize("scale", [1e20, 1e-30])
def test_qr_and_svd_extreme_scales(mojo_gpu: str, scale: float):
    a = torch.tensor([[1.0, 0.0], [1.0, 1.0]]) * scale
    q, r = torch.linalg.qr(a.to(mojo_gpu))
    q0, r0 = torch.linalg.qr(a)
    torch.testing.assert_close(q.cpu(), q0, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(r.cpu(), r0, rtol=1e-5, atol=0.0)
    _close(torch.geqrf(a.to(mojo_gpu))[1], torch.geqrf(a)[1], torch.float32)
    s = torch.linalg.svdvals((scale * torch.eye(3)).to(mojo_gpu)).cpu()
    torch.testing.assert_close(s, torch.full((3,), scale), rtol=1e-6, atol=0.0)


def test_eigh_extreme_scale(mojo_gpu: str):
    a = torch.tensor([[-2e38, 1e38], [1e38, 2e38]])
    w = torch.linalg.eigvalsh(a.to(mojo_gpu)).cpu()
    torch.testing.assert_close(
        w, torch.tensor([-2.2361e38, 2.2361e38]), rtol=1e-4, atol=0.0
    )


def test_eigh_rank_deficient_converges(mojo_gpu: str):
    torch.manual_seed(20)
    v = torch.randn(64, 1)
    a = v @ v.mT  # rank 1
    w, q = torch.linalg.eigh(a.to(mojo_gpu))
    _close(w, torch.linalg.eigvalsh(a), torch.float32)
    _close(q.cpu() @ torch.diag(w.cpu()) @ q.cpu().mT, a, torch.float32)


def test_eigh_and_svd_of_nan_raise(mojo_gpu: str):
    a = torch.full((3, 3), float("nan"))
    with pytest.raises(RuntimeError, match="failed to converge"):
        torch.linalg.eigh(a.to(mojo_gpu))
    with pytest.raises(RuntimeError, match="failed to converge"):
        torch.linalg.svd(a.to(mojo_gpu))


def test_solve_triangular_strides_match_torch(mojo_gpu: str):
    a = torch.randn(2, 3, 3).triu() + 3 * torch.eye(3)
    for left, b in ((True, torch.randn(2, 3, 4)), (False, torch.randn(2, 4, 3))):
        got = torch.linalg.solve_triangular(
            a.to(mojo_gpu), b.to(mojo_gpu), upper=True, left=left
        )
        assert (
            got.stride()
            == torch.linalg.solve_triangular(a, b, upper=True, left=left).stride()
        )


def test_ldl_solve_accepts_int64_pivots_as_torch(mojo_gpu: str):
    a = _sym(4, 4)
    LD, piv = torch.linalg.ldl_factor(a)
    b = torch.randn(4, 2)
    _close(
        torch.linalg.ldl_solve(
            LD.to(mojo_gpu), piv.long().to(mojo_gpu), b.to(mojo_gpu)
        ),
        torch.linalg.ldl_solve(LD, piv.long(), b),
        torch.float32,
    )


# --- round-2 review regressions -----------------------------------------------------


def test_svd_keeps_small_singular_values_next_to_large(mojo_gpu: str):
    a = torch.diag(torch.tensor([2.0**30, 2.0**-60]))
    s = torch.linalg.svdvals(a.to(mojo_gpu)).cpu()
    assert s.tolist() == [2.0**30, 2.0**-60]
    U, S, Vh = torch.linalg.svd(a.to(mojo_gpu))
    _close(U.cpu().abs(), torch.eye(2), torch.float32)


def test_jacobi_small_block_next_to_large(mojo_gpu: str):
    a = torch.tensor([[1e8, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    s = torch.linalg.svdvals(a.to(mojo_gpu)).cpu()
    torch.testing.assert_close(
        s, torch.tensor([1e8, 1.618034, 0.618034]), rtol=1e-5, atol=0.0
    )
    e = torch.tensor([[1e8, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    w = torch.linalg.eigvalsh(e.to(mojo_gpu)).cpu()
    torch.testing.assert_close(w, torch.tensor([-1.0, 1.0, 1e8]), rtol=1e-6, atol=1e-6)


def test_jacobi_float32_accuracy_at_size(mojo_gpu: str):
    """Against the float64 answer, within a small factor of LAPACK's own
    float32 error (CPU torch's). Off Metal the float32 Jacobi runs in
    float64 and rounds once; on Metal (no float64) the rotations run in
    float32, and each sweep's rounding of the off-diagonal entries adds up:
    measured 20x CPU's error on the M4 for this case, so the bar is 50x
    there."""
    factor = 50 if is_metal(mojo_gpu) else 10
    torch.manual_seed(21)
    a = torch.randn(256, 256)
    a = a + a.mT
    exact = torch.linalg.eigvalsh(a.double())
    err = (torch.linalg.eigvalsh(a.to(mojo_gpu)).cpu().double() - exact).abs().max()
    cpu_err = (torch.linalg.eigvalsh(a).double() - exact).abs().max()
    assert err <= factor * cpu_err + 1e-5, (err, cpu_err)
    b = torch.randn(300, 200)
    exact = torch.linalg.svdvals(b.double())
    err = (torch.linalg.svdvals(b.to(mojo_gpu)).cpu().double() - exact).abs().max()
    cpu_err = (torch.linalg.svdvals(b).double() - exact).abs().max()
    assert err <= factor * cpu_err + 1e-5, (err, cpu_err)


def test_lstsq_empty_overdetermined_residuals(mojo_gpu: str):
    a = torch.empty(0, 3, 2)
    b = torch.empty(0, 3, 1)
    got = torch.linalg.lstsq(a.to(mojo_gpu), b.to(mojo_gpu))
    assert (
        got.residuals.shape
        == torch.linalg.lstsq(a, b, driver="gels").residuals.shape
        == (0, 1)
    )


def test_lstsq_complex_out_declines(mojo_gpu: str):
    # The device holds no complex tensor, so the only complex out that can
    # arrive is a host one: declined, not refused with a wrong message.
    a = torch.randn(4, 3).to(mojo_gpu)
    outs = (
        torch.empty(0, dtype=torch.complex64),
        torch.empty(0, device=mojo_gpu),
        torch.empty(0, dtype=torch.int64, device=mojo_gpu),
        torch.empty(0, device=mojo_gpu),
    )
    with pytest.raises(NotImplementedError):
        torch.linalg.lstsq(a, a, out=outs)


def test_matrix_exp_huge_norm_scaling(mojo_gpu: str):
    a = torch.diag(torch.tensor([-3e38, -3e38]))
    got = torch.linalg.matrix_exp(a.to(mojo_gpu)).cpu()
    assert torch.equal(got, torch.zeros(2, 2)), got


# --- round-3 review regressions -----------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_svd_converges_on_exactly_rank_deficient(mojo_gpu: str, dtype: torch.dtype):
    _dtype_or_skip(mojo_gpu, dtype)
    shapes = [(m, n) for m in (1, 2, 5, 12, 23, 37, 70) for n in (1, 3, 12, 22, 33)]
    for m, n in shapes:
        a = torch.ones(m, n, dtype=dtype)
        s = torch.linalg.svdvals(a.to(mojo_gpu)).cpu()
        assert abs(s[0].item() - (m * n) ** 0.5) <= 1e-4 * (m * n) ** 0.5, (m, n)
        assert (
            s[1:].abs().max().item() <= 1e-4 * (m * n) ** 0.5 if s.numel() > 1 else True
        )
    torch.manual_seed(22)
    for m, n, r in ((40, 30, 3), (17, 50, 1), (64, 64, 10)):
        a = (
            torch.randn(m, r, dtype=torch.float64)
            @ torch.randn(r, n, dtype=torch.float64)
        ).to(dtype)
        U, S, Vh = torch.linalg.svd(a.to(mojo_gpu), full_matrices=False)
        U, S, Vh = U.cpu(), S.cpu(), Vh.cpu()
        _close(
            S[:r], torch.linalg.svdvals(a)[:r], dtype
        ) if dtype == torch.float64 else None
        recon = U @ torch.diag(S) @ Vh
        assert (recon - a).abs().max() <= 1e-4 * a.abs().max(), (m, n, r)
    batch = torch.stack(
        [torch.ones(23, 22, dtype=dtype), torch.eye(23, 22, dtype=dtype)]
    )
    s = torch.linalg.svdvals(batch.to(mojo_gpu)).cpu()
    _close(s[1], torch.ones(22, dtype=dtype), dtype)


def test_eigh_negligible_off_diagonal_float64(mojo_gpu: str):
    skip_if_metal(mojo_gpu, "Apple GPUs have no float64")
    a = torch.tensor([[1e100, 1e-250], [1e-250, 0.0]], dtype=torch.float64)
    w = torch.linalg.eigvalsh(a.to(mojo_gpu)).cpu()
    assert w.tolist() == [0.0, 1e100]


def test_svd_tiny_singular_values_survive(mojo_gpu: str):
    if not is_metal(mojo_gpu):
        for vals in ((2.0**-400, 2.0**-600), (1.0, 1e-170)):
            a = torch.diag(torch.tensor(vals, dtype=torch.float64))
            s = torch.linalg.svdvals(a.to(mojo_gpu)).cpu()
            torch.testing.assert_close(
                s, torch.tensor(sorted(vals, reverse=True), dtype=torch.float64)
            )
    a = torch.diag(torch.tensor([1.0, 1e-25, 1e-36]))
    s = torch.linalg.svdvals(a.to(mojo_gpu)).cpu()
    torch.testing.assert_close(s, torch.tensor([1.0, 1e-25, 1e-36]))
    U, S, Vh = torch.linalg.svd(a.to(mojo_gpu))
    _close(U.cpu().abs(), torch.eye(3), torch.float32)


# --- round-4 review regressions -----------------------------------------------------


def test_svd_norm_ratio_beyond_range_float64(mojo_gpu: str):
    skip_if_metal(mojo_gpu, "Apple GPUs have no float64")
    a = torch.tensor([[1e70, 1e-250], [0.0, 1e-250]], dtype=torch.float64)
    s = torch.linalg.svdvals(a.to(mojo_gpu)).cpu()
    torch.testing.assert_close(s, torch.linalg.svdvals(a), rtol=1e-12, atol=0.0)


def test_svd_float64_accuracy_tall(mojo_gpu: str):
    """Rotating down to dgesvj's sqrt(m) * eps keeps a tall float64 SVD's
    reconstruction within a small factor of LAPACK's."""
    skip_if_metal(mojo_gpu, "Apple GPUs have no float64")
    torch.manual_seed(23)
    a = torch.randn(1000, 50, dtype=torch.float64)
    U, S, Vh = torch.linalg.svd(a.to(mojo_gpu), full_matrices=False)
    err = (U.cpu() @ torch.diag(S.cpu()) @ Vh.cpu() - a).abs().max()
    U0, S0, Vh0 = torch.linalg.svd(a, full_matrices=False)
    cpu_err = (U0 @ torch.diag(S0) @ Vh0 - a).abs().max()
    assert err <= 10 * cpu_err, (err, cpu_err)
