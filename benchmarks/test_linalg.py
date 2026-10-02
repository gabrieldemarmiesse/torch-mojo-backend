"""Dense linear algebra: the factorizations of the `linalg` kernel family
(tmb/kernels/linalg) and the solves composed on them.

Each factorization is driven through its torch.ops.aten entry point. The
reference leg is stock PyTorch's (cuSOLVER / MAGMA on CUDA), so these nodes
measure one thread block per matrix against vendor solvers. Shapes: a batch
of small matrices (the regime one block per matrix is built for), one
mid-size matrix, and an awkward size.
"""

from __future__ import annotations

import pytest
import torch
from bench_lib.cases import DTYPES, both
from bench_lib.check import Bench
from bench_lib.hw import Hardware

SHAPES: dict[str, tuple[int, ...]] = {
    "B_512x8x8": (512, 8, 8),
    "S_256x256": (256, 256),
    "A_37x37": (37, 37),
}

COVERS: dict[str, str] = {
    "aten::linalg_cholesky_ex": "test_cholesky",
    "aten::linalg_lu_factor_ex": "test_lu_factor",
    "aten::linalg_lu_solve": "test_lu_solve",
    "aten::linalg_solve_triangular": "test_solve_triangular",
    "aten::geqrf": "test_geqrf",
    "aten::linalg_householder_product": "test_householder_product",
    "aten::_linalg_eigh": "test_eigh",
    "aten::_linalg_svd": "test_svd",
    "aten::linalg_ldl_factor_ex": "test_ldl_factor",
}

_OUT = "out= plumbing over the functional linalg op this module measures"
_COMPOSED = (
    "composed on the factorizations and solves this module measures "
    "(tmb/ops/linalg.mojo), no kernel of its own"
)
SKIPPED: dict[str, str] = {
    "aten::linalg_cholesky_ex.L": _OUT,
    "aten::linalg_lu_factor_ex.out": _OUT,
    "aten::linalg_lu_solve.out": _OUT,
    "aten::linalg_solve_triangular.out": _OUT,
    "aten::geqrf.a": _OUT,
    "aten::linalg_householder_product.out": _OUT,
    "aten::_linalg_eigh.eigenvalues": _OUT,
    "aten::_linalg_svd.U": _OUT,
    "aten::linalg_ldl_factor_ex.out": _OUT,
    "aten::cholesky": _COMPOSED,
    "aten::cholesky.out": _COMPOSED,
    "aten::cholesky_inverse": _COMPOSED,
    "aten::cholesky_inverse.out": _COMPOSED,
    "aten::_cholesky_solve_helper": _COMPOSED,
    "aten::linalg_lu": _COMPOSED,
    "aten::linalg_lu.out": _COMPOSED,
    "aten::lu_unpack": _COMPOSED,
    "aten::lu_unpack.out": _COMPOSED,
    "aten::linalg_inv_ex": _COMPOSED,
    "aten::linalg_inv_ex.inverse": _COMPOSED,
    "aten::_linalg_solve_ex": _COMPOSED,
    "aten::_linalg_solve_ex.result": _COMPOSED,
    "aten::_linalg_det": _COMPOSED,
    "aten::_linalg_det.result": _COMPOSED,
    "aten::_linalg_slogdet": _COMPOSED,
    "aten::_linalg_slogdet.sign": _COMPOSED,
    "aten::triangular_solve": _COMPOSED,
    "aten::triangular_solve.X": _COMPOSED,
    "aten::ormqr": "the Ormqr kernel test_householder_product measures",
    "aten::ormqr.out": _OUT,
    "aten::linalg_qr": _COMPOSED,
    "aten::linalg_qr.out": _COMPOSED,
    "aten::linalg_lstsq.out": _COMPOSED,
    "aten::linalg_ldl_solve": (
        "one thread per right-hand side over the Sytf2 factorization "
        "test_ldl_factor measures; a backward/forward substitution"
    ),
    "aten::linalg_ldl_solve.out": _OUT,
    "aten::linalg_matrix_exp": (
        "composed of the GEMMs and elementwise ops test_gemm and "
        "test_elementwise measure (torch's mexp)"
    ),
    "aten::linalg_matrix_exp.out": _OUT,
}


def _spd(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    a = torch.randn(shape, dtype=torch.float64)
    n = shape[-1]
    return (a @ a.mT + n * torch.eye(n, dtype=torch.float64)).to(dtype)


def _sym(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    a = torch.randn(shape, dtype=dtype)
    return a + a.mT


def _flops(shape: tuple[int, ...], per_matrix: float) -> float:
    batch = 1
    for d in shape[:-2]:
        batch *= d
    return batch * per_matrix


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("linalg_cholesky_ex")
def test_cholesky(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = SHAPES[shape_id]
    a_ref, a_our = both(_spd(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.linalg_cholesky_ex(a_ref),
        lambda: torch.ops.aten.linalg_cholesky_ex(a_our),
        flops=_flops(shape, shape[-1] ** 3 / 3),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("linalg_lu_factor_ex")
def test_lu_factor(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = SHAPES[shape_id]
    a_ref, a_our = both(torch.randn(shape, dtype=DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.linalg_lu_factor_ex(a_ref),
        lambda: torch.ops.aten.linalg_lu_factor_ex(a_our),
        flops=_flops(shape, 2 * shape[-1] ** 3 / 3),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("linalg_lu_solve")
def test_lu_solve(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    a = torch.randn(shape, dtype=dtype)
    lu, piv, _ = torch.linalg.lu_factor_ex(a)
    b = torch.randn(shape, dtype=dtype)
    lu_ref, lu_our = both(lu, hw, mojo_device)
    p_ref, p_our = both(piv, hw, mojo_device)
    b_ref, b_our = both(b, hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.linalg_lu_solve(lu_ref, p_ref, b_ref),
        lambda: torch.ops.aten.linalg_lu_solve(lu_our, p_our, b_our),
        flops=_flops(shape, 2 * shape[-1] ** 3),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("linalg_solve_triangular")
def test_solve_triangular(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    n = shape[-1]
    a = torch.randn(shape, dtype=dtype).triu() + n * torch.eye(n, dtype=dtype)
    a_ref, a_our = both(a, hw, mojo_device)
    b_ref, b_our = both(torch.randn(shape, dtype=dtype), hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.linalg_solve_triangular(a_ref, b_ref, upper=True),
        lambda: torch.ops.aten.linalg_solve_triangular(a_our, b_our, upper=True),
        flops=_flops(shape, n**3),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("geqrf")
def test_geqrf(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = SHAPES[shape_id]
    a_ref, a_our = both(torch.randn(shape, dtype=DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.geqrf(a_ref),
        lambda: torch.ops.aten.geqrf(a_our),
        flops=_flops(shape, 4 * shape[-1] ** 3 / 3),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("linalg_householder_product")
def test_householder_product(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = SHAPES[shape_id]
    qr, tau = torch.geqrf(torch.randn(shape, dtype=DTYPES[dtype_id]))
    q_ref, q_our = both(qr, hw, mojo_device)
    t_ref, t_our = both(tau, hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.linalg_householder_product(q_ref, t_ref),
        lambda: torch.ops.aten.linalg_householder_product(q_our, t_our),
        flops=_flops(shape, 4 * shape[-1] ** 3 / 3),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("_linalg_eigh")
def test_eigh(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = SHAPES[shape_id]
    a_ref, a_our = both(_sym(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten._linalg_eigh(a_ref),
        lambda: torch.ops.aten._linalg_eigh(a_our),
        flops=_flops(shape, 9 * shape[-1] ** 3),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("_linalg_svd")
def test_svd(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = SHAPES[shape_id]
    a_ref, a_our = both(torch.randn(shape, dtype=DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten._linalg_svd(a_ref),
        lambda: torch.ops.aten._linalg_svd(a_our),
        flops=_flops(shape, 12 * shape[-1] ** 3),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("linalg_ldl_factor_ex")
def test_ldl_factor(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = SHAPES[shape_id]
    a_ref, a_our = both(_sym(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.linalg_ldl_factor_ex(a_ref),
        lambda: torch.ops.aten.linalg_ldl_factor_ex(a_our),
        flops=_flops(shape, shape[-1] ** 3 / 3),
    )
