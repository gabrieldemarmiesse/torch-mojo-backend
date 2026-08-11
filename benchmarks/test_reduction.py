"""Reduction benchmarks.

The reduced-dim axis is folded into the shape token (design rule: any
extra axis an op needs goes into the shape id): S_4096x4096_d0 reduces
the strided dim, S_4096x4096_d1 the contiguous dim, C_16777216_all is the
full reduction of a large vector.  all/any's .dim/.dims overloads and
mean.dim are covered by the same fold.

nonzero uses a fixed 50% density mask from the seeded fixture: the
output-shape host sync is a real cost of the op, but this suite measures
device kernel time only, which is exactly what the design pinned.
"""

from __future__ import annotations

import pytest
import torch
from bench_lib.cases import DTYPES, both, unit_interval
from bench_lib.check import Bench
from bench_lib.hw import Hardware

# shape id -> (tensor shape, reduced dim or None for a full reduction)
DIM_SHAPES: dict[str, tuple[tuple[int, ...], int | None]] = {
    "S_4096x4096_d0": ((4096, 4096), 0),
    "S_4096x4096_d1": ((4096, 4096), 1),
    "C_16777216_all": ((16777216,), None),
}
FULL_SHAPES: dict[str, tuple[int, ...]] = {
    "C_16777216": (16777216,),
    "A_357x789": (357, 789),
}
LASTDIM_SHAPES: dict[str, tuple[tuple[int, ...], int]] = {
    "S_4096x4096_d0": ((4096, 4096), 0),
    "S_4096x4096_d1": ((4096, 4096), 1),
}

COVERS: dict[str, str] = {
    "aten::sum.dim_IntList": "test_sum",
    "aten::mean": "test_mean (full-reduction case)",
    "aten::mean.dim": "test_mean (dim cases)",
    "aten::max": "test_max",
    "aten::min": "test_min",
    "aten::amax": "test_amax",
    "aten::amin": "test_amin",
    "aten::argmax": "test_argmax",
    "aten::argmin": "test_argmin",
    "aten::all": "test_all (full-reduction case)",
    "aten::all.dim": "test_all (dim case)",
    "aten::all.dims": "test_all (same fast impl as .dim)",
    "aten::any": "test_any (full-reduction case)",
    "aten::any.dim": "test_any (dim case)",
    "aten::any.dims": "test_any (same fast impl as .dim)",
    "aten::min.dim": "test_min_dim",
    "aten::var.correction": "test_var",
    "aten::linalg_vector_norm.out": (
        "test_vector_norm (the .out form is the only registered entry; "
        "torch.linalg.vector_norm reaches it)"
    ),
    "aten::cumsum": "test_cumsum",
    "aten::nonzero": "test_nonzero",
}

SKIPPED: dict[str, str] = {}


def _dim_case(
    shape_id: str, dtype_id: str, hw: Hardware, mojo: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int | None]:
    shape, dim = DIM_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo)
    return x_ref, x_our, dim


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
@pytest.mark.bench_op("sum.dim_IntList")
def test_sum(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    x_ref, x_our, dim = _dim_case(shape_id, dtype_id, hw, mojo_device)
    d = 0 if dim is None else dim
    bench.run(
        lambda: torch.sum(x_ref, dim=d),
        lambda: torch.sum(x_our, dim=d),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
def test_mean(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    x_ref, x_our, dim = _dim_case(shape_id, dtype_id, hw, mojo_device)
    if dim is None:
        bench.run(
            lambda: torch.mean(x_ref),
            lambda: torch.mean(x_our),
            flops=float(x_ref.numel()),
        )
    else:
        bench.run(
            lambda: torch.mean(x_ref, dim=dim),
            lambda: torch.mean(x_our, dim=dim),
            flops=float(x_ref.numel()),
        )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", FULL_SHAPES)
def test_max(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    x_ref, x_our = both(
        unit_interval(FULL_SHAPES[shape_id], DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: torch.max(x_ref), lambda: torch.max(x_our), flops=float(x_ref.numel())
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", FULL_SHAPES)
def test_min(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    x_ref, x_our = both(
        unit_interval(FULL_SHAPES[shape_id], DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: torch.min(x_ref), lambda: torch.min(x_our), flops=float(x_ref.numel())
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", LASTDIM_SHAPES)
def test_amax(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape, dim = LASTDIM_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.amax(x_ref, dim=dim),
        lambda: torch.amax(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", LASTDIM_SHAPES)
def test_amin(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape, dim = LASTDIM_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.amin(x_ref, dim=dim),
        lambda: torch.amin(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
def test_argmax(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    x_ref, x_our, dim = _dim_case(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.argmax(x_ref, dim=dim),
        lambda: torch.argmax(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
def test_argmin(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    x_ref, x_our, dim = _dim_case(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.argmin(x_ref, dim=dim),
        lambda: torch.argmin(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bool",))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
def test_all(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape, dim = DIM_SHAPES[shape_id]
    x_ref, x_our = both(torch.rand(shape) < 0.999, hw, mojo_device)
    if dim is None:
        bench.run(
            lambda: torch.all(x_ref),
            lambda: torch.all(x_our),
            flops=float(x_ref.numel()),
        )
    else:
        bench.run(
            lambda: torch.all(x_ref, dim=dim),
            lambda: torch.all(x_our, dim=dim),
            flops=float(x_ref.numel()),
        )


@pytest.mark.parametrize("dtype_id", ("bool",))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
def test_any(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape, dim = DIM_SHAPES[shape_id]
    x_ref, x_our = both(torch.rand(shape) < 0.001, hw, mojo_device)
    if dim is None:
        bench.run(
            lambda: torch.any(x_ref),
            lambda: torch.any(x_our),
            flops=float(x_ref.numel()),
        )
    else:
        bench.run(
            lambda: torch.any(x_ref, dim=dim),
            lambda: torch.any(x_our, dim=dim),
            flops=float(x_ref.numel()),
        )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", LASTDIM_SHAPES)
@pytest.mark.bench_op("min.dim")
def test_min_dim(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape, dim = LASTDIM_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.min(x_ref, dim=dim),
        lambda: torch.min(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DIM_SHAPES)
@pytest.mark.bench_op("var.correction")
def test_var(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    x_ref, x_our, dim = _dim_case(shape_id, dtype_id, hw, mojo_device)
    if dim is None:
        bench.run(
            lambda: torch.var(x_ref, correction=1),
            lambda: torch.var(x_our, correction=1),
            flops=float(x_ref.numel()),
        )
    else:
        bench.run(
            lambda: torch.var(x_ref, dim=dim, correction=1),
            lambda: torch.var(x_our, dim=dim, correction=1),
            flops=float(x_ref.numel()),
        )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", FULL_SHAPES)
@pytest.mark.bench_op("linalg_vector_norm")
def test_vector_norm(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    x_ref, x_our = both(
        unit_interval(FULL_SHAPES[shape_id], DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: torch.linalg.vector_norm(x_ref),
        lambda: torch.linalg.vector_norm(x_our),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", LASTDIM_SHAPES)
def test_cumsum(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape, dim = LASTDIM_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.cumsum(x_ref, dim=dim),
        lambda: torch.cumsum(x_our, dim=dim),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bool",))
@pytest.mark.parametrize("shape_id", FULL_SHAPES)
def test_nonzero(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    # Fixed 50% density under the seeded fixture: the output size, and so
    # the kernel work, is identical on both legs and across runs.
    x_ref, x_our = both(torch.rand(FULL_SHAPES[shape_id]) < 0.5, hw, mojo_device)
    bench.run(
        lambda: torch.nonzero(x_ref),
        lambda: torch.nonzero(x_our),
        flops=float(x_ref.numel()),
    )
