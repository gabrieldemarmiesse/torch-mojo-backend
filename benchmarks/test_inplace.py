"""In-place elementwise benchmarks: add_, mul_, relu_, fill_, masked_fill_.

The in-place ops mutate their input across iterations, so operand values
are chosen to stay numerically tame over thousands of calls (mul_ by
values around 1.0, add_ of small increments); the kernels are
data-oblivious so the drift cannot skew the timing, both legs mutate
their own copy of identical starting data, and the seeded fixture
regenerates the operands per node.

fill_.Scalar here is also the measurement for the whole alloc+fill
family (zeros / ones / full / *_like / new_* / zero_ / fill.Scalar all
delegate to the same fill kernel; see test_coverage.py SKIPPED notes).
"""

from __future__ import annotations

import pytest
import torch
from bench_lib.cases import DTYPES, both, unit_interval
from bench_lib.check import Bench
from bench_lib.hw import Hardware

SHAPES: dict[str, tuple[int, ...]] = {
    "C_16777216": (16777216,),
    "A_357x789": (357, 789),
}

COVERS: dict[str, str] = {
    "aten::add_.Tensor": "test_add_",
    "aten::mul_.Tensor": "test_mul_",
    "aten::relu_": "test_relu_",
    "aten::fill_.Scalar": "test_fill_",
    "aten::masked_fill_.Scalar": "test_masked_fill_[Scalar]",
    "aten::masked_fill_.Tensor": "test_masked_fill_[Tensor]",
}

SKIPPED: dict[str, str] = {}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("add_.Tensor")
def test_add_(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape = SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(unit_interval(shape, dtype), hw, mojo_device)
    d_ref, d_our = both((unit_interval(shape, dtype) - 0.5) * 1e-4, hw, mojo_device)
    bench.run(
        lambda: x_ref.add_(d_ref), lambda: x_our.add_(d_our), flops=float(x_ref.numel())
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("mul_.Tensor")
def test_mul_(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape = SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(unit_interval(shape, dtype), hw, mojo_device)
    # Multipliers hugging 1.0: thousands of in-place iterations stay finite.
    m_ref, m_our = both(
        (1.0 + (torch.rand(shape) - 0.5) * 1e-4).to(dtype), hw, mojo_device
    )
    bench.run(
        lambda: x_ref.mul_(m_ref), lambda: x_our.mul_(m_our), flops=float(x_ref.numel())
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("relu_")
def test_relu_(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape = SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(lambda: x_ref.relu_(), lambda: x_our.relu_(), flops=float(x_ref.numel()))


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("fill_.Scalar")
def test_fill_(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape = SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: x_ref.fill_(0.5), lambda: x_our.fill_(0.5), flops=float(x_ref.numel())
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("layout", ("Scalar", "Tensor"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("masked_fill_")
def test_masked_fill_(
    shape_id: str,
    layout: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
) -> None:
    shape = SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(unit_interval(shape, dtype), hw, mojo_device)
    mask_ref, mask_our = both(torch.rand(shape) < 0.5, hw, mojo_device)
    if layout == "Scalar":
        bench.run(
            lambda: x_ref.masked_fill_(mask_ref, 0.0),
            lambda: x_our.masked_fill_(mask_our, 0.0),
            flops=float(x_ref.numel()),
        )
    else:
        v_ref, v_our = both(torch.tensor(0.0, dtype=dtype), hw, mojo_device)
        bench.run(
            lambda: x_ref.masked_fill_(mask_ref, v_ref),
            lambda: x_our.masked_fill_(mask_our, v_our),
            flops=float(x_ref.numel()),
        )
