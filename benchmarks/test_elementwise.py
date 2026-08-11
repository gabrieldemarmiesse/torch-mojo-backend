"""Elementwise unary benchmarks: one node per (op, dtype, shape).

Every op here is a single memory-bound kernel; two shapes cover the two
regimes that matter: C_16777216 is a large contiguous vector (bandwidth
bound) and A_357x789 is small and awkward (launch/tail bound).  Operands
live in (0.05, 0.95) so one generator serves every op's domain
(acos/atanh need |x| < 1, log/rsqrt need x > 0).

The op axis carries a bench_op mark per param, so the baseline tree path
of node test_unary[abs-C_16777216-bf16] is bf16/abs/C_16777216/contig.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from bench_lib.cases import DTYPES, both, op_params, unit_interval
from bench_lib.check import Bench
from bench_lib.hw import Hardware

SHAPES: dict[str, tuple[int, ...]] = {
    "C_16777216": (16777216,),
    "A_357x789": (357, 789),
}

UNARY_OPS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "abs": torch.abs,
    "acos": torch.acos,
    "asinh": torch.asinh,
    "atanh": torch.atanh,
    "ceil": torch.ceil,
    "cos": torch.cos,
    "cosh": torch.cosh,
    "erf": torch.erf,
    "exp": torch.exp,
    "floor": torch.floor,
    "gelu": F.gelu,
    "isnan": torch.isnan,
    "log": torch.log,
    "log1p": torch.log1p,
    "neg": torch.neg,
    "reciprocal": torch.reciprocal,
    "relu": torch.relu,
    "rsqrt": torch.rsqrt,
    "sigmoid": torch.sigmoid,
    "sign": torch.sign,
    "silu": F.silu,
    "sin": torch.sin,
    "sinh": torch.sinh,
    "sqrt": torch.sqrt,
    "tan": torch.tan,
    "tanh": torch.tanh,
}

# Registered elementwise ops NOT benchmarked here, and why.  Reconciled
# against the live registration table by test_coverage.py.
SKIPPED: dict[str, str] = {}

COVERS: dict[str, str] = {f"aten::{name}": "test_unary" for name in UNARY_OPS} | {
    "aten::bitwise_not": "test_bitwise_not",
    "aten::logical_not": "test_logical_not",
}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.parametrize("op_name", op_params(UNARY_OPS))
def test_unary(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
) -> None:
    fn = UNARY_OPS[op_name]
    shape = SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(lambda: fn(x_ref), lambda: fn(x_our), flops=float(x_ref.numel()))


@pytest.mark.parametrize("dtype_id", ("i32",))
@pytest.mark.parametrize("shape_id", SHAPES)
def test_bitwise_not(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape = SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randint(-1000, 1000, shape, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: torch.bitwise_not(x_ref),
        lambda: torch.bitwise_not(x_our),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bool",))
@pytest.mark.parametrize("shape_id", SHAPES)
def test_logical_not(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape = SHAPES[shape_id]
    x_ref, x_our = both(torch.rand(shape) < 0.5, hw, mojo_device)
    bench.run(
        lambda: torch.logical_not(x_ref),
        lambda: torch.logical_not(x_our),
        flops=float(x_ref.numel()),
    )
