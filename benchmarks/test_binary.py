"""Elementwise binary / ternary / comparison benchmarks.

Overload variants are folded into the layout axis, not into the op token
(design rule: eq / eq.Scalar / eq.Tensor are ONE subject): a comparison
node's layout is "Scalar" or "Tensor", an arithmetic node's layout is
"contig" or "bcast" (second operand a broadcast row — the broadcast
binary kernels are a separately-tuned path).  Shapes are the two
elementwise regimes (bandwidth-bound big vector, launch-bound awkward
2-D) shared with test_elementwise.py.
"""

from __future__ import annotations

import pytest
import torch
from bench_lib.cases import DTYPES, both, op_params, unit_interval
from bench_lib.check import Bench
from bench_lib.hw import Hardware

SHAPES: dict[str, tuple[int, ...]] = {
    "C_4096x4096": (4096, 4096),
    "A_357x789": (357, 789),
}

ARITH_OPS = {
    "add.Tensor": torch.add,
    "sub.Tensor": torch.sub,
    "mul.Tensor": torch.mul,
    "div.Tensor": torch.div,
}
MINMAX_OPS = {"maximum": torch.maximum, "minimum": torch.minimum}
COMPARE_OPS = {
    "eq": torch.eq,
    "ne": torch.ne,
    "ge": torch.ge,
    "gt": torch.gt,
    "le": torch.le,
    "lt": torch.lt,
}
BITWISE_OPS = {
    "bitwise_and": torch.bitwise_and,
    "bitwise_or": torch.bitwise_or,
    "bitwise_xor": torch.bitwise_xor,
}
LOGICAL_OPS = {"logical_and": torch.logical_and, "logical_xor": torch.logical_xor}

COVERS: dict[str, str] = (
    {f"aten::{name}": "test_arith" for name in ARITH_OPS}
    | {f"aten::{name}": "test_minmax" for name in MINMAX_OPS}
    | {
        f"aten::{name}{variant}": "test_compare"
        for name in COMPARE_OPS
        for variant in ("", ".Scalar", ".Tensor")
    }
    | {
        f"aten::{name}.{variant}": "test_bitwise"
        for name in BITWISE_OPS
        for variant in ("Scalar", "Tensor")
    }
    | {f"aten::{name}": "test_logical" for name in LOGICAL_OPS}
    | {
        "aten::pow.Tensor_Scalar": "test_pow[Scalar]",
        "aten::pow.Tensor_Tensor": "test_pow[Tensor]",
        "aten::floor_divide": "test_floor_divide[Tensor]",
        "aten::floor_divide.Scalar": "test_floor_divide[Scalar]",
        "aten::floordiv": "test_floor_divide (same fast impl, alias entry)",
        "aten::remainder.Tensor": "test_remainder[Tensor]",
        "aten::remainder.Scalar": "test_remainder[Scalar]",
        "aten::remainder.Scalar_Tensor": (
            "test_remainder (same kernel, scalar lhs plumbing)"
        ),
        "aten::lerp.Scalar": "test_lerp",
        "aten::clamp": "test_clamp",
        "aten::addcdiv": "test_addcdiv",
        "aten::addcmul": "test_addcmul",
        "aten::where.self": "test_where",
        "aten::masked_fill.Scalar": "test_masked_fill[Scalar]",
        "aten::masked_fill.Tensor": "test_masked_fill[Tensor]",
        "aten::isin.Tensor_Tensor": "test_isin",
    }
)

SKIPPED: dict[str, str] = {}


def _pair(
    shape_id: str, dtype_id: str, layout: str, hw: Hardware, mojo: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(a_ref, b_ref, a_our, b_our); layout "bcast" makes b a broadcast row."""
    shape = SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    a_ref, a_our = both(unit_interval(shape, dtype), hw, mojo)
    b_shape = (shape[-1],) if layout == "bcast" else shape
    b_ref, b_our = both(unit_interval(b_shape, dtype), hw, mojo)
    return a_ref, b_ref, a_our, b_our


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("layout", ("contig", "bcast"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.parametrize("op_name", op_params(ARITH_OPS))
def test_arith(
    op_name: str,
    shape_id: str,
    layout: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
) -> None:
    fn = ARITH_OPS[op_name]
    a_ref, b_ref, a_our, b_our = _pair(shape_id, dtype_id, layout, hw, mojo_device)
    bench.run(
        lambda: fn(a_ref, b_ref), lambda: fn(a_our, b_our), flops=float(a_ref.numel())
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.parametrize("op_name", op_params(MINMAX_OPS))
def test_minmax(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
) -> None:
    fn = MINMAX_OPS[op_name]
    a_ref, b_ref, a_our, b_our = _pair(shape_id, dtype_id, "contig", hw, mojo_device)
    bench.run(
        lambda: fn(a_ref, b_ref), lambda: fn(a_our, b_our), flops=float(a_ref.numel())
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("layout", ("Scalar", "Tensor"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.parametrize("op_name", op_params(COMPARE_OPS))
def test_compare(
    op_name: str,
    shape_id: str,
    layout: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
) -> None:
    fn = COMPARE_OPS[op_name]
    a_ref, b_ref, a_our, b_our = _pair(shape_id, dtype_id, "contig", hw, mojo_device)
    if layout == "Scalar":
        bench.run(
            lambda: fn(a_ref, 0.5), lambda: fn(a_our, 0.5), flops=float(a_ref.numel())
        )
    else:
        bench.run(
            lambda: fn(a_ref, b_ref),
            lambda: fn(a_our, b_our),
            flops=float(a_ref.numel()),
        )


@pytest.mark.parametrize("dtype_id", ("i32",))
@pytest.mark.parametrize("layout", ("Scalar", "Tensor"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.parametrize("op_name", op_params(BITWISE_OPS))
def test_bitwise(
    op_name: str,
    shape_id: str,
    layout: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
) -> None:
    fn = BITWISE_OPS[op_name]
    shape = SHAPES[shape_id]
    a_ref, a_our = both(
        torch.randint(0, 1 << 30, shape, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    b_ref, b_our = both(
        torch.randint(0, 1 << 30, shape, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    if layout == "Scalar":
        bench.run(
            lambda: fn(a_ref, 21), lambda: fn(a_our, 21), flops=float(a_ref.numel())
        )
    else:
        bench.run(
            lambda: fn(a_ref, b_ref),
            lambda: fn(a_our, b_our),
            flops=float(a_ref.numel()),
        )


@pytest.mark.parametrize("dtype_id", ("bool",))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.parametrize("op_name", op_params(LOGICAL_OPS))
def test_logical(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
) -> None:
    fn = LOGICAL_OPS[op_name]
    shape = SHAPES[shape_id]
    a_ref, a_our = both(torch.rand(shape) < 0.5, hw, mojo_device)
    b_ref, b_our = both(torch.rand(shape) < 0.5, hw, mojo_device)
    bench.run(
        lambda: fn(a_ref, b_ref), lambda: fn(a_our, b_our), flops=float(a_ref.numel())
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("layout", ("Scalar", "Tensor"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("pow")
def test_pow(
    shape_id: str,
    layout: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
) -> None:
    a_ref, b_ref, a_our, b_our = _pair(shape_id, dtype_id, "contig", hw, mojo_device)
    if layout == "Scalar":
        bench.run(
            lambda: torch.pow(a_ref, 1.5),
            lambda: torch.pow(a_our, 1.5),
            flops=float(a_ref.numel()),
        )
    else:
        bench.run(
            lambda: torch.pow(a_ref, b_ref),
            lambda: torch.pow(a_our, b_our),
            flops=float(a_ref.numel()),
        )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("layout", ("Scalar", "Tensor"))
@pytest.mark.parametrize("shape_id", SHAPES)
def test_floor_divide(
    shape_id: str,
    layout: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
) -> None:
    a_ref, b_ref, a_our, b_our = _pair(shape_id, dtype_id, "contig", hw, mojo_device)
    if layout == "Scalar":
        bench.run(
            lambda: torch.floor_divide(a_ref, 0.25),
            lambda: torch.floor_divide(a_our, 0.25),
            flops=float(a_ref.numel()),
        )
    else:
        bench.run(
            lambda: torch.floor_divide(a_ref, b_ref),
            lambda: torch.floor_divide(a_our, b_our),
            flops=float(a_ref.numel()),
        )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("layout", ("Scalar", "Tensor"))
@pytest.mark.parametrize("shape_id", SHAPES)
def test_remainder(
    shape_id: str,
    layout: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
) -> None:
    a_ref, b_ref, a_our, b_our = _pair(shape_id, dtype_id, "contig", hw, mojo_device)
    if layout == "Scalar":
        bench.run(
            lambda: torch.remainder(a_ref, 0.25),
            lambda: torch.remainder(a_our, 0.25),
            flops=float(a_ref.numel()),
        )
    else:
        bench.run(
            lambda: torch.remainder(a_ref, b_ref),
            lambda: torch.remainder(a_our, b_our),
            flops=float(a_ref.numel()),
        )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("lerp.Scalar")
def test_lerp(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    a_ref, b_ref, a_our, b_our = _pair(shape_id, dtype_id, "contig", hw, mojo_device)
    bench.run(
        lambda: torch.lerp(a_ref, b_ref, 0.3),
        lambda: torch.lerp(a_our, b_our, 0.3),
        flops=float(a_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
def test_clamp(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape = SHAPES[shape_id]
    a_ref, a_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.clamp(a_ref, 0.2, 0.8),
        lambda: torch.clamp(a_our, 0.2, 0.8),
        flops=float(a_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
def test_addcdiv(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape = SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    a_ref, a_our = both(unit_interval(shape, dtype), hw, mojo_device)
    t1_ref, t1_our = both(unit_interval(shape, dtype), hw, mojo_device)
    t2_ref, t2_our = both(unit_interval(shape, dtype), hw, mojo_device)
    bench.run(
        lambda: torch.addcdiv(a_ref, t1_ref, t2_ref, value=0.5),
        lambda: torch.addcdiv(a_our, t1_our, t2_our, value=0.5),
        flops=float(a_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
def test_addcmul(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape = SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    a_ref, a_our = both(unit_interval(shape, dtype), hw, mojo_device)
    t1_ref, t1_our = both(unit_interval(shape, dtype), hw, mojo_device)
    t2_ref, t2_our = both(unit_interval(shape, dtype), hw, mojo_device)
    bench.run(
        lambda: torch.addcmul(a_ref, t1_ref, t2_ref, value=0.5),
        lambda: torch.addcmul(a_our, t1_our, t2_our, value=0.5),
        flops=float(a_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("where.self")
def test_where(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape = SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    mask_ref, mask_our = both(torch.rand(shape) < 0.5, hw, mojo_device)
    a_ref, a_our = both(unit_interval(shape, dtype), hw, mojo_device)
    b_ref, b_our = both(unit_interval(shape, dtype), hw, mojo_device)
    bench.run(
        lambda: torch.where(mask_ref, a_ref, b_ref),
        lambda: torch.where(mask_our, a_our, b_our),
        flops=float(a_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("layout", ("Scalar", "Tensor"))
@pytest.mark.parametrize("shape_id", SHAPES)
def test_masked_fill(
    shape_id: str,
    layout: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
) -> None:
    shape = SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    a_ref, a_our = both(unit_interval(shape, dtype), hw, mojo_device)
    mask_ref, mask_our = both(torch.rand(shape) < 0.5, hw, mojo_device)
    if layout == "Scalar":
        bench.run(
            lambda: a_ref.masked_fill(mask_ref, 0.0),
            lambda: a_our.masked_fill(mask_our, 0.0),
            flops=float(a_ref.numel()),
        )
    else:
        v_ref, v_our = both(torch.tensor(0.0, dtype=dtype), hw, mojo_device)
        bench.run(
            lambda: a_ref.masked_fill(mask_ref, v_ref),
            lambda: a_our.masked_fill(mask_our, v_our),
            flops=float(a_ref.numel()),
        )


@pytest.mark.parametrize("dtype_id", ("i64",))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("isin.Tensor_Tensor")
def test_isin(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape = SHAPES[shape_id]
    elements = torch.randint(0, 4096, shape, dtype=DTYPES[dtype_id])
    tests = torch.randint(0, 4096, (512,), dtype=DTYPES[dtype_id])
    e_ref, e_our = both(elements, hw, mojo_device)
    t_ref, t_our = both(tests, hw, mojo_device)
    bench.run(
        lambda: torch.isin(e_ref, t_ref),
        lambda: torch.isin(e_our, t_our),
        flops=float(e_ref.numel()),
    )
