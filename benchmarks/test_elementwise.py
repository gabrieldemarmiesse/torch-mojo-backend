"""Elementwise unary benchmarks: one node per (op, dtype, shape).

Every op here is a single memory-bound kernel; two shapes cover the two
regimes that matter: C_16777216 is a large contiguous vector (bandwidth
bound) and A_357x789 is small and awkward (launch/tail bound).  Operands
live in (0.05, 0.95) so one generator serves every op's domain
(acos/atanh need |x| < 1, log/rsqrt need x > 0).

The op axis carries a bench_op mark per param, so the baseline tree path
of node test_unary[abs-C_16777216-bf16] is bf16/abs/C_16777216/contig.
The separate test_unary_unaligned adds offset views without renaming any
existing contiguous benchmark node or baseline key.
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
    "log2": torch.log2,
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
    "aten::sqrt.out": "test_sqrt_contiguous_out",
}


@pytest.mark.parametrize("dtype_id", ("f16", "bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.parametrize("op_name", op_params(UNARY_OPS))
def test_unary(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    fn = UNARY_OPS[op_name]
    shape = SHAPES[shape_id]
    cpu = unit_interval(shape, DTYPES[dtype_id])
    x_ref, x_our = both(cpu, hw, mojo_device)
    for value in (x_ref, x_our):
        assert value.data_ptr() % (4 * cpu.element_size()) == 0
    torch.testing.assert_close(fn(x_our).cpu(), fn(cpu))
    bench.run(lambda: fn(x_ref), lambda: fn(x_our), flops=float(x_ref.numel()))


@pytest.mark.parametrize("layout", ("offset_1",))
@pytest.mark.parametrize("dtype_id", ("f16", "bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.parametrize("op_name", op_params(UNARY_OPS))
def test_unary_unaligned(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    layout: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    fn = UNARY_OPS[op_name]
    shape = SHAPES[shape_id]
    cpu = unit_interval(shape, DTYPES[dtype_id])
    # Slice after the device copy: copying a CPU view would realign it.
    storage = torch.cat((torch.zeros(1, dtype=cpu.dtype), cpu.flatten()))
    ref_storage, our_storage = both(storage, hw, mojo_device)
    x_ref, x_our = (value[1:].view(shape) for value in (ref_storage, our_storage))
    for value in (x_ref, x_our):
        assert value.is_contiguous()
        assert value.data_ptr() % (4 * cpu.element_size()) == cpu.element_size()
    torch.testing.assert_close(fn(x_our).cpu(), fn(cpu))
    bench.run(lambda: fn(x_ref), lambda: fn(x_our), flops=float(cpu.numel()))


@pytest.mark.bench_op("gelu")
@pytest.mark.parametrize("layout", ("contig", "offset_1"))
@pytest.mark.parametrize("dtype_id", ("f16", "bf16", "f32"))
@pytest.mark.parametrize("shape_id", ("C_16777216_tanh", "A_357x789_tanh"))
def test_gelu_tanh(
    shape_id: str,
    dtype_id: str,
    layout: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    # Approximation is a separate benchmark regime, encoded in the shape key
    # without renaming the existing default-GELU baseline entries.
    shape = SHAPES[shape_id.removesuffix("_tanh")]
    cpu = unit_interval(shape, DTYPES[dtype_id])
    if layout == "offset_1":
        storage = torch.cat((torch.zeros(1, dtype=cpu.dtype), cpu.flatten()))
        ref_storage, our_storage = both(storage, hw, mojo_device)
        x_ref, x_our = (value[1:].view(shape) for value in (ref_storage, our_storage))
    else:
        x_ref, x_our = both(cpu, hw, mojo_device)
    for value in (x_ref, x_our):
        expected_offset = cpu.element_size() if layout == "offset_1" else 0
        assert value.data_ptr() % (4 * cpu.element_size()) == expected_offset
    torch.testing.assert_close(
        F.gelu(x_our, approximate="tanh").cpu(), F.gelu(cpu, approximate="tanh")
    )
    bench.run(
        lambda: F.gelu(x_ref, approximate="tanh"),
        lambda: F.gelu(x_our, approximate="tanh"),
        flops=float(cpu.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("i32",))
@pytest.mark.parametrize("shape_id", SHAPES)
def test_bitwise_not(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
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
):
    shape = SHAPES[shape_id]
    x_ref, x_our = both(torch.rand(shape) < 0.5, hw, mojo_device)
    bench.run(
        lambda: torch.logical_not(x_ref),
        lambda: torch.logical_not(x_our),
        flops=float(x_ref.numel()),
    )


_SQRT_OUT_SHAPES = {
    "small_800": (800, 0),
    "l2_3840000": (3840000, 0),
    "l2_5120000": (5120000, 0),
    "stream_40206400": (40206400, 0),
    "awkward_281673": (357 * 789, 0),
    "offset_281673": (357 * 789, 1),
}


@pytest.mark.bench_op("sqrt.out")
@pytest.mark.parametrize("dtype_id", ["f32"])
@pytest.mark.parametrize("shape_id", _SQRT_OUT_SHAPES)
def test_sqrt_contiguous_out(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    count, offset = _SQRT_OUT_SHAPES[shape_id]
    host = ((torch.arange(count + 16, dtype=torch.int64) * 7919 + 13) % 65521).to(
        DTYPES[dtype_id]
    ) / 4096
    ref_base, our_base = both(host, hw, mojo_device)
    ref_storage, our_storage = torch.empty_like(ref_base), torch.empty_like(our_base)
    ref, our = ref_base[offset : offset + count], our_base[offset : offset + count]
    ref_out = ref_storage[offset : offset + count]
    our_out = our_storage[offset : offset + count]
    bench.run(
        lambda: torch.sqrt(ref, out=ref_out),
        lambda: torch.sqrt(our, out=our_out),
        flops=float(count),
    )
