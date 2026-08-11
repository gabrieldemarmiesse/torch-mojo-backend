"""Shared operand plumbing for the per-family benchmark modules.

Operands are generated ONCE on the CPU (under the seeded
deterministic_operands fixture) and copied to both legs, so the stock
backend and the mojo device always compute on bit-identical data — any
data-dependent kernel timing then cancels out of the ratio.

The op axis of a family test is a parametrize axis whose params carry a
@pytest.mark.bench_op mark (see op_params), so one test function can
cover a whole family while every node still stores its ratio under its
own aten op token in baselines.html.
"""

from __future__ import annotations

import pytest
import torch

from bench_lib.hw import Hardware

# dtype axis tokens shared by every family module.
DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
    "i32": torch.int32,
    "i64": torch.int64,
    "bool": torch.bool,
}


def both(
    cpu_tensor: torch.Tensor, hw: Hardware, mojo: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """The same operand on the stock leg and on the mojo leg: (ref, ours)."""
    return cpu_tensor.to(hw.stock_device), cpu_tensor.to(mojo)


def both_list(
    cpu_tensors: list[torch.Tensor], hw: Hardware, mojo: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    refs, ours = [], []
    for tensor in cpu_tensors:
        ref, our = both(tensor, hw, mojo)
        refs.append(ref)
        ours.append(our)
    return refs, ours


def unit_interval(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Floats in (0.05, 0.95): inside the domain of every benchmarked unary
    op (acos/atanh need |x|<1, log/sqrt/rsqrt need x>0) and safely away
    from division-by-zero for the ratio ops."""
    return (torch.rand(shape, dtype=torch.float32) * 0.9 + 0.05).to(dtype)


def op_params(ops: dict[str, object]) -> list[object]:
    """One pytest.param per op, marked with bench_op so the baseline tree
    stores the case under the aten op token rather than the test name."""
    return [
        pytest.param(name, marks=pytest.mark.bench_op(name), id=name) for name in ops
    ]
