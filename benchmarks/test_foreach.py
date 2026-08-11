"""Foreach / fused-optimizer benchmarks.

These are the optimizer-step ops: one node measures the whole list call
(the subject is the horizontally-fused launch, not one tensor).  Shape
tokens name the list: L_4x1048576 is four 1M-element tensors (few big
params), L_16x65536 is sixteen 64k tensors (many small params — the
launch-overhead regime).  All f32, matching optimizer state in practice.

In-place list ops mutate their operands across iterations; operand
values are chosen to stay finite (multipliers near 1, tiny addends), and
the kernels are data-oblivious so drift cannot skew timing.
"""

from __future__ import annotations

import pytest
import torch
from bench_lib.cases import DTYPES, both_list, unit_interval
from bench_lib.check import Bench
from bench_lib.hw import Hardware

LISTS: dict[str, list[tuple[int, ...]]] = {
    "L_4x1048576": [(1048576,)] * 4,
    "L_16x65536": [(65536,)] * 16,
}

COVERS: dict[str, str] = {
    "aten::_foreach_add_.Scalar": "test_foreach_add_",
    "aten::_foreach_addcdiv_.ScalarList": "test_foreach_addcdiv_",
    "aten::_foreach_addcmul_.Scalar": "test_foreach_addcmul_",
    "aten::_foreach_div_.ScalarList": "test_foreach_div_",
    "aten::_foreach_lerp_.Scalar": "test_foreach_lerp_",
    "aten::_foreach_mul_.Scalar": "test_foreach_mul_scalar",
    "aten::_foreach_mul_.Tensor": "test_foreach_mul_tensor",
    "aten::_foreach_norm.Scalar": "test_foreach_norm",
    "aten::_foreach_sqrt": "test_foreach_sqrt",
    "aten::_fused_adamw_": "test_fused_adamw",
    "aten::_fused_adamw_.tensor_lr": (
        "test_fused_adamw (same fast impl; lr-as-tensor plumbing only)"
    ),
}

SKIPPED: dict[str, str] = {}


def _lists(
    shape_id: str, hw: Hardware, mojo: torch.device, count: int = 1
) -> list[tuple[list[torch.Tensor], list[torch.Tensor]]]:
    """`count` independent (ref_list, our_list) operand lists."""
    dtype = DTYPES["f32"]
    out = []
    for _ in range(count):
        cpu = [unit_interval(shape, dtype) + 0.05 for shape in LISTS[shape_id]]
        out.append(both_list(cpu, hw, mojo))
    return out


def _total(shape_id: str) -> float:
    return float(sum(torch.Size(s).numel() for s in LISTS[shape_id]))


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", LISTS)
@pytest.mark.bench_op("_foreach_add_.Scalar")
def test_foreach_add_(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    ((ref, our),) = _lists(shape_id, hw, mojo_device)
    bench.run(
        lambda: torch._foreach_add_(ref, 1e-5),
        lambda: torch._foreach_add_(our, 1e-5),
        flops=_total(shape_id),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", LISTS)
@pytest.mark.bench_op("_foreach_addcdiv_.ScalarList")
def test_foreach_addcdiv_(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    (ref, our), (t1_ref, t1_our), (t2_ref, t2_our) = _lists(
        shape_id, hw, mojo_device, count=3
    )
    scalars = [1e-4] * len(ref)
    bench.run(
        lambda: torch._foreach_addcdiv_(ref, t1_ref, t2_ref, scalars),
        lambda: torch._foreach_addcdiv_(our, t1_our, t2_our, scalars),
        flops=_total(shape_id),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", LISTS)
@pytest.mark.bench_op("_foreach_addcmul_.Scalar")
def test_foreach_addcmul_(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    (ref, our), (t1_ref, t1_our), (t2_ref, t2_our) = _lists(
        shape_id, hw, mojo_device, count=3
    )
    bench.run(
        lambda: torch._foreach_addcmul_(ref, t1_ref, t2_ref, value=1e-4),
        lambda: torch._foreach_addcmul_(our, t1_our, t2_our, value=1e-4),
        flops=_total(shape_id),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", LISTS)
@pytest.mark.bench_op("_foreach_div_.ScalarList")
def test_foreach_div_(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    ((ref, our),) = _lists(shape_id, hw, mojo_device)
    scalars = [1.0001] * len(ref)
    bench.run(
        lambda: torch._foreach_div_(ref, scalars),
        lambda: torch._foreach_div_(our, scalars),
        flops=_total(shape_id),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", LISTS)
@pytest.mark.bench_op("_foreach_lerp_.Scalar")
def test_foreach_lerp_(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    (ref, our), (end_ref, end_our) = _lists(shape_id, hw, mojo_device, count=2)
    bench.run(
        lambda: torch._foreach_lerp_(ref, end_ref, 0.01),
        lambda: torch._foreach_lerp_(our, end_our, 0.01),
        flops=_total(shape_id),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", LISTS)
@pytest.mark.bench_op("_foreach_mul_.Scalar")
def test_foreach_mul_scalar(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    ((ref, our),) = _lists(shape_id, hw, mojo_device)
    bench.run(
        lambda: torch._foreach_mul_(ref, 1.0001),
        lambda: torch._foreach_mul_(our, 1.0001),
        flops=_total(shape_id),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", LISTS)
@pytest.mark.bench_op("_foreach_mul_.Tensor")
def test_foreach_mul_tensor(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    ((ref, our),) = _lists(shape_id, hw, mojo_device)
    m_ref, m_our = (
        torch.tensor(1.0001).to(hw.stock_device),
        torch.tensor(1.0001).to(mojo_device),
    )
    bench.run(
        lambda: torch._foreach_mul_(ref, m_ref),
        lambda: torch._foreach_mul_(our, m_our),
        flops=_total(shape_id),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", LISTS)
@pytest.mark.bench_op("_foreach_norm.Scalar")
def test_foreach_norm(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    ((ref, our),) = _lists(shape_id, hw, mojo_device)
    bench.run(
        lambda: torch._foreach_norm(ref, 2.0),
        lambda: torch._foreach_norm(our, 2.0),
        flops=_total(shape_id),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", LISTS)
@pytest.mark.bench_op("_foreach_sqrt")
def test_foreach_sqrt(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    ((ref, our),) = _lists(shape_id, hw, mojo_device)
    bench.run(
        lambda: torch._foreach_sqrt(ref),
        lambda: torch._foreach_sqrt(our),
        flops=_total(shape_id),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", LISTS)
@pytest.mark.bench_op("_fused_adamw_")
def test_fused_adamw(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    (
        (params_ref, params_our),
        (grads_ref, grads_our),
        (avg_ref, avg_our),
        (sq_ref, sq_our),
    ) = _lists(shape_id, hw, mojo_device, count=4)
    steps_cpu = [torch.tensor(1.0) for _ in params_ref]
    steps_ref = [s.to(hw.stock_device) for s in steps_cpu]
    steps_our = [s.to(mojo_device) for s in steps_cpu]

    def step(params, grads, avgs, sqs, steps) -> None:
        torch.ops.aten._fused_adamw_(
            params,
            grads,
            avgs,
            sqs,
            [],
            steps,
            lr=1e-3,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.01,
            eps=1e-8,
            amsgrad=False,
            maximize=False,
        )

    bench.run(
        lambda: step(params_ref, grads_ref, avg_ref, sq_ref, steps_ref),
        lambda: step(params_our, grads_our, avg_our, sq_our, steps_our),
        flops=_total(shape_id),
    )
