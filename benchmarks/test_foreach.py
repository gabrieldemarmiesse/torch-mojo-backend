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
from bench_lib.cases import DTYPES, both, both_list, unit_interval
from bench_lib.check import Bench
from bench_lib.hw import Hardware

LISTS: dict[str, list[tuple[int, ...]]] = {
    "L_4x1048576": [(1048576,)] * 4,
    "L_16x65536": [(65536,)] * 16,
}

COVERS: dict[str, str] = {
    "aten::_amp_foreach_non_finite_check_and_unscale_": (
        "test_amp_non_finite_check_and_unscale_"
    ),
    "aten::_amp_update_scale_": "test_amp_update_scale_",
    "aten::_foreach_copy_": "test_foreach_copy_cast",
    "aten::_foreach_add_.Scalar": "test_foreach_add_",
    "aten::_foreach_addcmul_.Scalar": "test_foreach_addcmul_",
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


COPY_LISTS = {
    "L_mixed_15370400": [
        800,
        800,
        3840000,
        2400,
        1280000,
        800,
        800,
        800,
        5120000,
        3200,
        5120000,
        800,
    ],
    "L_12x800": [800] * 12,
    "L_awkward_357x789": [357 * 789, 7 * 1025, 1025, 17, 1],
    "L_1x1048576": [1048576],
    "L_65x513": [513] * 65,
    "L_empty_mixed": [0, 17, 0, 1025],
}


# (source, destination) of each batched-copy dtype id: the mixed-precision
# cast both ways, and a same-dtype copy of each width class.
COPY_DTYPES: dict[str, tuple[torch.dtype, torch.dtype]] = {
    "f32_to_bf16": (torch.float32, torch.bfloat16),
    "bf16_to_f32": (torch.bfloat16, torch.float32),
    "bf16": (torch.bfloat16, torch.bfloat16),
    "f32": (torch.float32, torch.float32),
}


# Element offsets (source, destination) into each allocation: "contig" puts
# the two at different vector phases, "aligned" starts both on one.
COPY_OFFSETS: dict[str, tuple[int, int]] = {"contig": (3, 5), "aligned": (0, 0)}


@pytest.mark.bench_op("_foreach_copy_")
@pytest.mark.parametrize("dtype_id", COPY_DTYPES)
@pytest.mark.parametrize("layout", COPY_OFFSETS)
@pytest.mark.parametrize("shape_id", COPY_LISTS)
def test_foreach_copy_cast(
    shape_id: str,
    layout: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    """Separate allocations, so same-dtype lists take the batched kernel rather
    than the adjacent-views DMA."""
    sizes = COPY_LISTS[shape_id]
    src_dtype, dst_dtype = COPY_DTYPES[dtype_id]
    src_offset, dst_offset = COPY_OFFSETS[layout]
    src_ref, src_our = both_list(
        [torch.randn(n + 8).to(src_dtype) for n in sizes], hw, mojo_device
    )
    dst_ref, dst_our = both_list(
        [torch.empty(n + 12, dtype=dst_dtype) for n in sizes], hw, mojo_device
    )
    src_ref = [t[src_offset:][:n] for t, n in zip(src_ref, sizes, strict=True)]
    src_our = [t[src_offset:][:n] for t, n in zip(src_our, sizes, strict=True)]
    dst_ref = [t[dst_offset:][:n] for t, n in zip(dst_ref, sizes, strict=True)]
    dst_our = [t[dst_offset:][:n] for t, n in zip(dst_our, sizes, strict=True)]
    bench.run(
        lambda: torch._foreach_copy_(dst_ref, src_ref),
        lambda: torch._foreach_copy_(dst_our, src_our),
        flops=float(sum(sizes)),
    )


@pytest.mark.bench_op("_foreach_copy_")
@pytest.mark.parametrize("dtype_id", ("bf16",))
@pytest.mark.parametrize("layout", ("adjacent_views",))
@pytest.mark.parametrize("shape_id", COPY_LISTS)
def test_foreach_copy_adjacent_views(
    shape_id: str,
    dtype_id: str,
    layout: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    """Pure DMA routes skip if the device timer records only kernels."""
    sizes = COPY_LISTS[shape_id]
    source = unit_interval((sum(sizes) + 8,), DTYPES[dtype_id])
    src_ref, src_our = both(source, hw, mojo_device)
    dst_ref, dst_our = both(torch.empty_like(source), hw, mojo_device)
    srcs_ref, srcs_our = src_ref[3:-5].split(sizes), src_our[3:-5].split(sizes)
    dsts_ref, dsts_our = dst_ref[5:-3].split(sizes), dst_our[5:-3].split(sizes)
    bench.run(
        lambda: torch._foreach_copy_(dsts_ref, srcs_ref),
        lambda: torch._foreach_copy_(dsts_our, srcs_our),
        flops=float(sum(sizes)),
    )


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
@pytest.mark.bench_op("_amp_foreach_non_finite_check_and_unscale_")
def test_amp_non_finite_check_and_unscale_(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    """GradScaler.unscale_: finite grads, so every element is read and scaled."""
    ((ref, our),) = _lists(shape_id, hw, mojo_device)
    found_ref, found_our = both(torch.zeros(()), hw, mojo_device)
    inv_ref, inv_our = both(torch.tensor(0.9999), hw, mojo_device)
    bench.run(
        lambda: torch._amp_foreach_non_finite_check_and_unscale_(
            ref, found_ref, inv_ref
        ),
        lambda: torch._amp_foreach_non_finite_check_and_unscale_(
            our, found_our, inv_our
        ),
        flops=_total(shape_id),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", ("S_scalar",))
@pytest.mark.bench_op("_amp_update_scale_")
def test_amp_update_scale_(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    """GradScaler.update: one thread on three device scalars."""
    scale_ref, scale_our = both(torch.tensor(65536.0), hw, mojo_device)
    tracker_ref, tracker_our = both(torch.zeros((), dtype=torch.int32), hw, mojo_device)
    found_ref, found_our = both(torch.zeros(()), hw, mojo_device)
    bench.run(
        lambda: torch._amp_update_scale_(
            scale_ref, tracker_ref, found_ref, 2.0, 0.5, 2**30
        ),
        lambda: torch._amp_update_scale_(
            scale_our, tracker_our, found_our, 2.0, 0.5, 2**30
        ),
        flops=1.0,
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", LISTS)
@pytest.mark.bench_op("_foreach_add_.Scalar")
def test_foreach_add_(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
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
):
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
):
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
):
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
):
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
):
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
):
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
):
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
):
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
):
    (
        (params_ref, params_our),
        (grads_ref, grads_our),
        (avg_ref, avg_our),
        (sq_ref, sq_our),
    ) = _lists(shape_id, hw, mojo_device, count=4)
    steps_cpu = [torch.tensor(1.0) for _ in params_ref]
    steps_ref = [s.to(hw.stock_device) for s in steps_cpu]
    steps_our = [s.to(mojo_device) for s in steps_cpu]

    def step(
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        avgs: list[torch.Tensor],
        sqs: list[torch.Tensor],
        steps: list[torch.Tensor],
    ):
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
