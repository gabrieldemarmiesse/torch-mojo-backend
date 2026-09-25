"""Detection ROI sampling, pooling and IoU suppression device times."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType

import pytest
import torch
from bench_lib.cases import DTYPES, both
from bench_lib.check import Bench
from bench_lib.hw import Hardware

# N, C, H, W, K, pooled H, pooled W, sampling ratio.
ALIGN_SHAPES = {
    "N2C256H200W272K1000_o7_s2": (2, 256, 200, 272, 1000, 7, 7, 2),
    "N2C256H200W272K1000_o14_s2": (2, 256, 200, 272, 1000, 14, 14, 2),
    "N2C256H200W272K1000_o7_adaptive": (2, 256, 200, 272, 1000, 7, 7, -1),
    "N2C256H200W272K1000_o14_adaptive": (2, 256, 200, 272, 1000, 14, 14, -1),
    "N3C7H37W53K19_o7x5_s2": (3, 7, 37, 53, 19, 7, 5, 2),
}
POOL_SHAPES = {name: shape for name, shape in ALIGN_SHAPES.items() if shape[-1] == 2}
NMS_SHAPES = {"K1000": 1000, "K5000": 5000, "K20000": 20000}

COVERS = {
    "torchvision::nms": "test_nms",
    "torchvision::roi_align": "test_roi_align",
    "torchvision::_roi_align_backward": "test_roi_align_backward",
    "torchvision::roi_pool": "test_roi_pool",
    "torchvision::_roi_pool_backward": "test_roi_pool_backward",
}
COVERS.update(
    {
        "torchvision::ps_roi_align": "test_ps_roi_align",
        "torchvision::_ps_roi_align_backward": "test_ps_roi_align_backward",
        "torchvision::ps_roi_pool": "test_ps_roi_pool",
        "torchvision::_ps_roi_pool_backward": "test_ps_roi_pool_backward",
        "torchvision::deform_conv2d": "test_deform_conv2d",
        "torchvision::_deform_conv2d_backward": "test_deform_conv2d_backward",
    }
)
SKIPPED = {}
PS_SHAPES = {
    "N2C490H64W64K300_o7_s2": (2, 490, 64, 64, 300, 7, 7, 2),
    "N3C105H37W53K19_o7x5_adaptive": (3, 105, 37, 53, 19, 7, 5, -1),
}
# N, C, O, H, W, kernel, stride, padding, dilation, groups, offset groups, mask.
DEFORM_SHAPES = {
    "N8C256O256H64W64_k3_s1_p1_d1_g1_og1_mask": (
        8,
        256,
        256,
        64,
        64,
        3,
        1,
        1,
        1,
        1,
        1,
        True,
    ),
    "N3C12O10H17W23_k3_s2_p2_d2_g2_og3_nomask": (
        3,
        12,
        10,
        17,
        23,
        3,
        2,
        2,
        2,
        2,
        3,
        False,
    ),
}


@pytest.fixture
def vision() -> ModuleType:
    return pytest.importorskip("torchvision")


def _roi_inputs(
    shape: tuple[int, int, int, int, int, int, int, int], dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    n, c, h, w, k, _, _, _ = shape
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(n, c, h, w, dtype=dtype, generator=generator)
    starts = torch.rand(k, 2, generator=generator) * torch.tensor([w * 0.7, h * 0.7])
    sizes = (torch.rand(k, 2, generator=generator) * 0.2 + 0.05) * torch.tensor([w, h])
    batches = (torch.arange(k) % n).float().unsqueeze(1)
    return x, torch.cat((batches, starts, starts + sizes), dim=1).to(dtype)


@pytest.mark.parametrize("dtype_id", ["f32", "f16"])
@pytest.mark.parametrize("shape_id", ALIGN_SHAPES)
@pytest.mark.bench_op("torchvision::roi_align")
def test_roi_align(
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
    vision: ModuleType,
):
    shape = ALIGN_SHAPES[shape_id]
    _, c, _, _, k, ph, pw, sampling = shape
    x, rois = _roi_inputs(shape, DTYPES[dtype_id])
    x_ref, x_our = both(x, hw, mojo_device)
    r_ref, r_our = both(rois, hw, mojo_device)
    bench.run(
        lambda: vision.ops.roi_align(x_ref, r_ref, (ph, pw), 1.0, sampling, True),
        lambda: vision.ops.roi_align(x_our, r_our, (ph, pw), 1.0, sampling, True),
        flops=float(k * c * ph * pw * 32),
    )


@pytest.mark.parametrize("dtype_id", ["f32", "f16"])
@pytest.mark.parametrize("shape_id", ALIGN_SHAPES)
@pytest.mark.bench_op("torchvision::_roi_align_backward")
def test_roi_align_backward(
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
    vision: ModuleType,
):
    shape = ALIGN_SHAPES[shape_id]
    n, c, h, w, k, ph, pw, sampling = shape
    _, rois = _roi_inputs(shape, DTYPES[dtype_id])
    grad = torch.randn(
        k, c, ph, pw, dtype=DTYPES[dtype_id], generator=torch.Generator().manual_seed(1)
    )
    g_ref, g_our = both(grad, hw, mojo_device)
    r_ref, r_our = both(rois, hw, mojo_device)
    bench.run(
        lambda: torch.ops.torchvision._roi_align_backward(
            g_ref, r_ref, 1.0, ph, pw, n, c, h, w, sampling, True
        ),
        lambda: torch.ops.torchvision._roi_align_backward(
            g_our, r_our, 1.0, ph, pw, n, c, h, w, sampling, True
        ),
        flops=float(k * c * ph * pw * 32),
    )


@pytest.mark.parametrize("dtype_id", ["f32", "f16"])
@pytest.mark.parametrize("shape_id", POOL_SHAPES)
@pytest.mark.bench_op("torchvision::roi_pool")
def test_roi_pool(
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
    vision: ModuleType,
):
    shape = POOL_SHAPES[shape_id]
    _, c, _, _, k, ph, pw, _ = shape
    x, rois = _roi_inputs(shape, DTYPES[dtype_id])
    x_ref, x_our = both(x, hw, mojo_device)
    r_ref, r_our = both(rois, hw, mojo_device)
    bench.run(
        lambda: vision.ops.roi_pool(x_ref, r_ref, (ph, pw)),
        lambda: vision.ops.roi_pool(x_our, r_our, (ph, pw)),
        flops=float(k * c * ph * pw * 32),
    )


@pytest.mark.parametrize("dtype_id", ["f32", "f16"])
@pytest.mark.parametrize("shape_id", POOL_SHAPES)
@pytest.mark.bench_op("torchvision::_roi_pool_backward")
def test_roi_pool_backward(
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
    vision: ModuleType,
):
    shape = POOL_SHAPES[shape_id]
    n, c, h, w, k, ph, pw, _ = shape
    x, rois = _roi_inputs(shape, DTYPES[dtype_id])
    grad = torch.randn(
        k, c, ph, pw, dtype=DTYPES[dtype_id], generator=torch.Generator().manual_seed(1)
    )
    x_ref, x_our = both(x, hw, mojo_device)
    r_ref, r_our = both(rois, hw, mojo_device)
    g_ref, g_our = both(grad, hw, mojo_device)
    _, a_ref = torch.ops.torchvision.roi_pool(x_ref, r_ref, 1.0, ph, pw)
    _, a_our = torch.ops.torchvision.roi_pool(x_our, r_our, 1.0, ph, pw)
    bench.run(
        lambda: torch.ops.torchvision._roi_pool_backward(
            g_ref, r_ref, a_ref, 1.0, ph, pw, n, c, h, w
        ),
        lambda: torch.ops.torchvision._roi_pool_backward(
            g_our, r_our, a_our, 1.0, ph, pw, n, c, h, w
        ),
        flops=float(k * c * ph * pw * 4),
    )


@pytest.mark.parametrize("dtype_id", ["f32", "f16"])
@pytest.mark.parametrize("shape_id", NMS_SHAPES)
@pytest.mark.bench_op("torchvision::nms")
def test_nms(
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
    vision: ModuleType,
):
    k = NMS_SHAPES[shape_id]
    generator = torch.Generator().manual_seed(0)
    starts = torch.rand(k, 2, generator=generator) * 800
    sizes = torch.rand(k, 2, generator=generator) * 200 + 1
    boxes = torch.cat((starts, starts + sizes), dim=1).to(DTYPES[dtype_id])
    scores = torch.rand(k, generator=generator).to(DTYPES[dtype_id])
    b_ref, b_our = both(boxes, hw, mojo_device)
    s_ref, s_our = both(scores, hw, mojo_device)
    bench.run(
        lambda: vision.ops.nms(b_ref, s_ref, 0.5),
        lambda: vision.ops.nms(b_our, s_our, 0.5),
        flops=float(k * k * 12),
    )


def _ps_call(
    kind: str,
    backward: bool,
    x: torch.Tensor,
    rois: torch.Tensor,
    grad: torch.Tensor,
    shape: tuple[int, int, int, int, int, int, int, int],
) -> Callable[[], object]:
    n, c, h, w, _, ph, pw, sampling = shape
    args = (x, rois, 1.0, ph, pw)

    def forward() -> tuple[torch.Tensor, torch.Tensor]:
        if kind == "align":
            return torch.ops.torchvision.ps_roi_align(*args, sampling)
        return torch.ops.torchvision.ps_roi_pool(*args)

    if not backward:
        return forward
    _, mapping = forward()
    if kind == "align":
        return lambda: torch.ops.torchvision._ps_roi_align_backward(
            grad, rois, mapping, 1.0, ph, pw, sampling, n, c, h, w
        )
    return lambda: torch.ops.torchvision._ps_roi_pool_backward(
        grad, rois, mapping, 1.0, ph, pw, n, c, h, w
    )


def _bench_ps(
    kind: str,
    backward: bool,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    shape = PS_SHAPES[shape_id]
    _, c, _, _, k, ph, pw, _ = shape
    x, rois = _roi_inputs(shape, DTYPES[dtype_id])
    grad = torch.randn(
        k,
        c // (ph * pw),
        ph,
        pw,
        dtype=DTYPES[dtype_id],
        generator=torch.Generator().manual_seed(1),
    )
    x_ref, x_our = both(x, hw, mojo_device)
    r_ref, r_our = both(rois, hw, mojo_device)
    g_ref, g_our = both(grad, hw, mojo_device)
    ref = _ps_call(kind, backward, x_ref, r_ref, g_ref, shape)
    ours = _ps_call(kind, backward, x_our, r_our, g_our, shape)
    bench.run(ref, ours, flops=float(k * c * 32))


@pytest.mark.parametrize("dtype_id", ["f32", "f16"])
@pytest.mark.parametrize("shape_id", PS_SHAPES)
@pytest.mark.bench_op("torchvision::ps_roi_align")
def test_ps_roi_align(
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
    vision: ModuleType,
):
    _bench_ps("align", False, shape_id, dtype_id, bench, hw, mojo_device)


@pytest.mark.parametrize("dtype_id", ["f32", "f16"])
@pytest.mark.parametrize("shape_id", PS_SHAPES)
@pytest.mark.bench_op("torchvision::_ps_roi_align_backward")
def test_ps_roi_align_backward(
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
    vision: ModuleType,
):
    _bench_ps("align", True, shape_id, dtype_id, bench, hw, mojo_device)


@pytest.mark.parametrize("dtype_id", ["f32", "f16"])
@pytest.mark.parametrize("shape_id", PS_SHAPES)
@pytest.mark.bench_op("torchvision::ps_roi_pool")
def test_ps_roi_pool(
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
    vision: ModuleType,
):
    _bench_ps("pool", False, shape_id, dtype_id, bench, hw, mojo_device)


@pytest.mark.parametrize("dtype_id", ["f32", "f16"])
@pytest.mark.parametrize("shape_id", PS_SHAPES)
@pytest.mark.bench_op("torchvision::_ps_roi_pool_backward")
def test_ps_roi_pool_backward(
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
    vision: ModuleType,
):
    _bench_ps("pool", True, shape_id, dtype_id, bench, hw, mojo_device)


def _deform_inputs(
    shape: tuple[int, int, int, int, int, int, int, int, int, int, int, bool],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    n, c, o, h, w, k, stride, pad, dilation, groups, offset_groups, use_mask = shape
    oh = (h + 2 * pad - dilation * (k - 1) - 1) // stride + 1
    ow = (w + 2 * pad - dilation * (k - 1) - 1) // stride + 1
    generator = torch.Generator().manual_seed(312)
    x = torch.randn(n, c, h, w, generator=generator).to(dtype)
    weight = (torch.randn(o, c // groups, k, k, generator=generator) * 0.05).to(dtype)
    offset = (
        torch.rand(n, 2 * offset_groups * k * k, oh, ow, generator=generator) - 0.5
    ).to(dtype)
    mask = (
        torch.rand(n, offset_groups * k * k, oh, ow, generator=generator).to(dtype)
        if use_mask
        else torch.zeros(n, 1, dtype=dtype)
    )
    bias = torch.randn(o, generator=generator).to(dtype)
    grad = torch.randn(n, o, oh, ow, generator=generator).to(dtype)
    return x, weight, offset, mask, bias, grad


def _deform_call(
    tensors: tuple[torch.Tensor, ...],
    shape: tuple[int, int, int, int, int, int, int, int, int, int, int, bool],
    backward: bool,
) -> Callable[[], object]:
    _, _, _, _, _, _, stride, pad, dilation, groups, offset_groups, use_mask = shape
    x, weight, offset, mask, bias, grad = tensors
    args = (
        x,
        weight,
        offset,
        mask,
        bias,
        stride,
        stride,
        pad,
        pad,
        dilation,
        dilation,
        groups,
        offset_groups,
        use_mask,
    )
    if backward:
        return lambda: torch.ops.torchvision._deform_conv2d_backward(grad, *args)
    return lambda: torch.ops.torchvision.deform_conv2d(*args)


def _bench_deform(
    backward: bool,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    shape = DEFORM_SHAPES[shape_id]
    tensors = _deform_inputs(shape, DTYPES[dtype_id])
    pairs = [both(t, hw, mojo_device) for t in tensors]
    ref = _deform_call(tuple(p[0] for p in pairs), shape, backward)
    ours = _deform_call(tuple(p[1] for p in pairs), shape, backward)
    n, c, o, _, _, k, _, _, _, groups, _, _ = shape
    oh, ow = tensors[-1].shape[-2:]
    flops = 2 * n * o * oh * ow * (c // groups) * k * k
    bench.run(ref, ours, flops=float(flops * (3 if backward else 1)))


@pytest.mark.parametrize("dtype_id", ["f32", "f16"])
@pytest.mark.parametrize("shape_id", DEFORM_SHAPES)
@pytest.mark.bench_op("torchvision::deform_conv2d")
def test_deform_conv2d(
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
    vision: ModuleType,
):
    _bench_deform(False, shape_id, dtype_id, bench, hw, mojo_device)


@pytest.mark.parametrize("dtype_id", ["f32", "f16"])
@pytest.mark.parametrize("shape_id", DEFORM_SHAPES)
@pytest.mark.bench_op("torchvision::_deform_conv2d_backward")
def test_deform_conv2d_backward(
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
    vision: ModuleType,
):
    _bench_deform(True, shape_id, dtype_id, bench, hw, mojo_device)
