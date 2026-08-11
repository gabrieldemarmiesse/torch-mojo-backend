"""Convolution / pooling / resize benchmarks.

Driven through the public functional entry points (F.conv2d,
F.max_pool2d, F.interpolate, ...), which reach the registered aten ops:
convolution, max_pool2d_with_indices (max_pool2d is composite over it),
_adaptive_avg_pool2d, avg_pool2d, upsample_bilinear2d.  Shape tokens
fold the kernel/stride/output configuration in.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from bench_lib.cases import DTYPES, both
from bench_lib.check import Bench
from bench_lib.hw import Hardware

# (N, C_in, H, W, C_out, kernel, stride, padding)
CONV_SHAPES: dict[str, tuple[int, int, int, int, int, int, int, int]] = {
    "N32xC64x56x56_K64k3s1": (32, 64, 56, 56, 64, 3, 1, 1),
    "N8xC3x224x224_K64k7s2": (8, 3, 224, 224, 64, 7, 2, 3),
}
# (N, C, H, W, output)
ADAPTIVE_SHAPES: dict[str, tuple[int, int, int, int, int]] = {
    "N32xC512x28x28_o7": (32, 512, 28, 28, 7),
    "N8xC64x112x112_o1": (8, 64, 112, 112, 1),
}
# (N, C, H, W, kernel, stride, padding)
POOL_SHAPES: dict[str, tuple[int, int, int, int, int, int, int]] = {
    "N32xC64x112x112_k2s2": (32, 64, 112, 112, 2, 2, 0),
    "N8xC256x28x28_k3s2": (8, 256, 28, 28, 3, 2, 1),
}
# (N, C, H, W)
UPSAMPLE_SHAPES: dict[str, tuple[int, int, int, int]] = {
    "N8xC64x56x56_x2": (8, 64, 56, 56),
    "N2xC3x256x256_x2": (2, 3, 256, 256),
}

COVERS: dict[str, str] = {
    "aten::convolution": "test_conv2d",
    "aten::_adaptive_avg_pool2d": "test_adaptive_avg_pool2d",
    "aten::avg_pool2d": "test_avg_pool2d",
    "aten::max_pool2d_with_indices": (
        "test_max_pool2d (F.max_pool2d is composite over it)"
    ),
    "aten::upsample_bilinear2d": "test_upsample_bilinear2d",
}

SKIPPED: dict[str, str] = {}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", CONV_SHAPES)
@pytest.mark.bench_op("convolution")
def test_conv2d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    n, c_in, h, w, c_out, k, stride, pad = CONV_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(n, c_in, h, w, dtype=dtype), hw, mojo_device)
    w_ref, w_our = both(
        torch.randn(c_out, c_in, k, k, dtype=dtype) * 0.1, hw, mojo_device
    )
    b_ref, b_our = both(torch.randn(c_out, dtype=dtype), hw, mojo_device)
    h_out = (h + 2 * pad - k) // stride + 1
    w_out = (w + 2 * pad - k) // stride + 1
    flops = 2.0 * n * c_out * h_out * w_out * c_in * k * k
    bench.run(
        lambda: F.conv2d(x_ref, w_ref, b_ref, stride, pad),
        lambda: F.conv2d(x_our, w_our, b_our, stride, pad),
        flops=flops,
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", ADAPTIVE_SHAPES)
@pytest.mark.bench_op("_adaptive_avg_pool2d")
def test_adaptive_avg_pool2d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    n, c, h, w, out = ADAPTIVE_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(n, c, h, w, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: F.adaptive_avg_pool2d(x_ref, (out, out)),
        lambda: F.adaptive_avg_pool2d(x_our, (out, out)),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", POOL_SHAPES)
@pytest.mark.bench_op("avg_pool2d")
def test_avg_pool2d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    n, c, h, w, k, stride, pad = POOL_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(n, c, h, w, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: F.avg_pool2d(x_ref, k, stride, pad),
        lambda: F.avg_pool2d(x_our, k, stride, pad),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", POOL_SHAPES)
@pytest.mark.bench_op("max_pool2d_with_indices")
def test_max_pool2d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    n, c, h, w, k, stride, pad = POOL_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(n, c, h, w, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: F.max_pool2d(x_ref, k, stride, pad),
        lambda: F.max_pool2d(x_our, k, stride, pad),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", UPSAMPLE_SHAPES)
@pytest.mark.bench_op("upsample_bilinear2d")
def test_upsample_bilinear2d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    n, c, h, w = UPSAMPLE_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(n, c, h, w, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: F.interpolate(
            x_ref, scale_factor=2, mode="bilinear", align_corners=False
        ),
        lambda: F.interpolate(
            x_our, scale_factor=2, mode="bilinear", align_corners=False
        ),
        flops=float(x_ref.numel()) * 4.0,
    )
