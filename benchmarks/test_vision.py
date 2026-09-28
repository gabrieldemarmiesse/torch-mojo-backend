"""Convolution / pooling / resize benchmarks.

Driven through the public functional entry points (F.conv2d,
F.max_pool2d, F.interpolate, ...), which reach the registered aten ops:
convolution, convolution_backward (called directly), max_pool2d_with_indices (max_pool2d is composite over it),
_adaptive_avg_pool2d, avg_pool2d, upsample_bilinear2d, upsample_nearest2d.
The other upsampling ops (1-d, 3-d, nearest-exact, bicubic, antialiased
bilinear) and every upsampling backward are called directly as aten ops,
one test per spatial rank with the op as a parametrize axis.
Shape tokens fold the kernel/stride/output configuration in.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from bench_lib.cases import DTYPES, both, op_params
from bench_lib.check import Bench
from bench_lib.hw import Hardware

# (N, C_in, H, W, C_out, kernel, stride, padding)
CONV_SHAPES: dict[str, tuple[int, int, int, int, int, int, int, int]] = {
    "N32xC64x56x56_K64k3s1": (32, 64, 56, 56, 64, 3, 1, 1),
    "N8xC3x224x224_K64k7s2": (8, 3, 224, 224, 64, 7, 2, 3),
}
# (N, C_in, L, C_out, kernel, stride, padding). conv1d is the rank-3
# aten::convolution; one awkward length on purpose.
CONV1D_SHAPES: dict[str, tuple[int, int, int, int, int, int, int]] = {
    "N4xC80xL3000_K384k3s1": (4, 80, 3000, 384, 3, 1, 1),
    "N8xC256xL357_K512k5s2": (8, 256, 357, 512, 5, 2, 2),
}
# conv2d backward: the two forward geometries plus an awkward mid-network
# shape and the deepest ResNet stage, where K = C*KH*KW outgrows N*OH*OW.
CONV_BACKWARD_SHAPES: dict[str, tuple[int, int, int, int, int, int, int, int]] = {
    **CONV_SHAPES,
    "N7xC48x39x53_K80k3s1": (7, 48, 39, 53, 80, 3, 1, 1),
    "N32xC512x7x7_K512k3s1": (32, 512, 7, 7, 512, 3, 1, 1),
}
# One node per gradient (output_mask with one slot set): the three are
# different kernels whose ratios move independently. It rides the `layout`
# axis so the three land as three leaves of one shape.
CONV_BACKWARD_GRADS: dict[str, list[bool]] = {
    "dgrad": [True, False, False],
    "wgrad": [False, True, False],
    "bgrad": [False, False, True],
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

# (N, C, L) / (N, C, D, H, W); every case doubles each spatial extent.
UPSAMPLE1D_SHAPES: dict[str, tuple[int, ...]] = {
    "N8xC64xL4096_x2": (8, 64, 4096),
    "N4xC3xL35789_x2": (4, 3, 35789),
}
UPSAMPLE3D_SHAPES: dict[str, tuple[int, ...]] = {
    "N2xC32x16x32x32_x2": (2, 32, 16, 32, 32),
    "N1xC3x13x27x35_x2": (1, 3, 13, 27, 35),
}
UPSAMPLE_BY_RANK = {1: UPSAMPLE1D_SHAPES, 2: UPSAMPLE_SHAPES, 3: UPSAMPLE3D_SHAPES}
UPSAMPLE1D_OPS = {
    "upsample_nearest1d": 1,
    "_upsample_nearest_exact1d": 1,
    "upsample_linear1d": 1,
}
UPSAMPLE2D_OPS = {
    "_upsample_nearest_exact2d": 2,
    "upsample_bicubic2d": 2,
    "_upsample_bilinear2d_aa": 2,
}
UPSAMPLE3D_OPS = {
    "upsample_nearest3d": 3,
    "_upsample_nearest_exact3d": 3,
    "upsample_trilinear3d": 3,
}
UPSAMPLE1D_BACKWARD_OPS = {f"{name}_backward": 1 for name in UPSAMPLE1D_OPS}
UPSAMPLE2D_BACKWARD_OPS = {
    f"{name}_backward": 2
    for name in ("upsample_nearest2d", "upsample_bilinear2d", *UPSAMPLE2D_OPS)
}
UPSAMPLE3D_BACKWARD_OPS = {f"{name}_backward": 3 for name in UPSAMPLE3D_OPS}
_UPSAMPLE_TESTS = {
    "test_upsample1d": UPSAMPLE1D_OPS,
    "test_upsample2d": UPSAMPLE2D_OPS,
    "test_upsample3d": UPSAMPLE3D_OPS,
    "test_upsample1d_backward": UPSAMPLE1D_BACKWARD_OPS,
    "test_upsample2d_backward": UPSAMPLE2D_BACKWARD_OPS,
    "test_upsample3d_backward": UPSAMPLE3D_BACKWARD_OPS,
}

COVERS: dict[str, str] = {
    "aten::convolution": "test_conv2d, test_conv1d (rank 3)",
    "aten::convolution_backward": "test_conv2d_backward",
    "aten::_adaptive_avg_pool2d": "test_adaptive_avg_pool2d",
    "aten::avg_pool2d": "test_avg_pool2d",
    "aten::max_pool2d_with_indices": (
        "test_max_pool2d (F.max_pool2d is composite over it)"
    ),
    "aten::upsample_bilinear2d": "test_upsample_bilinear2d",
    "aten::upsample_nearest2d": "test_upsample_nearest2d",
    **{f"aten::{name}": test for test, ops in _UPSAMPLE_TESTS.items() for name in ops},
}

_UPSAMPLE_OUT = (
    "out-variant plumbing over an already-benchmarked functional impl: the "
    "same resample kernel, written into the caller's tensor"
)
SKIPPED: dict[str, str] = {
    f"aten::{name}{suffix}": _UPSAMPLE_OUT
    for name in (
        "upsample_nearest2d",
        "upsample_bilinear2d",
        *UPSAMPLE1D_OPS,
        *UPSAMPLE2D_OPS,
        *UPSAMPLE3D_OPS,
    )
    for suffix in (".out", "_backward.grad_input")
}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", CONV_SHAPES)
@pytest.mark.bench_op("convolution")
def test_conv2d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
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
@pytest.mark.parametrize("layout", CONV_BACKWARD_GRADS)
@pytest.mark.parametrize("shape_id", CONV_BACKWARD_SHAPES)
@pytest.mark.bench_op("convolution_backward")
def test_conv2d_backward(
    shape_id: str,
    layout: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    n, c_in, h, w, c_out, k, stride, pad = CONV_BACKWARD_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    h_out = (h + 2 * pad - k) // stride + 1
    w_out = (w + 2 * pad - k) // stride + 1
    g_ref, g_our = both(
        torch.randn(n, c_out, h_out, w_out, dtype=dtype), hw, mojo_device
    )
    x_ref, x_our = both(torch.randn(n, c_in, h, w, dtype=dtype), hw, mojo_device)
    w_ref, w_our = both(
        torch.randn(c_out, c_in, k, k, dtype=dtype) * 0.1, hw, mojo_device
    )
    tail = (
        [c_out],
        [stride, stride],
        [pad, pad],
        [1, 1],
        False,
        [0, 0],
        1,
        CONV_BACKWARD_GRADS[layout],
    )
    # The bias gradient is a reduction: its "flops" is its element count.
    flops = (
        float(n * c_out * h_out * w_out)
        if layout == "bgrad"
        else 2.0 * n * c_out * h_out * w_out * c_in * k * k
    )
    bench.run(
        lambda: torch.ops.aten.convolution_backward(g_ref, x_ref, w_ref, *tail),
        lambda: torch.ops.aten.convolution_backward(g_our, x_our, w_our, *tail),
        flops=flops,
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", CONV1D_SHAPES)
@pytest.mark.bench_op("convolution")
def test_conv1d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c_in, length, c_out, k, stride, pad = CONV1D_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(n, c_in, length, dtype=dtype), hw, mojo_device)
    w_ref, w_our = both(torch.randn(c_out, c_in, k, dtype=dtype) * 0.1, hw, mojo_device)
    b_ref, b_our = both(torch.randn(c_out, dtype=dtype), hw, mojo_device)
    l_out = (length + 2 * pad - k) // stride + 1
    flops = 2.0 * n * c_out * l_out * c_in * k
    bench.run(
        lambda: F.conv1d(x_ref, w_ref, b_ref, stride, pad),
        lambda: F.conv1d(x_our, w_our, b_our, stride, pad),
        flops=flops,
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", ADAPTIVE_SHAPES)
@pytest.mark.bench_op("_adaptive_avg_pool2d")
def test_adaptive_avg_pool2d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
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
):
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
):
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
):
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


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", UPSAMPLE_SHAPES)
@pytest.mark.bench_op("upsample_nearest2d")
def test_upsample_nearest2d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c, h, w = UPSAMPLE_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(n, c, h, w, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: F.interpolate(x_ref, scale_factor=2, mode="nearest"),
        lambda: F.interpolate(x_our, scale_factor=2, mode="nearest"),
        # No interpolation weights (unlike bilinear): a pure index gather, so
        # there is no meaningful FLOP count -- element throughput instead.
        flops=float(x_ref.numel()),
    )


def _interpolates(name: str) -> bool:
    """Whether the op takes `align_corners` (every mode but nearest)."""
    return "linear" in name or "bicubic" in name


def _bench_upsample(
    name: str,
    rank: int,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    """One upsampling op (or, for a `_backward` name, its backward) called
    directly at twice the input's spatial size."""
    shape = UPSAMPLE_BY_RANK[rank][shape_id]
    osize = [2 * d for d in shape[2:]]
    out_shape = [*shape[:2], *osize]
    dtype = DTYPES[dtype_id]
    op = getattr(torch.ops.aten, name)
    align = [False] if _interpolates(name) else []
    if name.endswith("_backward"):
        g_ref, g_our = both(torch.randn(out_shape, dtype=dtype), hw, mojo_device)
        bench.run(
            lambda: op(g_ref, osize, list(shape), *align),
            lambda: op(g_our, osize, list(shape), *align),
            flops=float(math.prod(out_shape)),
        )
    else:
        x_ref, x_our = both(torch.randn(shape, dtype=dtype), hw, mojo_device)
        bench.run(
            lambda: op(x_ref, osize, *align),
            lambda: op(x_our, osize, *align),
            flops=float(math.prod(out_shape)),
        )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", UPSAMPLE1D_SHAPES)
@pytest.mark.parametrize("op_name", op_params(UPSAMPLE1D_OPS))
def test_upsample1d(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    _bench_upsample(op_name, 1, shape_id, dtype_id, bench, hw, mojo_device)


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", UPSAMPLE_SHAPES)
@pytest.mark.parametrize("op_name", op_params(UPSAMPLE2D_OPS))
def test_upsample2d(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    _bench_upsample(op_name, 2, shape_id, dtype_id, bench, hw, mojo_device)


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", UPSAMPLE3D_SHAPES)
@pytest.mark.parametrize("op_name", op_params(UPSAMPLE3D_OPS))
def test_upsample3d(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    _bench_upsample(op_name, 3, shape_id, dtype_id, bench, hw, mojo_device)


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", UPSAMPLE1D_SHAPES)
@pytest.mark.parametrize("op_name", op_params(UPSAMPLE1D_BACKWARD_OPS))
def test_upsample1d_backward(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    _bench_upsample(op_name, 1, shape_id, dtype_id, bench, hw, mojo_device)


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", UPSAMPLE_SHAPES)
@pytest.mark.parametrize("op_name", op_params(UPSAMPLE2D_BACKWARD_OPS))
def test_upsample2d_backward(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    _bench_upsample(op_name, 2, shape_id, dtype_id, bench, hw, mojo_device)


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", UPSAMPLE3D_SHAPES)
@pytest.mark.parametrize("op_name", op_params(UPSAMPLE3D_BACKWARD_OPS))
def test_upsample3d_backward(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    _bench_upsample(op_name, 3, shape_id, dtype_id, bench, hw, mojo_device)
