"""Convolution / pooling / resize benchmarks.

Driven through the public functional entry points (F.conv2d,
F.max_pool2d, F.interpolate, ...), which reach the registered aten ops:
convolution, convolution_backward (called directly), max_pool2d_with_indices (max_pool2d is composite over it),
_adaptive_avg_pool2d, avg_pool2d, upsample_bilinear2d, upsample_nearest2d,
the 3-D and adaptive-max pools, max_unpool and im2col / col2im
(F.unfold / F.fold). The pooling backwards are called as aten ops.
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
# (N, C_in, H, W, C_out, kernel, stride, padding, output_padding): a
# transposed convolution is col2im(weight^T @ x), the conv data gradient.
CONV_T_SHAPES: dict[str, tuple[int, int, int, int, int, int, int, int, int]] = {
    "N16xC128x28x28_K64k4s2": (16, 128, 28, 28, 64, 4, 2, 1, 0),
    "N4xC64x37x29_K32k3s2op1": (4, 64, 37, 29, 32, 3, 2, 1, 1),
}
# (N, C_in, D, H, W, C_out, kernel, stride, padding): the volumetric im2col.
CONV3D_SHAPES: dict[str, tuple[int, int, int, int, int, int, int, int, int]] = {
    "N4xC32x16x28x28_K64k3s1": (4, 32, 16, 28, 28, 64, 3, 1, 1),
    "N2xC16x9x21x33_K24k3s2": (2, 16, 9, 21, 33, 24, 3, 2, 1),
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
# (N, C, D, H, W, kernel, stride, padding); one awkward volume on purpose.
POOL3D_SHAPES: dict[str, tuple[int, int, int, int, int, int, int, int]] = {
    "N4xC32x16x28x28_k2s2": (4, 32, 16, 28, 28, 2, 2, 0),
    "N2xC16x9x21x33_k3s2": (2, 16, 9, 21, 33, 3, 2, 1),
}
# (N, C, D, H, W, output)
ADAPTIVE3D_SHAPES: dict[str, tuple[int, int, int, int, int, int]] = {
    "N4xC64x8x28x28_o4": (4, 64, 8, 28, 28, 4),
    "N2xC32x9x21x33_o5": (2, 32, 9, 21, 33, 5),
}
# (N, C, H, W, kernel, stride, padding) for F.unfold / F.fold.
FOLD_SHAPES: dict[str, tuple[int, int, int, int, int, int, int]] = {
    "N8xC64x56x56_k3s1": (8, 64, 56, 56, 3, 1, 1),
    "N4xC32x37x53_k4s2": (4, 32, 37, 53, 4, 2, 1),
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
    "aten::convolution": (
        "test_conv2d, test_conv1d (rank 3), test_conv_transpose2d, test_conv3d"
    ),
    "aten::convolution_backward": "test_conv2d_backward",
    "aten::_adaptive_avg_pool2d": "test_adaptive_avg_pool2d",
    "aten::avg_pool2d": "test_avg_pool2d",
    "aten::max_pool2d_with_indices": (
        "test_max_pool2d (F.max_pool2d is composite over it)"
    ),
    "aten::upsample_bilinear2d": "test_upsample_bilinear2d",
    "aten::upsample_nearest2d": "test_upsample_nearest2d",
    **{f"aten::{name}": test for test, ops in _UPSAMPLE_TESTS.items() for name in ops},
    "aten::max_pool2d_with_indices_backward": "test_max_pool2d_backward",
    "aten::avg_pool2d_backward": "test_avg_pool2d_backward",
    "aten::_adaptive_avg_pool2d_backward": "test_adaptive_avg_pool2d_backward",
    "aten::adaptive_max_pool2d": "test_adaptive_max_pool2d",
    "aten::adaptive_max_pool2d_backward": "test_adaptive_max_pool2d_backward",
    "aten::max_pool3d_with_indices": "test_max_pool3d",
    "aten::max_pool3d_with_indices_backward": "test_max_pool3d_backward",
    "aten::avg_pool3d": "test_avg_pool3d",
    "aten::avg_pool3d_backward": "test_avg_pool3d_backward",
    "aten::_adaptive_avg_pool3d": "test_adaptive_avg_pool3d",
    "aten::_adaptive_avg_pool3d_backward": "test_adaptive_avg_pool3d_backward",
    "aten::adaptive_max_pool3d": "test_adaptive_max_pool3d",
    "aten::adaptive_max_pool3d_backward": "test_adaptive_max_pool3d_backward",
    "aten::max_unpool2d": "test_max_unpool2d",
    "aten::max_unpool3d": "test_max_unpool3d",
    "aten::im2col": "test_im2col (F.unfold)",
    "aten::col2im": "test_col2im (F.fold)",
}

_UPSAMPLE_OUT = (
    "out-variant plumbing over an already-benchmarked functional impl: the "
    "same resample kernel, written into the caller's tensor"
)

_OUT = (
    "out-variant plumbing over the benchmarked functional op (compute, then "
    "copy into the caller's tensor)"
)
_CONV_ENTRY = (
    "a backend-specific convolution entry point (fixed groups / dilation / "
    "transposition): the same im2col + GEMM route as aten::convolution, which "
    "test_conv2d, test_conv3d and test_conv_transpose2d measure"
)
SKIPPED: dict[str, str] = {
    **{
        f"aten::{name}": _CONV_ENTRY
        for name in (
            "_conv_depthwise2d",
            "_conv_depthwise2d.out",
            "conv_depthwise3d",
            "conv_depthwise3d.out",
            "_slow_conv2d_forward",
            "_slow_conv2d_forward.output",
            "_slow_conv2d_backward.output_mask",
            "_slow_conv2d_backward.grad_input",
            "slow_conv3d_forward",
            "slow_conv3d_forward.output",
            "slow_conv_dilated2d",
            "slow_conv_dilated2d.out",
            "slow_conv_dilated3d",
            "slow_conv_dilated3d.out",
            "slow_conv_transpose2d",
            "slow_conv_transpose2d.out",
            "slow_conv_transpose3d",
            "slow_conv_transpose3d.out",
        )
    },
    **{
        f"aten::{name}{suffix}": _UPSAMPLE_OUT
        for name in (
            "upsample_nearest2d",
            "upsample_bilinear2d",
            *UPSAMPLE1D_OPS,
            *UPSAMPLE2D_OPS,
            *UPSAMPLE3D_OPS,
        )
        for suffix in (".out", "_backward.grad_input")
    },
    **{
        name: _OUT
        for name in (
            "aten::adaptive_avg_pool2d.out",
            "aten::adaptive_avg_pool3d.out",
            "aten::adaptive_avg_pool3d_backward.grad_input",
            "aten::adaptive_max_pool2d.out",
            "aten::adaptive_max_pool2d_backward.grad_input",
            "aten::adaptive_max_pool3d.out",
            "aten::adaptive_max_pool3d_backward.grad_input",
            "aten::avg_pool2d.out",
            "aten::avg_pool2d_backward.grad_input",
            "aten::avg_pool3d.out",
            "aten::avg_pool3d_backward.grad_input",
            "aten::col2im.out",
            "aten::im2col.out",
            "aten::max_pool2d_with_indices.out",
            "aten::max_pool2d_with_indices_backward.grad_input",
            "aten::max_pool3d_with_indices.out",
            "aten::max_pool3d_with_indices_backward.grad_input",
            "aten::max_unpool2d.out",
            "aten::max_unpool3d.out",
        )
    },
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
@pytest.mark.parametrize("shape_id", CONV_T_SHAPES)
@pytest.mark.bench_op("convolution")
def test_conv_transpose2d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c_in, h, w, c_out, k, stride, pad, opad = CONV_T_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(n, c_in, h, w, dtype=dtype), hw, mojo_device)
    w_ref, w_our = both(
        torch.randn(c_in, c_out, k, k, dtype=dtype) * 0.1, hw, mojo_device
    )
    b_ref, b_our = both(torch.randn(c_out, dtype=dtype), hw, mojo_device)
    flops = 2.0 * n * c_in * h * w * c_out * k * k
    bench.run(
        lambda: F.conv_transpose2d(x_ref, w_ref, b_ref, stride, pad, opad),
        lambda: F.conv_transpose2d(x_our, w_our, b_our, stride, pad, opad),
        flops=flops,
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", CONV3D_SHAPES)
@pytest.mark.bench_op("convolution")
def test_conv3d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c_in, d, h, w, c_out, k, stride, pad = CONV3D_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(n, c_in, d, h, w, dtype=dtype), hw, mojo_device)
    w_ref, w_our = both(
        torch.randn(c_out, c_in, k, k, k, dtype=dtype) * 0.1, hw, mojo_device
    )
    b_ref, b_our = both(torch.randn(c_out, dtype=dtype), hw, mojo_device)
    outs = [(e + 2 * pad - k) // stride + 1 for e in (d, h, w)]
    flops = 2.0 * n * c_out * math.prod(outs) * c_in * k**3
    bench.run(
        lambda: F.conv3d(x_ref, w_ref, b_ref, stride, pad),
        lambda: F.conv3d(x_our, w_our, b_our, stride, pad),
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


# ---------------------------------------------------------------------------
# Pooling backwards, 3-D and adaptive-max pools, unpooling, unfold / fold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", POOL_SHAPES)
@pytest.mark.bench_op("max_pool2d_with_indices_backward")
def test_max_pool2d_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c, h, w, k, stride, pad = POOL_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(n, c, h, w, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    out_ref, idx_ref = F.max_pool2d(x_ref, k, stride, pad, return_indices=True)
    out_our, idx_our = F.max_pool2d(x_our, k, stride, pad, return_indices=True)
    g_ref, g_our = both(
        torch.randn(out_ref.shape, dtype=out_ref.dtype), hw, mojo_device
    )
    tail = ([k, k], [stride, stride], [pad, pad], [1, 1], False)
    bench.run(
        lambda: torch.ops.aten.max_pool2d_with_indices_backward(
            g_ref, x_ref, *tail, idx_ref
        ),
        lambda: torch.ops.aten.max_pool2d_with_indices_backward(
            g_our, x_our, *tail, idx_our
        ),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", POOL_SHAPES)
@pytest.mark.bench_op("avg_pool2d_backward")
def test_avg_pool2d_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c, h, w, k, stride, pad = POOL_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(n, c, h, w, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    out_shape = F.avg_pool2d(x_ref, k, stride, pad).shape
    g_ref, g_our = both(torch.randn(out_shape, dtype=x_ref.dtype), hw, mojo_device)
    tail = ([k, k], [stride, stride], [pad, pad], False, True, None)
    bench.run(
        lambda: torch.ops.aten.avg_pool2d_backward(g_ref, x_ref, *tail),
        lambda: torch.ops.aten.avg_pool2d_backward(g_our, x_our, *tail),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", ADAPTIVE_SHAPES)
@pytest.mark.bench_op("_adaptive_avg_pool2d_backward")
def test_adaptive_avg_pool2d_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c, h, w, out = ADAPTIVE_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(n, c, h, w, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    g_ref, g_our = both(
        torch.randn(n, c, out, out, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: torch.ops.aten._adaptive_avg_pool2d_backward(g_ref, x_ref),
        lambda: torch.ops.aten._adaptive_avg_pool2d_backward(g_our, x_our),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", ADAPTIVE_SHAPES)
@pytest.mark.bench_op("adaptive_max_pool2d")
def test_adaptive_max_pool2d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c, h, w, out = ADAPTIVE_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(n, c, h, w, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: F.adaptive_max_pool2d(x_ref, (out, out)),
        lambda: F.adaptive_max_pool2d(x_our, (out, out)),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", ADAPTIVE_SHAPES)
@pytest.mark.bench_op("adaptive_max_pool2d_backward")
def test_adaptive_max_pool2d_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c, h, w, out = ADAPTIVE_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(n, c, h, w, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    _, idx_ref = F.adaptive_max_pool2d(x_ref, (out, out), return_indices=True)
    _, idx_our = F.adaptive_max_pool2d(x_our, (out, out), return_indices=True)
    g_ref, g_our = both(
        torch.randn(n, c, out, out, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: torch.ops.aten.adaptive_max_pool2d_backward(g_ref, x_ref, idx_ref),
        lambda: torch.ops.aten.adaptive_max_pool2d_backward(g_our, x_our, idx_our),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", POOL3D_SHAPES)
@pytest.mark.bench_op("max_pool3d_with_indices")
def test_max_pool3d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c, d, h, w, k, stride, pad = POOL3D_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(n, c, d, h, w, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: F.max_pool3d(x_ref, k, stride, pad),
        lambda: F.max_pool3d(x_our, k, stride, pad),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", POOL3D_SHAPES)
@pytest.mark.bench_op("max_pool3d_with_indices_backward")
def test_max_pool3d_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c, d, h, w, k, stride, pad = POOL3D_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(n, c, d, h, w, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    out_ref, idx_ref = F.max_pool3d(x_ref, k, stride, pad, return_indices=True)
    _, idx_our = F.max_pool3d(x_our, k, stride, pad, return_indices=True)
    g_ref, g_our = both(
        torch.randn(out_ref.shape, dtype=out_ref.dtype), hw, mojo_device
    )
    tail = ([k] * 3, [stride] * 3, [pad] * 3, [1, 1, 1], False)
    bench.run(
        lambda: torch.ops.aten.max_pool3d_with_indices_backward(
            g_ref, x_ref, *tail, idx_ref
        ),
        lambda: torch.ops.aten.max_pool3d_with_indices_backward(
            g_our, x_our, *tail, idx_our
        ),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", POOL3D_SHAPES)
@pytest.mark.bench_op("avg_pool3d")
def test_avg_pool3d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c, d, h, w, k, stride, pad = POOL3D_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(n, c, d, h, w, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: F.avg_pool3d(x_ref, k, stride, pad),
        lambda: F.avg_pool3d(x_our, k, stride, pad),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", POOL3D_SHAPES)
@pytest.mark.bench_op("avg_pool3d_backward")
def test_avg_pool3d_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c, d, h, w, k, stride, pad = POOL3D_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(n, c, d, h, w, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    out_shape = F.avg_pool3d(x_ref, k, stride, pad).shape
    g_ref, g_our = both(torch.randn(out_shape, dtype=x_ref.dtype), hw, mojo_device)
    tail = ([k] * 3, [stride] * 3, [pad] * 3, False, True, None)
    bench.run(
        lambda: torch.ops.aten.avg_pool3d_backward(g_ref, x_ref, *tail),
        lambda: torch.ops.aten.avg_pool3d_backward(g_our, x_our, *tail),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", ADAPTIVE3D_SHAPES)
@pytest.mark.bench_op("_adaptive_avg_pool3d")
def test_adaptive_avg_pool3d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c, d, h, w, out = ADAPTIVE3D_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(n, c, d, h, w, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: F.adaptive_avg_pool3d(x_ref, out),
        lambda: F.adaptive_avg_pool3d(x_our, out),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", ADAPTIVE3D_SHAPES)
@pytest.mark.bench_op("_adaptive_avg_pool3d_backward")
def test_adaptive_avg_pool3d_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c, d, h, w, out = ADAPTIVE3D_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(n, c, d, h, w, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    g_ref, g_our = both(
        torch.randn(n, c, out, out, out, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: torch.ops.aten._adaptive_avg_pool3d_backward(g_ref, x_ref),
        lambda: torch.ops.aten._adaptive_avg_pool3d_backward(g_our, x_our),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", ADAPTIVE3D_SHAPES)
@pytest.mark.bench_op("adaptive_max_pool3d")
def test_adaptive_max_pool3d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c, d, h, w, out = ADAPTIVE3D_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(n, c, d, h, w, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: F.adaptive_max_pool3d(x_ref, out),
        lambda: F.adaptive_max_pool3d(x_our, out),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", ADAPTIVE3D_SHAPES)
@pytest.mark.bench_op("adaptive_max_pool3d_backward")
def test_adaptive_max_pool3d_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c, d, h, w, out = ADAPTIVE3D_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(n, c, d, h, w, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    _, idx_ref = F.adaptive_max_pool3d(x_ref, out, return_indices=True)
    _, idx_our = F.adaptive_max_pool3d(x_our, out, return_indices=True)
    g_ref, g_our = both(
        torch.randn(n, c, out, out, out, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: torch.ops.aten.adaptive_max_pool3d_backward(g_ref, x_ref, idx_ref),
        lambda: torch.ops.aten.adaptive_max_pool3d_backward(g_our, x_our, idx_our),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", POOL_SHAPES)
@pytest.mark.bench_op("max_unpool2d")
def test_max_unpool2d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c, h, w, k, stride, pad = POOL_SHAPES[shape_id]
    x = torch.randn(n, c, h, w, dtype=DTYPES[dtype_id])
    pooled, idx = F.max_pool2d(x, k, stride, pad, return_indices=True)
    p_ref, p_our = both(pooled, hw, mojo_device)
    i_ref, i_our = both(idx, hw, mojo_device)
    bench.run(
        lambda: F.max_unpool2d(
            p_ref, i_ref, (k, k), (stride, stride), (pad, pad), (h, w)
        ),
        lambda: F.max_unpool2d(
            p_our, i_our, (k, k), (stride, stride), (pad, pad), (h, w)
        ),
        flops=float(x.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", POOL3D_SHAPES)
@pytest.mark.bench_op("max_unpool3d")
def test_max_unpool3d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c, d, h, w, k, stride, pad = POOL3D_SHAPES[shape_id]
    x = torch.randn(n, c, d, h, w, dtype=DTYPES[dtype_id])
    pooled, idx = F.max_pool3d(x.float(), k, stride, pad, return_indices=True)
    pooled = pooled.to(x.dtype)
    p_ref, p_our = both(pooled, hw, mojo_device)
    i_ref, i_our = both(idx, hw, mojo_device)
    size = (d, h, w)
    bench.run(
        lambda: F.max_unpool3d(p_ref, i_ref, (k,) * 3, (stride,) * 3, (pad,) * 3, size),
        lambda: F.max_unpool3d(p_our, i_our, (k,) * 3, (stride,) * 3, (pad,) * 3, size),
        flops=float(x.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", FOLD_SHAPES)
@pytest.mark.bench_op("im2col")
def test_im2col(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c, h, w, k, stride, pad = FOLD_SHAPES[shape_id]
    x_ref, x_our = both(
        torch.randn(n, c, h, w, dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: F.unfold(x_ref, k, padding=pad, stride=stride),
        lambda: F.unfold(x_our, k, padding=pad, stride=stride),
        flops=float(x_ref.numel() * k * k),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", FOLD_SHAPES)
@pytest.mark.bench_op("col2im")
def test_col2im(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    n, c, h, w, k, stride, pad = FOLD_SHAPES[shape_id]
    cols = F.unfold(
        torch.randn(n, c, h, w, dtype=DTYPES[dtype_id]), k, padding=pad, stride=stride
    )
    c_ref, c_our = both(cols, hw, mojo_device)
    bench.run(
        lambda: F.fold(c_ref, (h, w), k, padding=pad, stride=stride),
        lambda: F.fold(c_our, (h, w), k, padding=pad, stride=stride),
        flops=float(cols.numel()),
    )
