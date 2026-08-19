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

# (N, C_in, H, W, C_out, kernel, stride, padding, dilation)
#
# The first five shapes are all stride-1 dense convs, which is exactly the
# domain of the implicit-GEMM route; the block after them deliberately leaves
# that domain (stride 2, dilation, 1x1-with-stride, batch 1, ragged channels)
# so the recorded numbers describe the whole op and not one fast path.
CONV_SHAPES: dict[str, tuple[int, int, int, int, int, int, int, int, int]] = {
    "N32xC64x56x56_K64k3s1": (32, 64, 56, 56, 64, 3, 1, 1, 1),
    "N8xC3x224x224_K64k7s2": (8, 3, 224, 224, 64, 7, 2, 3, 1),
    # Deeper-C, small-spatial ResNet stage. Same implicit-GEMM domain as the
    # body shape above but the output row pitch (hw*2 = 392 B) is not TMA
    # store aligned, so it exercises the scalar epilogue instead.
    "N32xC256x14x14_K256k3s1": (32, 256, 14, 14, 256, 3, 1, 1, 1),
    # Nothing divides anything: C=48 pads to 64 (25% wasted reduction),
    # out_c=80 is ragged against BM=64, H/W are both odd. Worst-case shape
    # inside the implicit-GEMM domain.
    "N7xC48x39x53_K80k3s1": (7, 48, 39, 53, 80, 3, 1, 1, 1),
    # 1x1 control: the materialized route already skips im2col for this case
    # (the NCHW input IS the col matrix), so the implicit-GEMM route must
    # decline it rather than regress it with an added NHWC pass.
    "N32xC256x56x56_K64k1s1": (32, 256, 56, 56, 64, 1, 1, 0, 1),
    # --- regimes OUTSIDE the implicit-GEMM domain -------------------------
    # ResNet-50 stage transition. stride != 1 is declined by the implicit
    # GEMM, so this is the materialized im2col route, whose gather is the
    # dominant cost at stride 2 (measured: 362 of 409 us here).
    "N32xC128x28x28_K128k3s2": (32, 128, 28, 28, 128, 3, 2, 1, 1),
    # ResNet-50 shortcut: a 1x1 that CANNOT reuse the in-place col matrix
    # (that fast path needs stride 1), so a 1x1 still materializes.
    "N32xC256x14x14_K512k1s2": (32, 256, 14, 14, 512, 1, 2, 0, 1),
    # Deepest ResNet-50 stage: 7x7 spatial with C=K=512 puts the per-call
    # weight repack (a 4.7 MB tensor) on the critical path, not the GEMM.
    "N32xC512x7x7_K512k3s1": (32, 512, 7, 7, 512, 3, 1, 1, 1),
    # VGG-style big spatial: the largest activation in the suite, where the
    # implicit GEMM's saved column buffer matters most.
    "N8xC128x112x112_K128k3s1": (8, 128, 112, 112, 128, 3, 1, 1, 1),
    # Batch-1 inference variants of the body and mid shapes: one sample is
    # 1/32 of the GEMM rows, i.e. the regime where a tile grid either fills
    # the machine or does not.
    "N1xC64x56x56_K64k3s1": (1, 64, 56, 56, 64, 3, 1, 1, 1),
    "N1xC256x14x14_K256k3s1": (1, 256, 14, 14, 256, 3, 1, 1, 1),
    # C and K both indivisible by the 64-channel tile, at stride 2.
    "N16xC96x28x28_K192k3s2": (16, 96, 28, 28, 192, 3, 2, 1, 1),
    # Dilation (segmentation / ASPP). The implicit GEMM declines dilation by
    # design; this node is what pins the fallback's cost for it.
    "N16xC64x32x32_K64k3s1d2": (16, 64, 32, 32, 64, 3, 1, 2, 2),
}
# Backward is measured on a subset of the same CONV_SHAPES geometries, one
# node per GRADIENT: the three gradients are three different kernels (a
# col2im-fed GEMM, a transposed-B GEMM plus a batch reduce, and a plain
# reduction) whose ratios move independently, so folding them into one node
# would hide which of them regressed.  The subset spans the regimes that
# matter to the backward specifically: the body/mid shapes the forward is
# tuned on, the thin-C stem, an awkward one, a stride-2 one (the only
# regime where col2im's tap loop mostly misses), and the deepest stage,
# where the weight gradient's per-sample partials buffer is at its largest
# relative to the column buffer.
CONV_BACKWARD_SHAPES: tuple[str, ...] = (
    "N32xC64x56x56_K64k3s1",
    "N8xC3x224x224_K64k7s2",
    "N32xC256x14x14_K256k3s1",
    "N7xC48x39x53_K80k3s1",
    "N32xC128x28x28_K128k3s2",
    "N32xC512x7x7_K512k3s1",
)
# output_mask, one gradient at a time.  This is the suite's `layout` axis:
# `bench_key` builds the baseline path op/dtype/shape/layout from the
# parametrize ids, so naming the gradient selector `layout` is what keeps the
# three gradients of one shape three separate entries (and lands them next to
# each other in the tree) instead of three writes to one key.
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

COVERS: dict[str, str] = {
    "aten::convolution": "test_conv2d",
    "aten::convolution_backward": "test_conv2d_backward",
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
    n, c_in, h, w, c_out, k, stride, pad, dil = CONV_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(n, c_in, h, w, dtype=dtype), hw, mojo_device)
    w_ref, w_our = both(
        torch.randn(c_out, c_in, k, k, dtype=dtype) * 0.1, hw, mojo_device
    )
    b_ref, b_our = both(torch.randn(c_out, dtype=dtype), hw, mojo_device)
    eff_k = dil * (k - 1) + 1
    h_out = (h + 2 * pad - eff_k) // stride + 1
    w_out = (w + 2 * pad - eff_k) // stride + 1
    flops = 2.0 * n * c_out * h_out * w_out * c_in * k * k
    bench.run(
        lambda: F.conv2d(x_ref, w_ref, b_ref, stride, pad, dil),
        lambda: F.conv2d(x_our, w_our, b_our, stride, pad, dil),
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
) -> None:
    n, c_in, h, w, c_out, k, stride, pad, dil = CONV_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    eff_k = dil * (k - 1) + 1
    h_out = (h + 2 * pad - eff_k) // stride + 1
    w_out = (w + 2 * pad - eff_k) // stride + 1
    grad_ref, grad_our = both(
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
        [dil, dil],
        False,
        [0, 0],
        1,
        CONV_BACKWARD_GRADS[layout],
    )
    # The bias gradient is a pure reduction over grad_output, so its "flops"
    # is its element count, the same convention the pooling nodes use.
    flops = (
        float(n * c_out * h_out * w_out)
        if layout == "bgrad"
        else 2.0 * n * c_out * h_out * w_out * c_in * k * k
    )
    bench.run(
        lambda: torch.ops.aten.convolution_backward(grad_ref, x_ref, w_ref, *tail),
        lambda: torch.ops.aten.convolution_backward(grad_our, x_our, w_our, *tail),
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
