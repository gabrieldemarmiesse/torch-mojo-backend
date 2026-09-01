"""Normalization benchmarks: layer norm (fwd/bwd), batch norm (training
and inference forms), group norm.

Driven via torch.ops.aten so the registered entry point is pinned; the
backward gets its mean/rstd from a single un-timed forward call.  Batch
norm's training form updates running stats in place — both legs do,
symmetrically.
"""

from __future__ import annotations

import pytest
import torch
from bench_lib.cases import DTYPES, both
from bench_lib.check import Bench
from bench_lib.hw import Hardware

LN_SHAPES: dict[str, tuple[int, int]] = {
    "B32768xD1024": (32768, 1024),
    "A_357x789": (357, 789),
}
BN_SHAPES: dict[str, tuple[int, int, int, int]] = {
    "N32xC64xH112xW112": (32, 64, 112, 112),
    "N8xC256xH28xW28": (8, 256, 28, 28),
}
# (N, C, H, W, groups)
GN_SHAPES: dict[str, tuple[int, int, int, int, int]] = {
    "N32xC64xH56xW56_G32": (32, 64, 56, 56, 32),
    "N8xC256xH14xW14_G32": (8, 256, 14, 14, 32),
}

COVERS: dict[str, str] = {
    "aten::native_layer_norm": "test_layer_norm",
    "aten::native_layer_norm_backward": "test_layer_norm_backward",
    "aten::native_batch_norm": "test_batch_norm",
    "aten::native_batch_norm_backward": "test_batch_norm_backward",
    "aten::_native_batch_norm_legit_no_training": "test_batch_norm_inference",
    "aten::native_group_norm": "test_group_norm",
    "aten::native_group_norm_backward": "test_group_norm_backward",
}

SKIPPED: dict[str, str] = {}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", LN_SHAPES)
@pytest.mark.bench_op("native_layer_norm")
def test_layer_norm(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    rows, dim = LN_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(rows, dim, dtype=dtype), hw, mojo_device)
    w_ref, w_our = both(torch.randn(dim, dtype=dtype), hw, mojo_device)
    b_ref, b_our = both(torch.randn(dim, dtype=dtype), hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.native_layer_norm(x_ref, [dim], w_ref, b_ref, 1e-5),
        lambda: torch.ops.aten.native_layer_norm(x_our, [dim], w_our, b_our, 1e-5),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", LN_SHAPES)
@pytest.mark.bench_op("native_layer_norm_backward")
def test_layer_norm_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    rows, dim = LN_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(rows, dim, dtype=dtype), hw, mojo_device)
    w_ref, w_our = both(torch.randn(dim, dtype=dtype), hw, mojo_device)
    b_ref, b_our = both(torch.randn(dim, dtype=dtype), hw, mojo_device)
    g_ref, g_our = both(torch.randn(rows, dim, dtype=dtype), hw, mojo_device)
    _, mean_ref, rstd_ref = torch.ops.aten.native_layer_norm(
        x_ref, [dim], w_ref, b_ref, 1e-5
    )
    _, mean_our, rstd_our = torch.ops.aten.native_layer_norm(
        x_our, [dim], w_our, b_our, 1e-5
    )
    mask = [True, True, True]
    bench.run(
        lambda: torch.ops.aten.native_layer_norm_backward(
            g_ref, x_ref, [dim], mean_ref, rstd_ref, w_ref, b_ref, mask
        ),
        lambda: torch.ops.aten.native_layer_norm_backward(
            g_our, x_our, [dim], mean_our, rstd_our, w_our, b_our, mask
        ),
        flops=float(x_ref.numel()),
    )


def _bn_operands(
    shape_id: str, dtype_id: str, hw: Hardware, mojo: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    n, c, h, w = BN_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x = torch.randn(n, c, h, w, dtype=dtype)
    weight = torch.randn(c, dtype=dtype)
    bias = torch.randn(c, dtype=dtype)
    running_mean = torch.zeros(c, dtype=torch.float32)
    running_var = torch.ones(c, dtype=torch.float32)
    refs, ours = [], []
    for tensor in (x, weight, bias, running_mean, running_var):
        ref, our = both(tensor, hw, mojo)
        refs.append(ref)
        ours.append(our)
    return refs, ours


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", BN_SHAPES)
@pytest.mark.bench_op("native_batch_norm")
def test_batch_norm(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    refs, ours = _bn_operands(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.native_batch_norm(*refs, True, 0.1, 1e-5),
        lambda: torch.ops.aten.native_batch_norm(*ours, True, 0.1, 1e-5),
        flops=float(refs[0].numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", BN_SHAPES)
@pytest.mark.bench_op("native_batch_norm_backward")
def test_batch_norm_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    # ATen's backward checks that every per-channel buffer shares ONE dtype
    # (`check_mixed_data_type`), so the half-precision case is the layout AMP
    # actually produces: a half input with float32 affine and running stats.
    # `_bn_operands` gives the affine the input's dtype, which the forward
    # accepts and the backward rejects, so the operands are built here.
    n, c, h, w = BN_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    param_dtype = torch.float32
    x_ref, x_our = both(torch.randn(n, c, h, w, dtype=dtype), hw, mojo_device)
    g_ref, g_our = both(torch.randn(n, c, h, w, dtype=dtype), hw, mojo_device)
    w_ref, w_our = both(torch.randn(c, dtype=param_dtype), hw, mojo_device)
    b_ref, b_our = both(torch.randn(c, dtype=param_dtype), hw, mojo_device)
    rm_ref, rm_our = both(torch.zeros(c, dtype=param_dtype), hw, mojo_device)
    rv_ref, rv_our = both(torch.ones(c, dtype=param_dtype), hw, mojo_device)
    refs = [x_ref, w_ref, b_ref, rm_ref, rv_ref]
    ours = [x_our, w_our, b_our, rm_our, rv_our]
    # Un-timed forward, exactly as the layer-norm backward above: the saved
    # statistics are an input to the op under measurement, not part of it.
    _, mean_ref, rstd_ref = torch.ops.aten.native_batch_norm(*refs, True, 0.1, 1e-5)
    _, mean_our, rstd_our = torch.ops.aten.native_batch_norm(*ours, True, 0.1, 1e-5)
    mask = [True, True, True]
    bench.run(
        lambda: torch.ops.aten.native_batch_norm_backward(
            g_ref, x_ref, w_ref, rm_ref, rv_ref, mean_ref, rstd_ref, True, 1e-5, mask
        ),
        lambda: torch.ops.aten.native_batch_norm_backward(
            g_our, x_our, w_our, rm_our, rv_our, mean_our, rstd_our, True, 1e-5, mask
        ),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", BN_SHAPES)
@pytest.mark.bench_op("_native_batch_norm_legit_no_training")
def test_batch_norm_inference(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    refs, ours = _bn_operands(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten._native_batch_norm_legit_no_training(*refs, 0.1, 1e-5),
        lambda: torch.ops.aten._native_batch_norm_legit_no_training(*ours, 0.1, 1e-5),
        flops=float(refs[0].numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", GN_SHAPES)
@pytest.mark.bench_op("native_group_norm")
def test_group_norm(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    n, c, h, w, groups = GN_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(n, c, h, w, dtype=dtype), hw, mojo_device)
    w_ref, w_our = both(torch.randn(c, dtype=dtype), hw, mojo_device)
    b_ref, b_our = both(torch.randn(c, dtype=dtype), hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.native_group_norm(
            x_ref, w_ref, b_ref, n, c, h * w, groups, 1e-5
        ),
        lambda: torch.ops.aten.native_group_norm(
            x_our, w_our, b_our, n, c, h * w, groups, 1e-5
        ),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", GN_SHAPES)
@pytest.mark.bench_op("native_group_norm_backward")
def test_group_norm_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    n, c, h, w, groups = GN_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(n, c, h, w, dtype=dtype), hw, mojo_device)
    w_ref, w_our = both(torch.randn(c, dtype=dtype), hw, mojo_device)
    b_ref, b_our = both(torch.randn(c, dtype=dtype), hw, mojo_device)
    g_ref, g_our = both(torch.randn(n, c, h, w, dtype=dtype), hw, mojo_device)
    _, mean_ref, rstd_ref = torch.ops.aten.native_group_norm(
        x_ref, w_ref, b_ref, n, c, h * w, groups, 1e-5
    )
    _, mean_our, rstd_our = torch.ops.aten.native_group_norm(
        x_our, w_our, b_our, n, c, h * w, groups, 1e-5
    )
    mask = [True, True, True]
    bench.run(
        lambda: torch.ops.aten.native_group_norm_backward(
            g_ref, x_ref, mean_ref, rstd_ref, w_ref, n, c, h * w, groups, mask
        ),
        lambda: torch.ops.aten.native_group_norm_backward(
            g_our, x_our, mean_our, rstd_our, w_our, n, c, h * w, groups, mask
        ),
        flops=float(x_ref.numel()),
    )
