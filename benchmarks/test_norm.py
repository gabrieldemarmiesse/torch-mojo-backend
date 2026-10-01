"""Normalization benchmarks: layer norm (fwd/bwd), batch norm (training
and inference forms), SyncBatchNorm's building blocks, group norm.

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
    "aten::batch_norm_stats": "test_batch_norm_stats",
    "aten::batch_norm_update_stats": "test_batch_norm_update_stats",
    "aten::batch_norm_elemt": "test_batch_norm_elemt",
    "aten::batch_norm_gather_stats_with_counts": "test_batch_norm_gather_stats",
    "aten::batch_norm_backward_reduce": "test_batch_norm_backward_reduce",
    "aten::batch_norm_backward_elemt": "test_batch_norm_backward_elemt",
}

_BN_ALIAS = (
    "native_batch_norm's own route under another schema (tmb/ops/"
    "batch_norm.mojo dispatches to it, as Normalization.cu's non-cuDNN path "
    "does): the kernels test_batch_norm / test_batch_norm_backward measure"
)
_BN_OUT = (
    "out= plumbing over the functional overload the batch-norm benchmarks "
    "measure (computed fresh, then resized and copied)"
)
SKIPPED: dict[str, str] = {
    "aten::_native_batch_norm_legit": _BN_ALIAS,
    "aten::_native_batch_norm_legit.no_stats": _BN_ALIAS,
    "aten::_batch_norm_with_update": _BN_ALIAS,
    "aten::batch_norm_backward": _BN_ALIAS,
    "aten::native_batch_norm.out": _BN_OUT,
    "aten::_native_batch_norm_legit.out": _BN_OUT,
    "aten::_native_batch_norm_legit.no_stats_out": _BN_OUT,
    "aten::_batch_norm_with_update.out": _BN_OUT,
    "aten::batch_norm_elemt.out": _BN_OUT,
    "aten::batch_norm_gather_stats": (
        "batch_norm_gather_stats_with_counts (test_batch_norm_gather_stats) "
        "with a filled counts vector"
    ),
}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", LN_SHAPES)
@pytest.mark.bench_op("native_layer_norm")
def test_layer_norm(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
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
):
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
):
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
):
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
):
    refs, ours = _bn_operands(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten._native_batch_norm_legit_no_training(*refs, 0.1, 1e-5),
        lambda: torch.ops.aten._native_batch_norm_legit_no_training(*ours, 0.1, 1e-5),
        flops=float(refs[0].numel()),
    )


def _bn_operands_nhwc(
    shape_id: str, dtype_id: str, hw: Hardware, mojo: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """`_bn_operands` with the activation channels-last (NHWC), the layout
    a channels-last conv net hands batch norm."""
    refs, ours = _bn_operands(shape_id, dtype_id, hw, mojo)
    refs[0] = refs[0].to(memory_format=torch.channels_last)
    ours[0] = ours[0].to(memory_format=torch.channels_last)
    return refs, ours


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", BN_SHAPES)
@pytest.mark.bench_op("native_batch_norm")
def test_batch_norm_nhwc(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    refs, ours = _bn_operands_nhwc(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.native_batch_norm(*refs, True, 0.1, 1e-5),
        lambda: torch.ops.aten.native_batch_norm(*ours, True, 0.1, 1e-5),
        flops=float(refs[0].numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", BN_SHAPES)
@pytest.mark.bench_op("_native_batch_norm_legit_no_training")
def test_batch_norm_inference_nhwc(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    refs, ours = _bn_operands_nhwc(shape_id, dtype_id, hw, mojo_device)
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
):
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
):
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


# ---------------------------------------------------------------------------
# SyncBatchNorm building blocks: the same activation shapes, the statistics
# as the float32 vectors the per-replica forward produces.
# ---------------------------------------------------------------------------

# Replicas merged by batch_norm_gather_stats*, as in an 8-GPU DDP job.
WORLD = 8


def _sync_operands(
    shape_id: str, dtype_id: str, hw: Hardware, mojo: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    n, c, h, w = BN_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    host = {
        "x": torch.randn(n, c, h, w, dtype=dtype),
        "g": torch.randn(n, c, h, w, dtype=dtype),
        "mean": torch.randn(c),
        "invstd": torch.rand(c) + 0.5,
        "weight": torch.randn(c),
        "bias": torch.randn(c),
        "sum_dy": torch.randn(c),
        "sum_dy_xmu": torch.randn(c),
        "running_mean": torch.zeros(c, dtype=dtype),
        "running_var": torch.ones(c, dtype=dtype),
        "means": torch.randn(WORLD, c),
        "invstds": torch.rand(WORLD, c) + 0.5,
        "counts": torch.full((WORLD,), float(n * h * w), dtype=dtype),
        "count": torch.full((WORLD,), n * h * w, dtype=torch.int32),
    }
    refs: dict[str, torch.Tensor] = {}
    ours: dict[str, torch.Tensor] = {}
    for key, tensor in host.items():
        refs[key], ours[key] = both(tensor, hw, mojo)
    return refs, ours


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", BN_SHAPES)
@pytest.mark.bench_op("batch_norm_stats")
def test_batch_norm_stats(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    r, o = _sync_operands(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.batch_norm_stats(r["x"], 1e-5),
        lambda: torch.ops.aten.batch_norm_stats(o["x"], 1e-5),
        flops=float(r["x"].numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", BN_SHAPES)
@pytest.mark.bench_op("batch_norm_update_stats")
def test_batch_norm_update_stats(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    r, o = _sync_operands(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.batch_norm_update_stats(
            r["x"], r["running_mean"], r["running_var"], 0.1
        ),
        lambda: torch.ops.aten.batch_norm_update_stats(
            o["x"], o["running_mean"], o["running_var"], 0.1
        ),
        flops=float(r["x"].numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", BN_SHAPES)
@pytest.mark.bench_op("batch_norm_elemt")
def test_batch_norm_elemt(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    r, o = _sync_operands(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.batch_norm_elemt(
            r["x"], r["weight"], r["bias"], r["mean"], r["invstd"], 1e-5
        ),
        lambda: torch.ops.aten.batch_norm_elemt(
            o["x"], o["weight"], o["bias"], o["mean"], o["invstd"], 1e-5
        ),
        flops=float(r["x"].numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", BN_SHAPES)
@pytest.mark.bench_op("batch_norm_gather_stats_with_counts")
def test_batch_norm_gather_stats(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    r, o = _sync_operands(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.batch_norm_gather_stats_with_counts(
            r["x"],
            r["means"],
            r["invstds"],
            r["running_mean"],
            r["running_var"],
            0.1,
            1e-5,
            r["counts"],
        ),
        lambda: torch.ops.aten.batch_norm_gather_stats_with_counts(
            o["x"],
            o["means"],
            o["invstds"],
            o["running_mean"],
            o["running_var"],
            0.1,
            1e-5,
            o["counts"],
        ),
        flops=float(r["means"].numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", BN_SHAPES)
@pytest.mark.bench_op("batch_norm_backward_reduce")
def test_batch_norm_backward_reduce(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    r, o = _sync_operands(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.batch_norm_backward_reduce(
            r["g"], r["x"], r["mean"], r["invstd"], r["weight"], True, True, True
        ),
        lambda: torch.ops.aten.batch_norm_backward_reduce(
            o["g"], o["x"], o["mean"], o["invstd"], o["weight"], True, True, True
        ),
        flops=float(r["x"].numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", BN_SHAPES)
@pytest.mark.bench_op("batch_norm_backward_elemt")
def test_batch_norm_backward_elemt(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    r, o = _sync_operands(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.batch_norm_backward_elemt(
            r["g"],
            r["x"],
            r["mean"],
            r["invstd"],
            r["weight"],
            r["sum_dy"],
            r["sum_dy_xmu"],
            r["count"],
        ),
        lambda: torch.ops.aten.batch_norm_backward_elemt(
            o["g"],
            o["x"],
            o["mean"],
            o["invstd"],
            o["weight"],
            o["sum_dy"],
            o["sum_dy_xmu"],
            o["count"],
        ),
        flops=float(r["x"].numel()),
    )
