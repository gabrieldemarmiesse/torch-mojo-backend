"""Softmax-family benchmarks: _softmax, _log_softmax, its backward, and
gelu_backward.

Driven through torch.ops.aten so the exact registered entry point is
pinned (F.softmax would reach the same _softmax, but via softmax.int —
which shares the fast impl and is documented as covered by these nodes).
Shapes: a training-shaped batch (32768x1024), the nanoGPT vocab row
(768x50304, the log-softmax regime the eager campaign tuned), and the
small awkward 357x789.
"""

from __future__ import annotations

import pytest
import torch
from bench_lib.cases import DTYPES, both, unit_interval
from bench_lib.check import Bench
from bench_lib.hw import Hardware

SHAPES: dict[str, tuple[int, int]] = {
    "S_32768x1024": (32768, 1024),
    "S_768x50304": (768, 50304),
    "A_357x789": (357, 789),
}
ELEM_SHAPES: dict[str, tuple[int, ...]] = {
    "C_16777216": (16777216,),
    "A_357x789": (357, 789),
}

COVERS: dict[str, str] = {
    "aten::_softmax": "test_softmax",
    "aten::_log_softmax": "test_log_softmax",
    "aten::_log_softmax_backward_data": "test_log_softmax_backward",
    "aten::gelu_backward": "test_gelu_backward",
    "aten::tanh_backward": "test_tanh_backward_f32",
    "aten::tanh_backward.grad_input": "test_tanh_backward_f32",
}

SKIPPED: dict[str, str] = {}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("_softmax")
def test_softmax(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our = both(
        torch.randn(SHAPES[shape_id], dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: torch.ops.aten._softmax(x_ref, -1, False),
        lambda: torch.ops.aten._softmax(x_our, -1, False),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("_log_softmax")
def test_log_softmax(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our = both(
        torch.randn(SHAPES[shape_id], dtype=DTYPES[dtype_id]), hw, mojo_device
    )
    bench.run(
        lambda: torch.ops.aten._log_softmax(x_ref, -1, False),
        lambda: torch.ops.aten._log_softmax(x_our, -1, False),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("_log_softmax_backward_data")
def test_log_softmax_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(torch.randn(SHAPES[shape_id], dtype=dtype), hw, mojo_device)
    g_ref, g_our = both(torch.randn(SHAPES[shape_id], dtype=dtype), hw, mojo_device)
    out_ref = torch.ops.aten._log_softmax(x_ref, -1, False)
    out_our = torch.ops.aten._log_softmax(x_our, -1, False)
    bench.run(
        lambda: torch.ops.aten._log_softmax_backward_data(g_ref, out_ref, -1, dtype),
        lambda: torch.ops.aten._log_softmax_backward_data(g_our, out_our, -1, dtype),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", ELEM_SHAPES)
def test_gelu_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    dtype = DTYPES[dtype_id]
    shape = ELEM_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, dtype), hw, mojo_device)
    g_ref, g_our = both(unit_interval(shape, dtype), hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.gelu_backward(g_ref, x_ref),
        lambda: torch.ops.aten.gelu_backward(g_our, x_our),
        flops=float(x_ref.numel()),
    )


_TANH_BACKWARD_CASES = {
    "small1600": (1600, 0),
    "large6553600": (6553600, 0),
    "large5242880": (5242880, 0),
    "awkward357x789": (357 * 789, 0),
    "large16777216": (16777216, 0),
    "offset357x789": (357 * 789, 1),
}


@pytest.mark.parametrize("shape_id", _TANH_BACKWARD_CASES)
@pytest.mark.parametrize("dtype_id", ["f32"])
@pytest.mark.parametrize("layout", ["contiguous_backward"])
@pytest.mark.bench_op("tanh_backward")
def test_tanh_backward_f32(
    shape_id: str,
    dtype_id: str,
    layout: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    size, offset = _TANH_BACKWARD_CASES[shape_id]
    values = torch.arange(size + 16, dtype=torch.int64)
    grad = ((values * 7919 + 13) % 65521).to(DTYPES[dtype_id]) / 65536 - 0.5
    output = ((values * 7919 + 29) % 65521).to(DTYPES[dtype_id]) / 65536 - 0.5
    gr, gm = both(grad, hw, mojo_device)
    yr, ym = both(output, hw, mojo_device)
    gr, gm = gr[offset : offset + size], gm[offset : offset + size]
    yr, ym = yr[offset : offset + size], ym[offset : offset + size]
    bench.run(
        lambda: torch.ops.aten.tanh_backward(gr, yr),
        lambda: torch.ops.aten.tanh_backward(gm, ym),
        flops=float(size * 3),
    )
