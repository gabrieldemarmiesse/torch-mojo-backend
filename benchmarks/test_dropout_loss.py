"""Dropout and NLL-loss benchmarks.

Dropout output values differ between legs (each backend runs its own
RNG) but the work — one mask + one scale kernel over N elements — is
data-independent, so the ratio is still well-defined.  The backward
reuses one forward's mask, un-timed.

nll_loss is driven through F.nll_loss the way a training loop reaches
it; its registered forms are the .output/.grad_input out-variants, which
this call path lands on.  N12288xC50304 is the padded-vocab nanoGPT loss
regime.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from bench_lib.cases import DTYPES, both, unit_interval
from bench_lib.check import Bench
from bench_lib.hw import Hardware

DROPOUT_SHAPES: dict[str, tuple[int, ...]] = {
    "C_16777216": (16777216,),
    "A_357x789": (357, 789),
}
# (batch, classes)
NLL_SHAPES: dict[str, tuple[int, int]] = {
    "N12288xC50304": (12288, 50304),
    "N4096xC1000": (4096, 1000),
}

COVERS: dict[str, str] = {
    "aten::native_dropout": "test_dropout",
    "aten::native_dropout_backward": "test_dropout_backward",
    "aten::nll_loss_forward.output": "test_nll_loss",
    "aten::nll_loss_backward.grad_input": "test_nll_loss_backward",
}

SKIPPED: dict[str, str] = {}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DROPOUT_SHAPES)
@pytest.mark.bench_op("native_dropout")
def test_dropout(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape = DROPOUT_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.native_dropout(x_ref, 0.5, True),
        lambda: torch.ops.aten.native_dropout(x_our, 0.5, True),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DROPOUT_SHAPES)
@pytest.mark.bench_op("native_dropout_backward")
def test_dropout_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    shape = DROPOUT_SHAPES[shape_id]
    g_ref, g_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    mask_ref, mask_our = both(torch.rand(shape) < 0.5, hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.native_dropout_backward(g_ref, mask_ref, 2.0),
        lambda: torch.ops.aten.native_dropout_backward(g_our, mask_our, 2.0),
        flops=float(g_ref.numel()),
    )


def _nll_case(
    shape_id: str, dtype_id: str, hw: Hardware, mojo: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, classes = NLL_SHAPES[shape_id]
    logp = torch.log_softmax(
        torch.randn(batch, classes, dtype=torch.float32), dim=-1
    ).to(DTYPES[dtype_id])
    target = torch.randint(0, classes, (batch,))
    lp_ref, lp_our = both(logp, hw, mojo)
    t_ref, t_our = both(target, hw, mojo)
    return lp_ref, lp_our, t_ref, t_our


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", NLL_SHAPES)
@pytest.mark.bench_op("nll_loss_forward")
def test_nll_loss(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    lp_ref, lp_our, t_ref, t_our = _nll_case(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: F.nll_loss(lp_ref, t_ref),
        lambda: F.nll_loss(lp_our, t_our),
        flops=float(lp_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", NLL_SHAPES)
@pytest.mark.bench_op("nll_loss_backward")
def test_nll_loss_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
) -> None:
    lp_ref, lp_our, t_ref, t_our = _nll_case(shape_id, dtype_id, hw, mojo_device)
    # Synthetic grad/total_weight with the exact values nll_loss_forward
    # produces for mean reduction and no class weights (grad 1, total_weight
    # = batch): building them directly keeps the setup off the mojo forward,
    # whose missing dtype support must not error this backward benchmark.
    batch = float(NLL_SHAPES[shape_id][0])
    dtype = DTYPES[dtype_id]
    g_ref, g_our = both(torch.tensor(1.0, dtype=dtype), hw, mojo_device)
    tw_ref, tw_our = both(torch.tensor(batch, dtype=dtype), hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.nll_loss_backward(
            g_ref, lp_ref, t_ref, None, 1, -100, tw_ref
        ),
        lambda: torch.ops.aten.nll_loss_backward(
            g_our, lp_our, t_our, None, 1, -100, tw_our
        ),
        flops=float(lp_ref.numel()),
    )
