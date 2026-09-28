"""Dropout, NLL-loss and elementwise-loss benchmarks.

Dropout output values differ between legs (each backend runs its own
RNG) but the work — one mask + one scale kernel over N elements — is
data-independent, so the ratio is still well-defined.  The backward
reuses one forward's mask, un-timed.

nll_loss is driven through F.nll_loss the way a training loop reaches
it; its registered forms are the .output/.grad_input out-variants, which
this call path lands on.  N12288xC50304 is the padded-vocab nanoGPT loss
regime.

The elementwise losses (mse, smooth_l1, huber, binary_cross_entropy and
its logits form) run with reduction='mean', the default a training loop
reaches: one elementwise kernel plus the full mean reduction. Their
backwards take the 0-d grad a mean hands down.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from bench_lib.cases import DTYPES, both, op_params, unit_interval
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

# Operands in unit_interval: inside binary_cross_entropy's [0, 1] domain.
LOSS_OPS = {
    "mse_loss": F.mse_loss,
    "smooth_l1_loss": lambda x, t: F.smooth_l1_loss(x, t, beta=0.5),
    "huber_loss": lambda x, t: F.huber_loss(x, t, delta=0.5),
    "binary_cross_entropy": F.binary_cross_entropy,
    "binary_cross_entropy_with_logits": F.binary_cross_entropy_with_logits,
}
# (grad, input, target) with the 0-d grad of reduction='mean' (1).
LOSS_BACKWARD_OPS = {
    "mse_loss_backward": lambda g, x, t: torch.ops.aten.mse_loss_backward(g, x, t, 1),
    "smooth_l1_loss_backward": lambda g, x, t: torch.ops.aten.smooth_l1_loss_backward(
        g, x, t, 1, 0.5
    ),
    "huber_loss_backward": lambda g, x, t: torch.ops.aten.huber_loss_backward(
        g, x, t, 1, 0.5
    ),
    "binary_cross_entropy_backward": lambda g, x, t: (
        torch.ops.aten.binary_cross_entropy_backward(g, x, t, None, 1)
    ),
}

COVERS: dict[str, str] = (
    {
        "aten::native_dropout": "test_dropout",
        "aten::native_dropout_backward": "test_dropout_backward",
        "aten::nll_loss_forward.output": "test_nll_loss",
        "aten::nll_loss_backward.grad_input": "test_nll_loss_backward",
    }
    | {f"aten::{name}": "test_loss" for name in LOSS_OPS}
    | {f"aten::{name}": "test_loss_backward" for name in LOSS_BACKWARD_OPS}
)

_LOSS_OUT = (
    "out= plumbing over the loss kernel test_loss / test_loss_backward "
    "measures (computed straight into a fitting destination, or reduced "
    "and copied)"
)
SKIPPED: dict[str, str] = {
    f"aten::{name}": _LOSS_OUT
    for name in (
        "binary_cross_entropy.out",
        "binary_cross_entropy_backward.grad_input",
        "binary_cross_entropy_with_logits.out",
        "huber_loss.out",
        "huber_loss_backward.out",
        "mse_loss.out",
        "mse_loss_backward.grad_input",
        "smooth_l1_loss.out",
        "smooth_l1_loss_backward.grad_input",
    )
}


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DROPOUT_SHAPES)
@pytest.mark.bench_op("native_dropout")
def test_dropout(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
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
):
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
):
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
):
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


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DROPOUT_SHAPES)
@pytest.mark.parametrize("op_name", op_params(LOSS_OPS))
def test_loss(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    fn = LOSS_OPS[op_name]
    shape = DROPOUT_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    t_ref, t_our = both(
        unit_interval(shape, DTYPES[dtype_id]).flip(-1), hw, mojo_device
    )
    bench.run(
        lambda: fn(x_ref, t_ref), lambda: fn(x_our, t_our), flops=float(x_ref.numel())
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DROPOUT_SHAPES)
@pytest.mark.parametrize("op_name", op_params(LOSS_BACKWARD_OPS))
def test_loss_backward(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    fn = LOSS_BACKWARD_OPS[op_name]
    shape = DROPOUT_SHAPES[shape_id]
    dtype = DTYPES[dtype_id]
    x_ref, x_our = both(unit_interval(shape, dtype), hw, mojo_device)
    t_ref, t_our = both(unit_interval(shape, dtype).flip(-1), hw, mojo_device)
    g_ref, g_our = both(torch.tensor(0.5, dtype=dtype), hw, mojo_device)
    bench.run(
        lambda: fn(g_ref, x_ref, t_ref),
        lambda: fn(g_our, x_our, t_our),
        flops=float(x_ref.numel()),
    )
