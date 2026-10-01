"""Dropout, NLL-loss and elementwise-loss benchmarks.

Dropout output values differ between legs (each backend runs its own
RNG) but the work — one mask + one scale kernel over N elements — is
data-independent, so the ratio is still well-defined.  The backward
reuses one forward's mask, un-timed.

nll_loss is driven through F.nll_loss the way a training loop reaches
it (the functional nll_loss_forward / nll_loss_backward).  N12288xC50304
is the padded-vocab nanoGPT loss regime.  nll_loss2d is the segmentation
regime of F.cross_entropy over [N, C, H, W] logits; the margin losses run
at a classifier's (batch, classes); CTC at a speech recognizer's (time,
batch, vocabulary) with targets a quarter of the input length.

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
        "aten::_masked_scale": "test_masked_scale",
        "aten::nll_loss_forward": "test_nll_loss",
        "aten::nll_loss_backward": "test_nll_loss_backward",
        "aten::nll_loss2d_forward": "test_nll_loss2d",
        "aten::nll_loss2d_backward": "test_nll_loss2d_backward",
        "aten::multi_margin_loss": "test_multi_margin_loss",
        "aten::multi_margin_loss_backward": "test_multi_margin_loss_backward",
        "aten::multilabel_margin_loss_forward": "test_multilabel_margin_loss",
        "aten::multilabel_margin_loss_backward": (
            "test_multilabel_margin_loss_backward"
        ),
        "aten::_ctc_loss": "test_ctc_loss",
        "aten::_ctc_loss_backward": "test_ctc_loss_backward",
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
    "aten::_fused_dropout": (
        "the NativeDropout kernel test_dropout measures, with a uint8 mask "
        "and an explicit generator"
    ),
    "aten::_sample_dirichlet": (
        "test_sampler's _standard_gamma draw, a sum over the last dim and "
        "one pointwise ratio"
    ),
    "aten::bernoulli.out": (
        "out.resize_ + the BernoulliTensor kernel bernoulli_.Tensor launches"
    ),
    "aten::_ctc_loss.Tensor": (
        "a synchronizing read of the lengths, then the _ctc_loss kernel "
        "test_ctc_loss measures"
    ),
    "aten::_ctc_loss_backward.Tensor": (
        "a synchronizing read of the lengths, then the _ctc_loss_backward "
        "kernels test_ctc_loss_backward measures"
    ),
    "aten::_fill_mem_eff_dropout_mask_": (
        "a testing hook of memory-efficient attention (upstream: 'only used "
        "for testing, not much attention is paid to performance')"
    ),
} | {
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
        "nll_loss_forward.output",
        "nll_loss_backward.grad_input",
        "nll_loss2d_forward.output",
        "nll_loss2d_backward.grad_input",
        "multi_margin_loss.out",
        "multi_margin_loss_backward.grad_input",
        "multilabel_margin_loss_forward.output",
        "multilabel_margin_loss_backward.grad_input",
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


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", DROPOUT_SHAPES)
@pytest.mark.bench_op("_masked_scale")
def test_masked_scale(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = DROPOUT_SHAPES[shape_id]
    x_ref, x_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    m_ref, m_our = both((torch.rand(shape) < 0.5).to(torch.uint8), hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten._masked_scale(x_ref, m_ref, 2.0),
        lambda: torch.ops.aten._masked_scale(x_our, m_our, 2.0),
        flops=float(x_ref.numel()),
    )


# Rejection samplers: parameters where every regime's loop is short
# (poisson's PTRS above 10, gamma's Marsaglia-Tsang, binomial's BTRS).
# (scale of the first operand, the call)
SAMPLER_OPS = {
    "poisson": (40.0, lambda a, b: torch.poisson(a)),
    "_standard_gamma": (4.0, lambda a, b: torch._standard_gamma(a)),
    "binomial": (100.0, torch.binomial),
}
COVERS |= {f"aten::{name}": "test_sampler" for name in SAMPLER_OPS}


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", DROPOUT_SHAPES)
@pytest.mark.parametrize("op_name", op_params(SAMPLER_OPS))
def test_sampler(
    op_name: str,
    shape_id: str,
    dtype_id: str,
    bench: Bench,
    hw: Hardware,
    mojo_device: torch.device,
):
    """The draws differ between legs (each backend its own stream) but the
    expected work per element is the same distribution's."""
    scale, fn = SAMPLER_OPS[op_name]
    shape = DROPOUT_SHAPES[shape_id]
    a_ref, a_our = both(unit_interval(shape, DTYPES[dtype_id]) * scale, hw, mojo_device)
    b_ref, b_our = both(unit_interval(shape, DTYPES[dtype_id]), hw, mojo_device)
    bench.run(
        lambda: fn(a_ref, b_ref), lambda: fn(a_our, b_our), flops=float(a_ref.numel())
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


# (N, C, H, W): a segmentation head's logits.
NLL2D_SHAPES: dict[str, tuple[int, int, int, int]] = {
    "N8xC21xH128xW128": (8, 21, 128, 128),
    "N3xC7xH57xW89": (3, 7, 57, 89),
}
# (batch, classes)
MARGIN_SHAPES: dict[str, tuple[int, int]] = {
    "N1024xC1000": (1024, 1000),
    "N357xC89": (357, 89),
}


def _nll2d_case(
    shape_id: str, dtype_id: str, hw: Hardware, mojo: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, c, h, w = NLL2D_SHAPES[shape_id]
    logp = torch.log_softmax(torch.randn(n, c, h, w), dim=1).to(DTYPES[dtype_id])
    target = torch.randint(0, c, (n, h, w))
    lp_ref, lp_our = both(logp, hw, mojo)
    t_ref, t_our = both(target, hw, mojo)
    return lp_ref, lp_our, t_ref, t_our


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", NLL2D_SHAPES)
@pytest.mark.bench_op("nll_loss2d_forward")
def test_nll_loss2d(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    lp_ref, lp_our, t_ref, t_our = _nll2d_case(shape_id, dtype_id, hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.nll_loss2d_forward(lp_ref, t_ref, None, 1, -100),
        lambda: torch.ops.aten.nll_loss2d_forward(lp_our, t_our, None, 1, -100),
        flops=float(t_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", NLL2D_SHAPES)
@pytest.mark.bench_op("nll_loss2d_backward")
def test_nll_loss2d_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    lp_ref, lp_our, t_ref, t_our = _nll2d_case(shape_id, dtype_id, hw, mojo_device)
    dtype = DTYPES[dtype_id]
    g_ref, g_our = both(torch.tensor(1.0, dtype=dtype), hw, mojo_device)
    tw = float(t_ref.numel())
    tw_ref, tw_our = both(torch.tensor(tw, dtype=dtype), hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.nll_loss2d_backward(
            g_ref, lp_ref, t_ref, None, 1, -100, tw_ref
        ),
        lambda: torch.ops.aten.nll_loss2d_backward(
            g_our, lp_our, t_our, None, 1, -100, tw_our
        ),
        flops=float(lp_ref.numel()),
    )


def _margin_case(
    shape_id: str, dtype_id: str, hw: Hardware, mojo: torch.device, multilabel: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, c = MARGIN_SHAPES[shape_id]
    x = torch.randn(n, c).to(DTYPES[dtype_id])
    if multilabel:
        # Four labels per sample, then the -1 terminator.
        t = torch.full((n, c), -1, dtype=torch.long)
        t[:, :4] = torch.randint(0, c, (n, 4))
    else:
        t = torch.randint(0, c, (n,))
    x_ref, x_our = both(x, hw, mojo)
    t_ref, t_our = both(t, hw, mojo)
    return x_ref, x_our, t_ref, t_our


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", MARGIN_SHAPES)
@pytest.mark.bench_op("multi_margin_loss")
def test_multi_margin_loss(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our, t_ref, t_our = _margin_case(
        shape_id, dtype_id, hw, mojo_device, False
    )
    bench.run(
        lambda: torch.ops.aten.multi_margin_loss(x_ref, t_ref, 1, 1.0, None, 1),
        lambda: torch.ops.aten.multi_margin_loss(x_our, t_our, 1, 1.0, None, 1),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", MARGIN_SHAPES)
@pytest.mark.bench_op("multi_margin_loss_backward")
def test_multi_margin_loss_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our, t_ref, t_our = _margin_case(
        shape_id, dtype_id, hw, mojo_device, False
    )
    dtype = DTYPES[dtype_id]
    g_ref, g_our = both(torch.tensor(1.0, dtype=dtype), hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten.multi_margin_loss_backward(
            g_ref, x_ref, t_ref, 1, 1.0, None, 1
        ),
        lambda: torch.ops.aten.multi_margin_loss_backward(
            g_our, x_our, t_our, 1, 1.0, None, 1
        ),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", MARGIN_SHAPES)
@pytest.mark.bench_op("multilabel_margin_loss_forward")
def test_multilabel_margin_loss(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our, t_ref, t_our = _margin_case(shape_id, dtype_id, hw, mojo_device, True)
    bench.run(
        lambda: torch.ops.aten.multilabel_margin_loss_forward(x_ref, t_ref, 1),
        lambda: torch.ops.aten.multilabel_margin_loss_forward(x_our, t_our, 1),
        flops=float(x_ref.numel()),
    )


@pytest.mark.parametrize("dtype_id", ("bf16", "f32"))
@pytest.mark.parametrize("shape_id", MARGIN_SHAPES)
@pytest.mark.bench_op("multilabel_margin_loss_backward")
def test_multilabel_margin_loss_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    x_ref, x_our, t_ref, t_our = _margin_case(shape_id, dtype_id, hw, mojo_device, True)
    dtype = DTYPES[dtype_id]
    g_ref, g_our = both(torch.tensor(1.0, dtype=dtype), hw, mojo_device)
    _, it_ref = torch.ops.aten.multilabel_margin_loss_forward(x_ref, t_ref, 1)
    _, it_our = torch.ops.aten.multilabel_margin_loss_forward(x_our, t_our, 1)
    bench.run(
        lambda: torch.ops.aten.multilabel_margin_loss_backward(
            g_ref, x_ref, t_ref, 1, it_ref
        ),
        lambda: torch.ops.aten.multilabel_margin_loss_backward(
            g_our, x_our, t_our, 1, it_our
        ),
        flops=float(x_ref.numel()),
    )


# (T, B, C, S): time steps, batch, vocabulary, target length
CTC_SHAPES: dict[str, tuple[int, int, int, int]] = {
    "T200xB16xC32xS50": (200, 16, 32, 50),
    "T97xB5xC29xS23": (97, 5, 29, 23),
}


def _ctc_case(
    shape_id: str, hw: Hardware, mojo: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int], list[int]]:
    t, b, c, s = CTC_SHAPES[shape_id]
    lp = torch.randn(t, b, c).log_softmax(2)
    tg = torch.randint(1, c, (b, s))
    lp_ref, lp_our = both(lp, hw, mojo)
    tg_ref, tg_our = both(tg, hw, mojo)
    return [lp_ref, tg_ref], [lp_our, tg_our], [t] * b, [s] * b


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", CTC_SHAPES)
@pytest.mark.bench_op("_ctc_loss")
def test_ctc_loss(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    refs, ours, il, tl = _ctc_case(shape_id, hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten._ctc_loss(*refs, il, tl, 0, False),
        lambda: torch.ops.aten._ctc_loss(*ours, il, tl, 0, False),
        flops=float(refs[0].numel()),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", CTC_SHAPES)
@pytest.mark.bench_op("_ctc_loss_backward")
def test_ctc_loss_backward(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    refs, ours, il, tl = _ctc_case(shape_id, hw, mojo_device)
    nll_ref, la_ref = torch.ops.aten._ctc_loss(*refs, il, tl, 0, False)
    nll_our, la_our = torch.ops.aten._ctc_loss(*ours, il, tl, 0, False)
    g_ref, g_our = both(torch.ones(len(il)), hw, mojo_device)
    bench.run(
        lambda: torch.ops.aten._ctc_loss_backward(
            g_ref, *refs, il, tl, nll_ref, la_ref, 0, False
        ),
        lambda: torch.ops.aten._ctc_loss_backward(
            g_our, *ours, il, tl, nll_our, la_our, 0, False
        ),
        flops=float(refs[0].numel()),
    )
