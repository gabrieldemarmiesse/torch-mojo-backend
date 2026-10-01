"""Native-backend loss ops (tmb/ops/loss.mojo): NLL (1-D, 2-D, spatial,
class weights, byte targets), the multi-class and multi-label margin losses
and CTC, every overload, through public torch APIs on the mojo device.

float32 / float64 compare against CPU torch. torch's CUDA kernels round some
intermediates to the half dtype where CPU's do not, and this backend follows
CUDA, so the half cases compare against a float64 reference that applies
CUDA's rounding points (documented on each helper) instead of CPU torch.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.native.conftest import ran, skip_if_metal
from torch_mojo_backend import register_mojo_devices

aten = torch.ops.aten
HALF = [torch.float16, torch.bfloat16]


@pytest.fixture(autouse=True)
def _registered():
    register_mojo_devices()


def _dtypes(device: str, *, with_half: bool = True) -> list[torch.dtype]:
    out = [torch.float32]
    if with_half:
        out += HALF
    return out


def _f64_or_skip(device: str):
    skip_if_metal(device, "Apple GPUs have no float64")


def _close(got: torch.Tensor, want: torch.Tensor, dtype: torch.dtype):
    """One rounding step of `dtype` (plus a float32 reassociation margin)."""
    if dtype in HALF:
        tol = 2**-7 if dtype == torch.bfloat16 else 2**-10
        torch.testing.assert_close(
            got.cpu().double(), want.double(), atol=tol, rtol=tol, equal_nan=True
        )
    elif dtype == torch.float32:
        torch.testing.assert_close(
            got.cpu(), want.to(got.dtype), atol=2e-6, rtol=2e-6, equal_nan=True
        )
    else:
        torch.testing.assert_close(got.cpu(), want.to(got.dtype), equal_nan=True)


# ---------------------------------------------------------------------------
# NLL loss
# ---------------------------------------------------------------------------


def _nll_case(shape, dtype, weighted, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g).log_softmax(1 if len(shape) > 1 else 0)
    classes = shape[1] if len(shape) > 1 else shape[0]
    tshape = (shape[0], *shape[2:]) if len(shape) > 1 else ()
    t = torch.randint(0, classes, tshape, generator=g)
    w = None
    if weighted == "random":
        w = torch.rand(classes, generator=g) * 2
    elif weighted == "zero":
        w = torch.zeros(classes)
    x = x.to(dtype)
    return x, t, None if w is None else w.to(dtype)


def _nll_cuda_reference(x, t, w, reduction, ignore):
    """Loss.cu / NLLLoss2d.cu's values: the per-element loss `-w[t] * x` is
    formed (and rounded) in the input dtype, sums accumulate in float, the
    mean divides the float sums before the one rounding to the dtype; the
    spatial reduction adds per-block partials rounded to the dtype (one
    block per sample here: every map in these tests is below 128 * 128)."""
    dt = x.dtype
    classes = x.shape[1] if x.dim() > 1 else x.shape[0]
    wt = torch.ones(classes, dtype=dt) if w is None else w
    xr = x if x.dim() > 1 else x.unsqueeze(0)
    tr = t if x.dim() > 1 else t.reshape(1)
    valid = tr != ignore
    safe = tr.clamp(0, classes - 1)
    picked = xr.gather(1, safe.unsqueeze(1)).squeeze(1)
    wsel = wt[safe]
    loss = (-(wsel) * picked) * valid.to(dt)  # rounded to dt
    if reduction == 0 and x.dim() > 1:
        return loss, torch.zeros((), dtype=dt)
    if x.dim() == 1:
        if not bool(valid.all()):
            return torch.zeros((), dtype=dt), torch.zeros((), dtype=dt)
        w0 = wsel.reshape(())
        if reduction == 1:
            out = (
                torch.tensor(float("nan"), dtype=dt) if w0 == 0 else -picked.reshape(())
            )
        else:
            out = loss.reshape(())
        return out, w0
    lsum = (picked * wsel * valid.to(dt)).double()
    wsum = (wsel * valid.to(dt)).double()
    if x.dim() > 2:
        per_l = (-lsum.flatten(1).sum(1)).to(dt)
        per_w = wsum.flatten(1).sum(1).to(dt)
        out = torch.zeros((), dtype=dt)
        tw = torch.zeros((), dtype=dt)
        for i in range(per_l.numel()):
            out = out + per_l[i]
            tw = tw + per_w[i]
        if reduction == 1:
            out = out / tw
        return out, tw
    total = -lsum.sum()
    tw = wsum.sum()
    out = total / tw if reduction == 1 else total
    return out.float().to(dt), tw.float().to(dt)


@pytest.mark.parametrize("shape", [(5, 7), (7,), (3, 4, 5, 6), (300, 10)])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("weighted", [None, "random", "zero"])
@pytest.mark.parametrize("ignore", [-100, 2])
def test_nll_loss_float32_matches_cpu(mojo_device, shape, reduction, weighted, ignore):
    x, t, w = _nll_case(shape, torch.float32, weighted)
    xr = x.clone().requires_grad_()
    want = F.nll_loss(xr, t, weight=w, reduction=reduction, ignore_index=ignore)
    xm = x.to(mojo_device).requires_grad_()
    op = "aten::nll_loss2d_forward" if len(shape) == 4 else "aten::nll_loss_forward"
    with ran(op):
        got = F.nll_loss(
            xm,
            t.to(mojo_device),
            weight=None if w is None else w.to(mojo_device),
            reduction=reduction,
            ignore_index=ignore,
        )
    grad = torch.randn_like(want)
    want.backward(grad)
    got.backward(grad.to(mojo_device))
    _close(got.detach(), want.detach(), torch.float32)
    _close(xm.grad, xr.grad, torch.float32)


@pytest.mark.parametrize("dtype", HALF)
@pytest.mark.parametrize("shape", [(5, 7), (7,), (3, 4, 5, 6)])
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("weighted", [None, "random", "zero"])
def test_nll_loss_half_follows_cuda(mojo_device, dtype, shape, reduction, weighted):
    x, t, w = _nll_case(shape, dtype, weighted, seed=3)
    if t.numel() > 1:
        t.view(-1)[1] = 2  # one ignored element
    want_out, want_tw = _nll_cuda_reference(x, t, w, reduction, 2)
    fn = aten.nll_loss2d_forward if len(shape) == 4 else aten.nll_loss_forward
    out, tw = fn(
        x.to(mojo_device),
        t.to(mojo_device),
        None if w is None else w.to(mojo_device),
        reduction,
        2,
    )
    assert out.dtype == dtype and tw.dtype == dtype
    _close(out, want_out, dtype)
    _close(tw, want_tw, dtype)


@pytest.mark.parametrize("shape", [(6, 5), (5,), (2, 3, 4, 4)])
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_nll_loss_float64(mojo_device, shape, reduction):
    _f64_or_skip(mojo_device)
    x, t, w = _nll_case(shape, torch.float64, "random", seed=5)
    fn = aten.nll_loss2d_forward if len(shape) == 4 else aten.nll_loss_forward
    want = fn(x, t, w, reduction, -100)
    got = fn(x.to(mojo_device), t.to(mojo_device), w.to(mojo_device), reduction, -100)
    _close(got[0], want[0], torch.float64)
    _close(got[1], want[1], torch.float64)
    bwd = aten.nll_loss2d_backward if len(shape) == 4 else aten.nll_loss_backward
    grad = torch.randn(want[0].shape, dtype=torch.float64)
    want_gi = bwd(grad, x, t, w, reduction, -100, want[1])
    got_gi = bwd(
        grad.to(mojo_device),
        x.to(mojo_device),
        t.to(mojo_device),
        w.to(mojo_device),
        reduction,
        -100,
        got[1],
    )
    _close(got_gi, want_gi, torch.float64)


def test_nll_loss_byte_target(mojo_device):
    x = torch.randn(6, 5).log_softmax(1)
    t = torch.randint(0, 5, (6,), dtype=torch.uint8)
    w = torch.rand(5)
    for reduction in [0, 1, 2]:
        want = aten.nll_loss_forward(x, t, w, reduction, -100)
        got = aten.nll_loss_forward(
            x.to(mojo_device), t.to(mojo_device), w.to(mojo_device), reduction, -100
        )
        _close(got[0], want[0], torch.float32)
        _close(got[1], want[1], torch.float32)


def test_nll_loss_empty_batch(mojo_device):
    x = torch.randn(0, 3)
    t = torch.zeros(0, dtype=torch.long)
    for reduction in [0, 1, 2]:
        want = aten.nll_loss_forward(x, t, None, reduction, -100)
        got = aten.nll_loss_forward(
            x.to(mojo_device), t.to(mojo_device), None, reduction, -100
        )
        assert got[0].shape == want[0].shape
        torch.testing.assert_close(got[0].cpu(), want[0], equal_nan=True)
        torch.testing.assert_close(got[1].cpu(), want[1])
    x4 = torch.randn(0, 3, 2, 2)
    t3 = torch.zeros(0, 2, 2, dtype=torch.long)
    for reduction in [0, 1, 2]:
        want = aten.nll_loss2d_forward(x4, t3, None, reduction, -100)
        got = aten.nll_loss2d_forward(
            x4.to(mojo_device), t3.to(mojo_device), None, reduction, -100
        )
        assert got[0].shape == want[0].shape
        torch.testing.assert_close(got[0].cpu(), want[0], equal_nan=True)


def test_nll_loss_out_overloads(mojo_device):
    x, t, w = _nll_case((5, 4), torch.float32, "random", seed=9)
    want_out, want_tw = aten.nll_loss_forward(x, t, w, 0, -100)
    # Wrong-shaped outs are resized; a strided out keeps its storage.
    out = torch.empty(2, device=mojo_device)
    base = torch.zeros(10, device=mojo_device)
    tw = base[3:4].view(())
    with ran("aten::nll_loss_forward.output"):
        r = aten.nll_loss_forward.output(
            x.to(mojo_device),
            t.to(mojo_device),
            w.to(mojo_device),
            0,
            -100,
            output=out,
            total_weight=tw,
        )
    assert r[0] is out and r[1] is tw
    _close(out, want_out, torch.float32)
    assert float(base[3]) == float(want_tw)
    grad = torch.randn(5)
    want_gi = aten.nll_loss_backward(grad, x, t, w, 0, -100, want_tw)
    gi = torch.empty(0, device=mojo_device)
    with ran("aten::nll_loss_backward.grad_input"):
        aten.nll_loss_backward.grad_input(
            grad.to(mojo_device),
            x.to(mojo_device),
            t.to(mojo_device),
            w.to(mojo_device),
            0,
            -100,
            tw,
            grad_input=gi,
        )
    _close(gi, want_gi, torch.float32)
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
        aten.nll_loss_forward.output(
            x.to(mojo_device),
            t.to(mojo_device),
            None,
            1,
            -100,
            output=torch.empty((), dtype=torch.float64, device=mojo_device),
            total_weight=torch.empty((), device=mojo_device),
        )


def test_nll_loss2d_out_overloads(mojo_device):
    x, t, w = _nll_case((2, 3, 4, 5), torch.float32, "random", seed=11)
    for reduction in [0, 1, 2]:
        want_out, want_tw = aten.nll_loss2d_forward(x, t, w, reduction, 1)
        out = torch.empty(0, device=mojo_device)
        tw = torch.empty(0, device=mojo_device)
        aten.nll_loss2d_forward.output(
            x.to(mojo_device),
            t.to(mojo_device),
            w.to(mojo_device),
            reduction,
            1,
            output=out,
            total_weight=tw,
        )
        _close(out, want_out, torch.float32)
        _close(tw, want_tw, torch.float32)
        grad = torch.randn(want_out.shape)
        want_gi = aten.nll_loss2d_backward(grad, x, t, w, reduction, 1, want_tw)
        gi = torch.empty(0, device=mojo_device)
        aten.nll_loss2d_backward.grad_input(
            grad.to(mojo_device),
            x.to(mojo_device),
            t.to(mojo_device),
            w.to(mojo_device),
            reduction,
            1,
            tw,
            grad_input=gi,
        )
        _close(gi, want_gi, torch.float32)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (
            lambda d: (
                torch.randn(2, 3, 4, device=d),
                torch.zeros(2, dtype=torch.long, device=d),
            ),
            "input tensor should be 1D or 2D",
        ),
        (
            lambda d: (
                torch.randn(2, 3, device=d),
                torch.zeros(2, 2, dtype=torch.long, device=d),
            ),
            "0D or 1D target tensor expected",
        ),
        (
            lambda d: (
                torch.randn(2, 3, device=d),
                torch.zeros(3, dtype=torch.long, device=d),
            ),
            r"size mismatch \(got input: \[2, 3\], target: \[3\]\)",
        ),
        (
            lambda d: (
                torch.randn(2, 3, device=d),
                torch.zeros(2, dtype=torch.int32, device=d),
            ),
            "expected target dtype to be Long or Byte, but got Int",
        ),
    ],
)
def test_nll_loss_errors(mojo_device, args, match):
    x, t = args(mojo_device)
    with pytest.raises(RuntimeError, match=match):
        aten.nll_loss_forward(x, t, None, 1, -100)


def test_nll_loss_scalar_operands_raise(mojo_device):
    d = mojo_device
    with pytest.raises(RuntimeError, match="input tensor should be 1D or 2D"):
        aten.nll_loss_forward(
            torch.tensor(1.0, device=d), torch.tensor(0, device=d), None, 1, -100
        )
    with pytest.raises(
        IndexError, match="dimension specified as 0 but tensor has no dimensions"
    ):
        aten.nll_loss_forward(
            torch.randn(2, 3, device=d), torch.tensor(0, device=d), None, 1, -100
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("spatial", [False, True])
@pytest.mark.parametrize("reduction", [0, 1])
def test_nll_loss_target_out_of_range_raises(mojo_device, dtype, spatial, reduction):
    """CUDA asserts on a target outside [0, classes); the generic kernels
    raise (the weighted case leaves the f32 fast path)."""
    d = mojo_device
    shape = (2, 3, 2, 2) if spatial else (4, 3)
    x = torch.randn(shape, dtype=dtype, device=d)
    t = torch.zeros((2, 2, 2) if spatial else (4,), dtype=torch.long)
    t.view(-1)[1] = 7
    w = torch.ones(3, dtype=dtype, device=d)
    fn = aten.nll_loss2d_forward if spatial else aten.nll_loss_forward
    with pytest.raises(IndexError, match="Target 7 is out of bounds"):
        fn(x, t.to(d), w, reduction, -100)
    tw = torch.tensor(1.0, dtype=dtype, device=d)
    g = torch.ones(t.shape if reduction == 0 else (), dtype=dtype, device=d)
    bwd = aten.nll_loss2d_backward if spatial else aten.nll_loss_backward
    with pytest.raises(IndexError, match="Target 7 is out of bounds"):
        bwd(g, x, t.to(d), w, reduction, -100, tw)


def test_nll_loss_out_written_in_place(mojo_device):
    """A fitting contiguous `out=` is the kernel's destination: its storage
    is kept, no temporary."""
    d = mojo_device
    x, t, _ = _nll_case((6, 5), torch.float32, None, seed=2)
    out = torch.empty(6, device=d)
    tw = torch.empty((), device=d)
    ptr = out.data_ptr()
    aten.nll_loss_forward.output(
        x.to(d), t.to(d), None, 0, -100, output=out, total_weight=tw
    )
    assert out.data_ptr() == ptr
    want = aten.nll_loss_forward(x, t, None, 0, -100)[0]
    _close(out, want, torch.float32)
    gi = torch.full((6, 5), 7.0, device=d)
    gptr = gi.data_ptr()
    aten.nll_loss_backward.grad_input(
        torch.ones(6, device=d), x.to(d), t.to(d), None, 0, -100, tw, grad_input=gi
    )
    assert gi.data_ptr() == gptr
    _close(
        gi,
        aten.nll_loss_backward(torch.ones(6), x, t, None, 0, -100, torch.tensor(0.0)),
        torch.float32,
    )


def test_nll_loss_weight_errors(mojo_device):
    x = torch.randn(2, 3, device=mojo_device)
    t = torch.zeros(2, dtype=torch.long, device=mojo_device)
    with pytest.raises(
        RuntimeError, match="weight tensor should be defined either for all 3 classes"
    ):
        aten.nll_loss_forward(x, t, torch.ones(4, device=mojo_device), 1, -100)
    with pytest.raises(RuntimeError, match="expected scalar type Float but found Half"):
        aten.nll_loss_forward(
            x, t, torch.ones(3, dtype=torch.half, device=mojo_device), 1, -100
        )
    x4 = torch.randn(2, 3, 4, 4, device=mojo_device)
    with pytest.raises(RuntimeError, match="only batches of spatial targets supported"):
        aten.nll_loss2d_forward(x4, t, None, 1, -100)
    with pytest.raises(
        RuntimeError, match="input and target batch or spatial sizes don't match"
    ):
        aten.nll_loss2d_forward(
            x4,
            torch.zeros(2, 4, 5, dtype=torch.long, device=mojo_device),
            None,
            1,
            -100,
        )


def test_cross_entropy_spatial_and_label_smoothing(mojo_device):
    x = torch.randn(2, 5, 3, 3)
    t = torch.randint(0, 5, (2, 3, 3))
    xd, td = x.to(mojo_device), t.to(mojo_device)
    _close(F.cross_entropy(xd, td), F.cross_entropy(x, t), torch.float32)
    _close(
        F.cross_entropy(xd, td, label_smoothing=0.2),
        F.cross_entropy(x, t, label_smoothing=0.2),
        torch.float32,
    )
    _close(
        F.cross_entropy(xd, td, reduction="none", ignore_index=1),
        F.cross_entropy(x, t, reduction="none", ignore_index=1),
        torch.float32,
    )


# ---------------------------------------------------------------------------
# Multi-class margin loss
# ---------------------------------------------------------------------------


def _mm_cases():
    for shape, tshape in [
        ((), ()),
        ((5,), ()),
        ((5,), (1,)),
        ((4, 7), (4,)),
        ((40, 300), (40,)),
    ]:
        for p in [1, 2]:
            for weighted in [False, True]:
                yield shape, tshape, p, weighted


@pytest.mark.parametrize(("shape", "tshape", "p", "weighted"), list(_mm_cases()))
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_multi_margin_loss_matches_cpu(
    mojo_device, shape, tshape, p, weighted, reduction
):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(shape, generator=g)
    classes = 1 if shape == () else shape[-1]
    t = torch.randint(0, classes, tshape, generator=g)
    w = torch.rand(classes, generator=g) * 4 - 2 if weighted else None
    for margin in [1.0, -0.5]:
        xr = x.clone().requires_grad_()
        want = F.multi_margin_loss(
            xr, t, p=p, margin=margin, weight=w, reduction=reduction
        )
        xm = x.to(mojo_device).requires_grad_()
        with ran("aten::multi_margin_loss"):
            got = F.multi_margin_loss(
                xm,
                t.to(mojo_device),
                p=p,
                margin=margin,
                weight=None if w is None else w.to(mojo_device),
                reduction=reduction,
            )
        assert got.shape == want.shape
        grad = torch.randn_like(want)
        want.backward(grad)
        with ran("aten::multi_margin_loss_backward"):
            got.backward(grad.to(mojo_device))
        torch.testing.assert_close(
            got.detach().cpu(), want.detach(), atol=1e-5, rtol=1e-5
        )
        assert xm.grad is not None
        torch.testing.assert_close(xm.grad.cpu(), xr.grad, atol=1e-5, rtol=1e-5)


def _multi_margin_cuda_reference(x, t, p, margin, w, reduction):
    """MultiMarginLoss.cu: `margin - x[t] + x[i]` and its square / weight in
    the input dtype, the per-sample sum in float, one rounding per sample,
    then (for a reduction) the float sum of those rounded values."""
    dt = x.dtype
    m = torch.tensor(margin, dtype=torch.float32).to(dt)
    rows = x.reshape(-1, x.shape[-1] if x.dim() else 1)
    tt = t.reshape(-1)
    dim = rows.shape[1]
    outs = []
    for k in range(rows.shape[0]):
        xt = rows[k, tt[k]]
        z = (m - xt) + rows[k]
        h = z if p == 1 else z * z
        if w is not None:
            h = h * w[tt[k]]
        keep = (z > 0) & (torch.arange(dim) != tt[k])
        s = (h.float() * keep).sum()
        denom = rows.shape[0] * dim if reduction == "mean" else dim
        outs.append((s / denom).to(dt))
    per = torch.stack(outs)
    if x.dim() == 2 and reduction != "none":
        return per.float().sum().to(dt)
    return (
        per.reshape(())
        if x.dim() < 2 and (t.dim() == 0 or reduction != "none")
        else per
    )


@pytest.mark.parametrize("dtype", HALF)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_multi_margin_loss_half_follows_cuda(mojo_device, dtype, reduction):
    g = torch.Generator().manual_seed(4)
    x = torch.randn(6, 9, generator=g).to(dtype)
    t = torch.randint(0, 9, (6,), generator=g)
    w = (torch.rand(9, generator=g) + 0.5).to(dtype)
    for p in [1, 2]:
        want = _multi_margin_cuda_reference(x, t, p, 0.75, w, reduction)
        got = F.multi_margin_loss(
            x.to(mojo_device),
            t.to(mojo_device),
            p=p,
            margin=0.75,
            weight=w.to(mojo_device),
            reduction=reduction,
        )
        _close(got, want, dtype)


def test_multi_margin_loss_float64(mojo_device):
    _f64_or_skip(mojo_device)
    x = torch.randn(5, 6, dtype=torch.float64)
    t = torch.randint(0, 6, (5,))
    for reduction in ["none", "mean", "sum"]:
        want = F.multi_margin_loss(x, t, p=2, reduction=reduction)
        got = F.multi_margin_loss(
            x.to(mojo_device), t.to(mojo_device), p=2, reduction=reduction
        )
        _close(got, want, torch.float64)


def test_multi_margin_loss_out_overloads(mojo_device):
    x = torch.randn(4, 5)
    t = torch.randint(0, 5, (4,))
    want = aten.multi_margin_loss(x, t, 1, 1, None, 0)
    out = torch.empty(1, 1, device=mojo_device)
    with ran("aten::multi_margin_loss.out"):
        r = aten.multi_margin_loss.out(
            x.to(mojo_device), t.to(mojo_device), 1, 1, None, 0, out=out
        )
    assert r is out
    _close(out, want, torch.float32)
    grad = torch.randn(4)
    want_gi = aten.multi_margin_loss_backward(grad, x, t, 1, 1, None, 0)
    gi = torch.empty(0, device=mojo_device)
    with ran("aten::multi_margin_loss_backward.grad_input"):
        aten.multi_margin_loss_backward.grad_input(
            grad.to(mojo_device),
            x.to(mojo_device),
            t.to(mojo_device),
            1,
            1,
            None,
            0,
            grad_input=gi,
        )
    _close(gi, want_gi, torch.float32)


def test_multi_margin_loss_errors(mojo_device):
    d = mojo_device
    x = torch.randn(5, 4, device=d)
    t = torch.zeros(5, dtype=torch.long, device=d)
    with pytest.raises(
        RuntimeError,
        match=r"Expected non-empty vector or matrix with optional 0-dim batch size, but got: \[5, 0\]",
    ):
        aten.multi_margin_loss(torch.randn(5, 0, device=d), t, 1, 1, None, 1)
    with pytest.raises(
        RuntimeError, match=r"inconsistent target size, expected 5 but got \[5, 4\]"
    ):
        aten.multi_margin_loss(
            x, torch.zeros(5, 4, dtype=torch.long, device=d), 1, 1, None, 1
        )
    with pytest.raises(RuntimeError, match="expected scalar type Long but found Float"):
        aten.multi_margin_loss(x, torch.zeros(5, device=d), 1, 1, None, 1)
    with pytest.raises(
        RuntimeError, match=r"inconsistent weight size, expected 4 but got \[5\]"
    ):
        aten.multi_margin_loss(x, t, 1, 1, torch.ones(5, device=d), 1)
    with pytest.raises(RuntimeError, match="Invalid p, expected 1 or 2 but got 3"):
        aten.multi_margin_loss(x, t, 3, 1, None, 1)


# ---------------------------------------------------------------------------
# Multi-label margin loss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(), (5,), (4, 7), (30, 200)])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_multilabel_margin_loss_matches_cpu(mojo_device, shape, reduction):
    g = torch.Generator().manual_seed(2)
    x = torch.randn(shape, generator=g)
    classes = 1 if shape == () else shape[-1]
    t = torch.randint(-1, classes, shape, generator=g)
    xr = x.clone().requires_grad_()
    want = F.multilabel_margin_loss(xr, t, reduction=reduction)
    xm = x.to(mojo_device).requires_grad_()
    with ran("aten::multilabel_margin_loss_forward"):
        got = F.multilabel_margin_loss(xm, t.to(mojo_device), reduction=reduction)
    grad = torch.randn_like(want)
    want.backward(grad)
    with ran("aten::multilabel_margin_loss_backward"):
        got.backward(grad.to(mojo_device))
    torch.testing.assert_close(got.detach().cpu(), want.detach(), atol=1e-5, rtol=1e-5)
    assert xm.grad is not None
    torch.testing.assert_close(xm.grad.cpu(), xr.grad, atol=1e-5, rtol=1e-5)
    _, want_is_target = aten.multilabel_margin_loss_forward(x, t, 1)
    _, got_is_target = aten.multilabel_margin_loss_forward(
        x.to(mojo_device), t.to(mojo_device), 1
    )
    torch.testing.assert_close(got_is_target.cpu(), want_is_target)


def test_multilabel_margin_loss_repeated_and_terminated_labels(mojo_device):
    x = torch.randn(7)
    t = torch.tensor([2, 0, 6, -1, 4, -1, 6])
    want = aten.multilabel_margin_loss_forward(x, t, 1)
    got = aten.multilabel_margin_loss_forward(x.to(mojo_device), t.to(mojo_device), 1)
    torch.testing.assert_close(got[0].cpu(), want[0])
    torch.testing.assert_close(got[1].cpu(), want[1])


def test_multilabel_margin_loss_half_and_double(mojo_device):
    g = torch.Generator().manual_seed(6)
    x = torch.randn(4, 6, generator=g)
    t = torch.randint(-1, 6, (4, 6), generator=g)
    for dtype in HALF:
        # MultiLabelMarginCriterion.cu: `1 - x[t] + x[d]` in the dtype, the
        # per-sample float sum / dim (/ nframe), rounded once.
        xh = x.to(dtype)
        got = aten.multilabel_margin_loss_forward(
            xh.to(mojo_device), t.to(mojo_device), 0
        )[0]
        want = []
        for k in range(4):
            tgt = []
            for v in t[k].tolist():
                if v < 0:
                    break
                tgt.append(v)
            is_t = torch.zeros(6, dtype=torch.bool)
            is_t[tgt] = True
            s = 0.0
            for j in tgt:
                z = (torch.tensor(1.0, dtype=dtype) - xh[k, j]) + xh[k]
                s += float((z.float() * ((z > 0) & ~is_t)).sum())
            want.append(torch.tensor(s / 6, dtype=torch.float32).to(dtype))
        _close(got, torch.stack(want), dtype)
    _f64_or_skip(mojo_device)
    xd = x.double()
    want = aten.multilabel_margin_loss_forward(xd, t, 2)
    got = aten.multilabel_margin_loss_forward(xd.to(mojo_device), t.to(mojo_device), 2)
    _close(got[0], want[0], torch.float64)


def test_multilabel_margin_loss_out_overloads(mojo_device):
    x = torch.randn(3, 4)
    t = torch.tensor([[1, 0, -1, 2], [3, -1, 0, 0], [0, 1, 2, 3]])
    want_out, want_it = aten.multilabel_margin_loss_forward(x, t, 0)
    out = torch.empty(0, device=mojo_device)
    is_target = torch.empty(0, device=mojo_device)
    with ran("aten::multilabel_margin_loss_forward.output"):
        aten.multilabel_margin_loss_forward.output(
            x.to(mojo_device), t.to(mojo_device), 0, output=out, is_target=is_target
        )
    _close(out, want_out, torch.float32)
    _close(is_target, want_it, torch.float32)
    grad = torch.randn(3)
    want_gi = aten.multilabel_margin_loss_backward(grad, x, t, 0, want_it)
    gi = torch.empty(0, device=mojo_device)
    with ran("aten::multilabel_margin_loss_backward.grad_input"):
        aten.multilabel_margin_loss_backward.grad_input(
            grad.to(mojo_device),
            x.to(mojo_device),
            t.to(mojo_device),
            0,
            is_target,
            grad_input=gi,
        )
    _close(gi, want_gi, torch.float32)


def test_multilabel_margin_loss_errors(mojo_device):
    d = mojo_device
    with pytest.raises(
        RuntimeError,
        match=r"inconsistent target size: \[4\] for input of size: \[5, 4\]",
    ):
        aten.multilabel_margin_loss_forward(
            torch.randn(5, 4, device=d), torch.zeros(4, dtype=torch.long, device=d), 1
        )
    with pytest.raises(RuntimeError, match=r"but got: \[0\]"):
        aten.multilabel_margin_loss_forward(
            torch.randn(0, device=d), torch.zeros(0, dtype=torch.long, device=d), 1
        )


# ---------------------------------------------------------------------------
# CTC loss
# ---------------------------------------------------------------------------


def _ctc_case(T, B, C, S, tdtype, concat, seed=0):
    g = torch.Generator().manual_seed(seed)
    lp = torch.randn(T, B, C, generator=g).log_softmax(2)
    il = [T - (i % 2) for i in range(B)]
    tl = [max(0, min(S, (2 * i + 1) % (S + 1))) for i in range(B)]
    if B > 2:
        tl[2] = 0
    tg = torch.randint(1, C, (B, S), generator=g, dtype=tdtype)
    if concat:
        tg = torch.cat([tg[i, : tl[i]] for i in range(B)])
    return lp, tg, il, tl


@pytest.mark.parametrize("shape", [(6, 3, 5, 3), (20, 4, 7, 6), (50, 2, 30, 20)])
@pytest.mark.parametrize("tdtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("concat", [False, True])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_ctc_loss_matches_cpu(mojo_device, shape, tdtype, concat, reduction):
    steps, batch, labels, longest = shape
    lp, tg, il_list, tl_list = _ctc_case(steps, batch, labels, longest, tdtype, concat)
    il, tl = torch.tensor(il_list), torch.tensor(tl_list)
    for zero_infinity in [False, True]:
        lr = lp.clone().requires_grad_()
        want = F.ctc_loss(
            lr, tg, il, tl, reduction=reduction, zero_infinity=zero_infinity
        )
        lm = lp.to(mojo_device).requires_grad_()
        with ran("aten::_ctc_loss"):
            got = F.ctc_loss(
                lm,
                tg.to(mojo_device),
                il,
                tl,
                reduction=reduction,
                zero_infinity=zero_infinity,
            )
        grad = torch.rand_like(want) + 0.5
        want.backward(grad)
        with ran("aten::_ctc_loss_backward"):
            got.backward(grad.to(mojo_device))
        assert lm.grad is not None and lr.grad is not None
        torch.testing.assert_close(
            got.detach().cpu(), want.detach(), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(lm.grad.cpu(), lr.grad, atol=1e-4, rtol=1e-4)


def test_ctc_loss_float64_and_infinite(mojo_device):
    _f64_or_skip(mojo_device)
    lp, tg, il, tl = _ctc_case(12, 3, 6, 4, torch.int64, False, seed=4)
    lp = lp.double()
    want = torch.ops.aten._ctc_loss(lp, tg, il, tl, 0, False)
    got = torch.ops.aten._ctc_loss(
        lp.to(mojo_device), tg.to(mojo_device), il, tl, 0, False
    )
    torch.testing.assert_close(got[0].cpu(), want[0])
    # An impossible alignment (four repeated labels in three steps): inf,
    # and zero_infinity zeroes its gradient.
    lp3 = torch.randn(3, 1, 4, dtype=torch.float64).log_softmax(2)
    tg3 = torch.tensor([[1, 1, 1, 1]])
    for zi in [False, True]:
        lr = lp3.clone().requires_grad_()
        il3, tl3 = torch.tensor([3]), torch.tensor([4])
        want = F.ctc_loss(lr, tg3, il3, tl3, reduction="sum", zero_infinity=zi)
        lm = lp3.to(mojo_device).requires_grad_()
        got = F.ctc_loss(
            lm, tg3.to(mojo_device), il3, tl3, reduction="sum", zero_infinity=zi
        )
        want.backward()
        got.backward()
        assert lm.grad is not None and lr.grad is not None
        torch.testing.assert_close(got.detach().cpu(), want.detach())
        if zi:
            torch.testing.assert_close(lm.grad.cpu(), lr.grad)
        else:
            # LossCTC.cu's collect kernel: exp(-inf + inf - lp) is NaN for
            # every label (CPU's kernel differs here).
            assert torch.isnan(lm.grad.cpu()).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ctc_loss_large_route(mojo_device, dtype):
    """LossCTC.cu takes its large-problem gradient for 2T + 2.4B + C/5 >
    450: an impossible alignment leaves the absent labels finite there
    (the small route makes them NaN)."""
    if dtype == torch.float64:
        _f64_or_skip(mojo_device)
    g = torch.Generator().manual_seed(1)
    lp = torch.randn(230, 1, 4, generator=g, dtype=dtype).log_softmax(2)
    tg = torch.tensor([[1, 1, 1]])
    lm = lp.to(mojo_device).requires_grad_()
    il, tl = torch.tensor([2]), torch.tensor([3])
    F.ctc_loss(lm, tg.to(mojo_device), il, tl, reduction="sum").backward()
    grad = lm.grad
    assert grad is not None
    grad = grad.cpu()
    assert torch.isfinite(grad[:2, 0, 2:]).all()
    assert (grad[2:] == 0).all()
    # A feasible large problem matches CPU.
    lp2 = torch.randn(240, 3, 6, generator=g, dtype=dtype).log_softmax(2)
    tg2 = torch.randint(1, 6, (3, 20), generator=g)
    il2, tl2 = torch.tensor([240, 200, 150]), torch.tensor([20, 15, 0])
    lr2 = lp2.clone().requires_grad_()
    lm2 = lp2.to(mojo_device).requires_grad_()
    F.ctc_loss(lr2, tg2, il2, tl2, reduction="sum").backward()
    F.ctc_loss(lm2, tg2.to(mojo_device), il2, tl2, reduction="sum").backward()
    assert lm2.grad is not None and lr2.grad is not None
    torch.testing.assert_close(lm2.grad.cpu(), lr2.grad, atol=1e-4, rtol=1e-4)


def test_ctc_loss_tensor_overloads(mojo_device):
    lp, tg, il, tl = _ctc_case(10, 3, 5, 4, torch.int64, True, seed=8)
    d = mojo_device
    ilt, tlt = torch.tensor(il), torch.tensor(tl)
    want = torch.ops.aten._ctc_loss(lp, tg, il, tl, 0, False)
    with ran("aten::_ctc_loss.Tensor"):
        got = torch.ops.aten._ctc_loss.Tensor(
            lp.to(d), tg.to(d), ilt.to(d), tlt.to(d).int(), 0, False
        )
    torch.testing.assert_close(got[0].cpu(), want[0], atol=1e-5, rtol=1e-5)
    g = torch.rand(3) + 0.5
    want_g = torch.ops.aten._ctc_loss_backward(
        g, lp, tg, il, tl, want[0], want[1], 0, False
    )
    with ran("aten::_ctc_loss_backward.Tensor"):
        got_g = torch.ops.aten._ctc_loss_backward.Tensor(
            g.to(d), lp.to(d), tg.to(d), ilt.to(d), tlt.to(d), got[0], got[1], 0, False
        )
    torch.testing.assert_close(got_g.cpu(), want_g, atol=1e-5, rtol=1e-5)
    with pytest.raises(RuntimeError, match="input_lengths is on cpu"):
        torch.ops.aten._ctc_loss.Tensor(lp.to(d), tg.to(d), ilt, tlt.to(d), 0, False)
    with pytest.raises(RuntimeError, match="input_lengths must be integral"):
        torch.ops.aten._ctc_loss.Tensor(
            lp.to(d), tg.to(d), ilt.float().to(d), tlt.to(d), 0, False
        )


def test_ctc_loss_errors(mojo_device):
    d = mojo_device
    lp = torch.randn(5, 2, 4, device=d).log_softmax(2)
    tg = torch.zeros(2, 3, dtype=torch.long, device=d)
    cases = [
        ((lp, tg.float(), [5, 5], [2, 2], 0), "to have scalar type Int; but got"),
        (
            (lp[0], tg, [5, 5], [2, 2], 0),
            "Expected 3-dimensional tensor, but got 2-dimensional tensor for argument #1 'log_probs'",
        ),
        (
            (lp, tg.view(2, 3, 1), [5, 5], [2, 2], 0),
            "Expected 1 to 2 dimensions, but got 3-dimensional tensor",
        ),
        (
            (lp, torch.zeros(5, dtype=torch.long, device=d), [5, 5], [2, 2], 0),
            "Expected tensor to have size 4 at dimension 0, but got size 5",
        ),
        (
            (lp, tg[:, :1], [5, 5], [2, 2], 0),
            "Expected tensor to have size at least 2 at dimension 1, but got size 1",
        ),
        (
            (lp, tg, [5, 6], [2, 2], 0),
            "Expected input_lengths to have value at most 5, but got value 6",
        ),
        (
            (lp, tg, [5, 5], [2, -1], 0),
            "Expected target_lengths to have value at least 0",
        ),
        ((lp, tg, [5, 5], [2, 2], 4), "blank must be in label range"),
        ((lp, tg, [5], [2, 2], 0), "input_lengths must be of size batch_size"),
    ]
    for args, match in cases:
        with pytest.raises(RuntimeError, match=match):
            torch.ops.aten._ctc_loss(*args, False)
    with pytest.raises(NotImplementedError, match="not implemented for 'Half'"):
        torch.ops.aten._ctc_loss(lp.half(), tg, [5, 5], [2, 2], 0, False)
