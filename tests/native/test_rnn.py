"""Fused recurrent cells on the native mojo device (tmb/ops/rnn.mojo):
`_thnn_fused_lstm_cell` / `_thnn_fused_gru_cell` and their backwards, which
ATen's LSTM/GRU cells call on a PrivateUse1 device after the two gate GEMMs.

The reference is CPU torch: the ops themselves are CUDA/XPU-only, so the
expected values come from the documented gate math (RNN.cu) in float64, and
whole modules (nn.LSTM / nn.GRU / the cells) are compared against the same
module on the CPU, training step included.
"""

import copy

import pytest
import torch

from tests.native.conftest import ran

aten = torch.ops.aten

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _tol(dtype: torch.dtype) -> float:
    if dtype == torch.bfloat16:
        return 2e-2
    if dtype == torch.float16:
        return 2e-3
    return 2e-5


def _close(got: torch.Tensor, want: torch.Tensor, tol: float, msg: str = ""):
    torch.testing.assert_close(got, want, atol=tol, rtol=tol, msg=msg or None)


def _grad(t: torch.Tensor) -> torch.Tensor:
    assert t.grad is not None
    return t.grad


def _lstm_ref(ig, hg, cx, b1, b2):
    """RNN.cu's lstm_cell_forward in float64: (hy, cy, workspace)."""
    g = ig.double() + hg.double()
    if b1 is not None:
        g = g + b1.double() + b2.double()
    i, f, c, o = g.chunk(4, 1)
    i, f, c, o = i.sigmoid(), f.sigmoid(), c.tanh(), o.sigmoid()
    cy = f * cx.double() + i * c
    hy = o * cy.tanh()
    return hy, cy, torch.cat([i, f, c, o], 1)


def _gru_ref(ig, hg, hx, b1, b2):
    """RNN.cu's gru_cell_forward in float64: (hy, workspace)."""
    ig, hg, hx = ig.double(), hg.double(), hx.double()
    if b1 is not None:
        ig = ig + b1.double()
        hg = hg + b2.double()
    ir, ii, in_ = ig.chunk(3, 1)
    hr, hi, hn = hg.chunk(3, 1)
    r = (ir + hr).sigmoid()
    z = (ii + hi).sigmoid()
    n = (in_ + r * hn).tanh()
    hy = n + z * (hx - n)
    return hy, torch.cat([r, z, n, hx, hn], 1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("shape", [(3, 16), (5, 37), (1, 1)])
def test_fused_lstm_cell(mojo_gpu, dtype, bias, shape):
    n, h = shape
    gen = torch.Generator().manual_seed(0)
    ig, hg = (torch.randn(n, 4 * h, generator=gen).to(dtype) for _ in range(2))
    cx = torch.randn(n, h, generator=gen).to(dtype)
    b1 = torch.randn(4 * h, generator=gen).to(dtype) if bias else None
    b2 = torch.randn(4 * h, generator=gen).to(dtype) if bias else None
    dev = [None if t is None else t.to(mojo_gpu) for t in (ig, hg, cx, b1, b2)]
    with ran("aten::_thnn_fused_lstm_cell"):
        hy, cy, ws = aten._thnn_fused_lstm_cell(*dev)
    for got, want in zip((hy, cy, ws), _lstm_ref(ig, hg, cx, b1, b2)):
        assert got.dtype == dtype
        _close(got.cpu().double(), want, _tol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("shape", [(3, 16), (5, 37)])
def test_fused_gru_cell(mojo_gpu, dtype, bias, shape):
    n, h = shape
    gen = torch.Generator().manual_seed(1)
    ig, hg = (torch.randn(n, 3 * h, generator=gen).to(dtype) for _ in range(2))
    hx = torch.randn(n, h, generator=gen).to(dtype)
    b1 = torch.randn(3 * h, generator=gen).to(dtype) if bias else None
    b2 = torch.randn(3 * h, generator=gen).to(dtype) if bias else None
    dev = [None if t is None else t.to(mojo_gpu) for t in (ig, hg, hx, b1, b2)]
    with ran("aten::_thnn_fused_gru_cell"):
        hy, ws = aten._thnn_fused_gru_cell(*dev)
    for got, want in zip((hy, ws), _gru_ref(ig, hg, hx, b1, b2)):
        assert got.dtype == dtype
        _close(got.cpu().double(), want, _tol(dtype))


def test_fused_lstm_cell_non_contiguous(mojo_gpu):
    gen = torch.Generator().manual_seed(2)
    ig = torch.randn(32, 6, generator=gen).to(mojo_gpu).t()
    hg = torch.randn(6, 64, generator=gen).to(mojo_gpu)[:, ::2]
    cx = torch.randn(8, 6, generator=gen).to(mojo_gpu).t()
    assert not (ig.is_contiguous() or hg.is_contiguous() or cx.is_contiguous())
    hy, cy, ws = aten._thnn_fused_lstm_cell(ig, hg, cx)
    want = _lstm_ref(ig.cpu(), hg.cpu(), cx.cpu(), None, None)
    for got, w in zip((hy, cy, ws), want):
        _close(got.cpu().double(), w, _tol(torch.float32))


def test_fused_lstm_cell_empty(mojo_gpu):
    z = torch.empty(0, 8, device=mojo_gpu)
    hy, cy, ws = aten._thnn_fused_lstm_cell(z, z, torch.empty(0, 2, device=mojo_gpu))
    assert hy.shape == (0, 2) and cy.shape == (0, 2) and ws.shape == (0, 8)


def test_fused_lstm_cell_errors_match_cuda(mojo_gpu):
    ig = torch.randn(3, 8, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="to have same size as tensor for"):
        aten._thnn_fused_lstm_cell(ig, torch.randn(3, 4, device=mojo_gpu), ig[:, :2])
    with pytest.raises(RuntimeError, match="Expected 2-dimensional tensor"):
        aten._thnn_fused_lstm_cell(ig[0], ig[0], ig[:, :2])
    with pytest.raises(RuntimeError, match="to have 6 elements"):
        aten._thnn_fused_lstm_cell(ig, ig, torch.randn(3, 3, device=mojo_gpu))
    with pytest.raises(RuntimeError, match="expected scalar type"):
        aten._thnn_fused_lstm_cell(ig, ig.half(), ig[:, :2])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("which", ["both", "hy", "cy"])
def test_fused_lstm_cell_backward(mojo_gpu, dtype, which):
    n, h = 4, 9
    gen = torch.Generator().manual_seed(3)
    ig, hg = (torch.randn(n, 4 * h, generator=gen).to(dtype) for _ in range(2))
    cx = torch.randn(n, h, generator=gen).to(dtype)
    _, cy, ws = _lstm_ref(ig, hg, cx, None, None)
    cy, ws = cy.to(dtype), ws.to(dtype)
    ghy = torch.randn(n, h, generator=gen).to(dtype) if which != "cy" else None
    gcy = torch.randn(n, h, generator=gen).to(dtype) if which != "hy" else None
    dev = [None if t is None else t.to(mojo_gpu) for t in (ghy, gcy, cx, cy, ws)]
    with ran("aten::_thnn_fused_lstm_cell_backward_impl"):
        gg, gcx, gb = aten._thnn_fused_lstm_cell_backward_impl(*dev, True)
    # RNN.cu's lstm_cell_backward in float64
    i, f, c, o = ws.double().chunk(4, 1)
    go = ghy.double() if ghy is not None else torch.zeros(n, h, dtype=torch.float64)
    goc = gcy.double() if gcy is not None else torch.zeros(n, h, dtype=torch.float64)
    t = cy.double().tanh()
    gc = go * o * (1 - t * t) + goc
    want = torch.cat(
        [
            gc * c * (1 - i) * i,
            gc * cx.double() * (1 - f) * f,
            gc * i * (1 - c * c),
            go * t * (1 - o) * o,
        ],
        1,
    )
    tol = _tol(dtype)
    _close(gg.cpu().double(), want, tol)
    _close(gcx.cpu().double(), gc * f, tol)
    _close(gb.cpu().double(), want.sum(0), tol)
    _, _, none = aten._thnn_fused_lstm_cell_backward_impl(*dev, False)
    assert none is None


@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_gru_cell_backward(mojo_gpu, dtype):
    n, h = 4, 9
    gen = torch.Generator().manual_seed(4)
    ig, hg = (torch.randn(n, 3 * h, generator=gen).to(dtype) for _ in range(2))
    hx = torch.randn(n, h, generator=gen).to(dtype)
    _, ws = _gru_ref(ig, hg, hx, None, None)
    ws = ws.to(dtype)
    ghy = torch.randn(n, h, generator=gen).to(dtype)
    with ran("aten::_thnn_fused_gru_cell_backward"):
        gi, gh, ghx, gib, ghb = aten._thnn_fused_gru_cell_backward(
            ghy.to(mojo_gpu), ws.to(mojo_gpu), True
        )
    r, z, nn, hxw, hn = ws.double().chunk(5, 1)
    go = ghy.double()
    gin = go * (1 - z) * (1 - nn * nn)
    grg = gin * hn * (1 - r) * r
    gig = go * (hxw - nn) * (1 - z) * z
    tol = _tol(dtype)
    _close(gi.cpu().double(), torch.cat([grg, gig, gin], 1), tol)
    _close(gh.cpu().double(), torch.cat([grg, gig, gin * r], 1), tol)
    _close(ghx.cpu().double(), go * z, tol)
    _close(gib.cpu().double(), torch.cat([grg, gig, gin], 1).sum(0), tol)
    _close(ghb.cpu().double(), torch.cat([grg, gig, gin * r], 1).sum(0), tol)
    out = aten._thnn_fused_gru_cell_backward(ghy.to(mojo_gpu), ws.to(mojo_gpu), False)
    assert out[3] is None and out[4] is None


@pytest.mark.parametrize(
    "make",
    [
        lambda b: torch.nn.LSTM(8, 16, num_layers=2, batch_first=True, bias=b),
        lambda b: torch.nn.GRU(8, 16, num_layers=2, bias=b),
        lambda b: torch.nn.LSTM(8, 16, bidirectional=True, bias=b),
    ],
    ids=["lstm", "gru", "bilstm"],
)
@pytest.mark.parametrize("bias", [True, False])
def test_rnn_module_training_step_matches_cpu(mojo_gpu, make, bias):
    torch.manual_seed(0)
    cpu = make(bias)
    dev = copy.deepcopy(cpu).to(mojo_gpu)
    x = torch.randn(3, 5, 8)
    xd = x.to(mojo_gpu).requires_grad_()
    x.requires_grad_()
    out = cpu(x)[0]
    g = torch.randn_like(out)
    out.backward(g)
    out_d = dev(xd)[0]
    out_d.backward(g.to(mojo_gpu))
    tol = 1e-4
    _close(out_d.cpu(), out, tol)
    _close(_grad(xd).cpu(), _grad(x), tol)
    for (name, p), pd in zip(cpu.named_parameters(), dev.parameters()):
        _close(_grad(pd).cpu(), _grad(p), tol, name)
    # one SGD step, then the next forward still agrees
    for opt_params in (cpu.parameters(), dev.parameters()):
        torch.optim.SGD(opt_params, lr=0.1).step()
    _close(dev(xd.detach())[0].cpu(), cpu(x.detach())[0], tol)


@pytest.mark.parametrize("cell", [torch.nn.LSTMCell, torch.nn.GRUCell])
def test_rnn_cell_modules_match_cpu(mojo_gpu, cell):
    torch.manual_seed(0)
    cpu = cell(6, 10)
    dev = copy.deepcopy(cpu).to(mojo_gpu)
    x = torch.randn(4, 6)
    out = cpu(x)
    out_d = dev(x.to(mojo_gpu))
    if isinstance(out, tuple):
        out, out_d = out[0] + out[1], out_d[0] + out_d[1]
    out.sum().backward()
    out_d.sum().backward()
    _close(out_d.cpu(), out, 1e-4)
    for p, pd in zip(cpu.parameters(), dev.parameters()):
        _close(_grad(pd).cpu(), _grad(p), 1e-4)
