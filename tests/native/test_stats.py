"""The stats group of the native mojo backend (tmb/ops/stats.mojo): mode,
histc, bincount, segment_reduce, renorm, the weight-norm interface, the fused
RMS-norm backward and `_compute_linear_combination`, through public torch
APIs, compared against CPU torch (or a float64 reference where CUDA and CPU
round differently)."""

import pytest
import torch

from tests.native.conftest import ran, skip_if_metal

# ---------------------------------------------------------------------------
# mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int64,
        torch.int32,
        torch.int8,
        torch.uint8,
        torch.bool,
    ],
)
@pytest.mark.parametrize(
    ("shape", "dim"), [((5, 9), 1), ((5, 9), 0), ((3, 4, 6), 1), ((7,), 0)]
)
@pytest.mark.parametrize("keepdim", [False, True])
def test_mode_matches_cpu(mojo_gpu, dtype, shape, dim, keepdim):
    x = torch.randint(0, 4, shape).to(dtype)
    with ran("aten::mode"):
        values, indices = torch.mode(x.to(mojo_gpu), dim, keepdim)
    ref_v, ref_i = torch.mode(x, dim, keepdim)
    torch.testing.assert_close(values.cpu(), ref_v)
    torch.testing.assert_close(indices.cpu(), ref_i)


def test_mode_ties_pick_smallest_value_and_largest_index(mojo_gpu):
    x = torch.tensor([[3.0, 1.0, 3.0, 1.0, 2.0], [5.0, 5.0, 4.0, 4.0, 4.0]])
    values, indices = torch.mode(x.to(mojo_gpu), 1)
    assert values.cpu().tolist() == [1.0, 4.0]
    assert indices.cpu().tolist() == [3, 4]


def test_mode_scalar_size_one_empty_and_out(mojo_gpu):
    v, i = torch.mode(torch.tensor(7.0, device=mojo_gpu))
    assert v.item() == 7.0 and i.item() == 0
    x = torch.randn(4, 1)
    v, i = torch.mode(x.to(mojo_gpu), 1)
    torch.testing.assert_close(v.cpu(), x[:, 0])
    assert i.cpu().eq(0).all()
    v, i = torch.mode(torch.empty(0, 3, device=mojo_gpu), 1)
    assert v.shape == (0,) and i.shape == (0,)
    with pytest.raises(IndexError, match="non-zero size"):
        torch.mode(torch.empty(3, 0, device=mojo_gpu), 1)
    out_v = torch.empty(0, device=mojo_gpu)
    out_i = torch.empty(0, dtype=torch.int64, device=mojo_gpu)
    y = torch.randint(0, 3, (4, 6)).float()
    torch.mode(y.to(mojo_gpu), 0, out=(out_v, out_i))
    ref_v, ref_i = torch.mode(y, 0)
    torch.testing.assert_close(out_v.cpu(), ref_v)
    torch.testing.assert_close(out_i.cpu(), ref_i)


def test_mode_long_rows(mojo_gpu):
    """CPU's index is whichever its unstable std::sort leaves last; CUDA's
    (and ours) is the largest index holding the mode."""
    x = torch.randint(0, 50, (3, 5000))
    v, i = torch.mode(x.to(mojo_gpu), 1)
    ref_v, _ = torch.mode(x, 1)
    torch.testing.assert_close(v.cpu(), ref_v)
    for r in range(3):
        hits = (x[r] == ref_v[r]).nonzero().flatten()
        assert i[r].item() == hits.max().item()


# ---------------------------------------------------------------------------
# histc / bincount
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int64, torch.int32]
)
@pytest.mark.parametrize(
    ("bins", "lo", "hi"), [(10, 0, 0), (7, -3, 4), (1, -1, 1), (100, 0, 0)]
)
def test_histc_matches_cpu(mojo_gpu, dtype, bins, lo, hi):
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    x = (torch.randn(2000) * 5).to(dtype)
    with ran("aten::histc"):
        got = torch.histc(x.to(mojo_gpu), bins, lo, hi)
    assert got.dtype == dtype
    if dtype.is_floating_point:
        torch.testing.assert_close(got.cpu(), torch.histc(x, bins, lo, hi))
    else:
        # CPU has no integer histc; CUDA's bins are int64 arithmetic.
        if lo == hi:
            lo, hi = int(x.min()), int(x.max())
        xi = x.long()
        inside = (xi >= lo) & (xi <= hi)
        b = ((xi - lo) * bins).div(hi - lo, rounding_mode="floor").clamp(0, bins - 1)
        ref = torch.bincount(b[inside], minlength=bins).to(dtype)
        torch.testing.assert_close(got.cpu(), ref)


def test_histc_edges_nan_and_errors(mojo_gpu):
    x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, float("nan"), -1.0, 5.0])
    torch.testing.assert_close(
        torch.histc(x.to(mojo_gpu), 4, 0, 4).cpu(), torch.histc(x, 4, 0, 4)
    )
    c = torch.full((5,), 2.0)
    torch.testing.assert_close(torch.histc(c.to(mojo_gpu), 3).cpu(), torch.histc(c, 3))
    with pytest.raises(RuntimeError, match="bins must be > 0"):
        torch.histc(x.to(mojo_gpu), 0)
    with pytest.raises(RuntimeError, match="max must be larger than min"):
        torch.histc(x.to(mojo_gpu), 3, 2, 1)
    with pytest.raises(RuntimeError, match="not finite"):
        torch.histc(x.to(mojo_gpu), 3, 0, float("inf"))
    # CUDA's dtype set: no half.
    with pytest.raises(RuntimeError, match="HalfTensor is not supported"):
        torch.histc(x.half().to(mojo_gpu), 3)
    out = torch.empty(0, device=mojo_gpu)
    torch.histc(x.to(mojo_gpu), 4, 0, 4, out=out)
    torch.testing.assert_close(out.cpu(), torch.histc(x, 4, 0, 4))
    assert torch.histc(torch.empty(0, device=mojo_gpu), 5).cpu().eq(0).all()


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.uint8, torch.int8])
def test_bincount_matches_cpu(mojo_gpu, dtype):
    x = torch.randint(0, 12, (300,)).to(dtype)
    with ran("aten::bincount"):
        got = torch.bincount(x.to(mojo_gpu))
    torch.testing.assert_close(got.cpu(), torch.bincount(x))
    torch.testing.assert_close(
        torch.bincount(x.to(mojo_gpu), minlength=20).cpu(),
        torch.bincount(x, minlength=20),
    )


def test_bincount_weights_empty_and_errors(mojo_gpu):
    x = torch.randint(0, 6, (50,))
    w = torch.rand(50)
    torch.testing.assert_close(
        torch.bincount(x.to(mojo_gpu), w.to(mojo_gpu)).cpu(), torch.bincount(x, w)
    )
    got = torch.bincount(
        torch.empty(0, dtype=torch.int64, device=mojo_gpu), minlength=4
    )
    assert got.dtype == torch.int64 and got.cpu().tolist() == [0, 0, 0, 0]
    with pytest.raises(RuntimeError, match="non-negative"):
        torch.bincount(torch.tensor([1, -1], device=mojo_gpu))
    with pytest.raises(RuntimeError, match="1-d"):
        torch.bincount(torch.zeros(2, 2, dtype=torch.int64, device=mojo_gpu))
    with pytest.raises(RuntimeError, match="same length"):
        torch.bincount(x.to(mojo_gpu), torch.rand(3, device=mojo_gpu))
    with pytest.raises(RuntimeError, match="not implemented for 'Float'"):
        torch.bincount(torch.rand(3, device=mojo_gpu))
    skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    wh = torch.rand(50).half()
    got = torch.bincount(x.to(mojo_gpu), wh.to(mojo_gpu))
    assert got.dtype == torch.float64
    torch.testing.assert_close(got.cpu(), torch.bincount(x, wh.double()))


# ---------------------------------------------------------------------------
# segment_reduce
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reduce", ["sum", "max", "min", "mean", "prod"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("axis", [0, 1])
def test_segment_reduce_lengths_and_offsets(mojo_gpu, reduce, dtype, axis):
    if axis == 0:
        data = torch.randn(6, 3).to(dtype)
        lengths = torch.tensor([2, 0, 3, 1])
        offsets = torch.tensor([0, 2, 2, 5, 6])
    else:
        data = torch.randn(2, 6, 3).to(dtype)
        lengths = torch.tensor([[2, 0, 3, 1], [1, 1, 4, 0]])
        offsets = torch.tensor([[0, 2, 2, 5, 6], [0, 1, 2, 6, 6]])
    with ran("aten::segment_reduce"):
        got = torch.segment_reduce(
            data.to(mojo_gpu), reduce, lengths=lengths.to(mojo_gpu), axis=axis
        )
    ref = torch.segment_reduce(data, reduce, lengths=lengths, axis=axis)
    atol, rtol = (None, None) if dtype == torch.float32 else (2e-2, 2e-2)
    torch.testing.assert_close(got.cpu(), ref, equal_nan=True, atol=atol, rtol=rtol)
    got = torch.segment_reduce(
        data.to(mojo_gpu), reduce, offsets=offsets.to(mojo_gpu), axis=axis, initial=0.5
    )
    ref = torch.segment_reduce(data, reduce, offsets=offsets, axis=axis, initial=0.5)
    torch.testing.assert_close(got.cpu(), ref, equal_nan=True, atol=atol, rtol=rtol)


@pytest.mark.parametrize("reduce", ["sum", "max", "min", "mean", "prod"])
def test_segment_reduce_backward(mojo_gpu, reduce):
    data = torch.randn(7, 2)
    data[3, 0] = 0.0
    lengths = torch.tensor([3, 0, 4])
    d_cpu = data.clone().requires_grad_()
    out = torch.segment_reduce(d_cpu, reduce, lengths=lengths, initial=1.0)
    grad = torch.randn_like(out)
    out.backward(grad)
    d_dev = data.to(mojo_gpu).requires_grad_()
    out_dev = torch.segment_reduce(
        d_dev, reduce, lengths=lengths.to(mojo_gpu), initial=1.0
    )
    with ran("aten::_segment_reduce_backward"):
        out_dev.backward(grad.to(mojo_gpu))
    assert d_dev.grad is not None
    torch.testing.assert_close(d_dev.grad.cpu(), d_cpu.grad)


def test_segment_reduce_errors(mojo_gpu):
    data = torch.randn(4, 2, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="Either lengths or offsets"):
        torch.segment_reduce(data, "sum")
    with pytest.raises(RuntimeError, match="reduce argument must be"):
        torch.segment_reduce(data, "median", lengths=torch.tensor([4], device=mojo_gpu))
    with pytest.raises(RuntimeError, match="negative"):
        torch.segment_reduce(
            data, "sum", lengths=torch.tensor([5, -1], device=mojo_gpu)
        )
    with pytest.raises(RuntimeError, match="sum to data.size"):
        torch.segment_reduce(data, "sum", lengths=torch.tensor([1, 1], device=mojo_gpu))


# ---------------------------------------------------------------------------
# renorm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("p", "dim", "maxnorm"), [(2, 1, 1.5), (1, 0, 3.0), (3.5, -1, 0.5), (2, 0, 100.0)]
)
def test_renorm_matches_cpu(mojo_gpu, dtype, p, dim, maxnorm):
    x = torch.randn(4, 5, 6).to(dtype)
    with ran("aten::renorm"):
        got = torch.renorm(x.to(mojo_gpu), p, dim, maxnorm)
    ref = torch.renorm(x.float(), p, dim, maxnorm).to(dtype)
    atol, rtol = (None, None) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(got.cpu(), ref, atol=atol, rtol=rtol)


def test_renorm_inplace_out_and_errors(mojo_gpu):
    x = torch.randn(3, 4)
    y = x.to(mojo_gpu)
    y.renorm_(2, 0, 1.0)
    torch.testing.assert_close(y.cpu(), torch.renorm(x, 2, 0, 1.0))
    out = torch.empty(0, device=mojo_gpu)
    torch.renorm(x.to(mojo_gpu), 2, 1, 0.7, out=out)
    torch.testing.assert_close(out.cpu(), torch.renorm(x, 2, 1, 0.7))
    with pytest.raises(RuntimeError, match="non-positive-norm"):
        torch.renorm(x.to(mojo_gpu), 0, 0, 1.0)
    with pytest.raises(RuntimeError, match="maxnorm to be >= 0"):
        torch.renorm(x.to(mojo_gpu), 2, 0, -1.0)
    with pytest.raises(RuntimeError, match="at least 2 dimensions"):
        torch.renorm(torch.randn(3, device=mojo_gpu), 2, 0, 1.0)


# ---------------------------------------------------------------------------
# weight norm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [0, 2])
def test_weight_norm_interface_matches_cpu(mojo_gpu, dtype, dim):
    v = torch.randn(6, 4, 3).to(dtype)
    g = torch.randn(6, 1, 1) if dim == 0 else torch.randn(1, 1, 3)
    g = g.to(dtype)
    with ran("aten::_weight_norm_interface"):
        w, norms = torch._weight_norm_interface(v.to(mojo_gpu), g.to(mojo_gpu), dim)
    ref_w, ref_n = torch._weight_norm_interface(v.float(), g.float(), dim)
    assert norms.dtype == (torch.float32 if dtype != torch.float32 else dtype)
    atol, rtol = (None, None) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(w.cpu().float(), ref_w, atol=atol, rtol=rtol)
    torch.testing.assert_close(norms.cpu(), ref_n, atol=1e-3, rtol=1e-3)


def test_weight_norm_through_nn_utils_backward(mojo_gpu):
    torch.manual_seed(0)
    lin = torch.nn.Linear(5, 4)
    lin_dev = torch.nn.Linear(5, 4).to(mojo_gpu)
    lin_dev.load_state_dict({k: v.to(mojo_gpu) for k, v in lin.state_dict().items()})
    lin = torch.nn.utils.parametrizations.weight_norm(lin)
    lin_dev = torch.nn.utils.parametrizations.weight_norm(lin_dev)
    x = torch.randn(3, 5)
    lin(x).sum().backward()
    with ran("aten::_weight_norm_interface_backward"):
        lin_dev(x.to(mojo_gpu)).sum().backward()
    for (_, p), (_, q) in zip(lin.named_parameters(), lin_dev.named_parameters()):
        torch.testing.assert_close(q.grad.cpu(), p.grad, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# _fused_rms_norm_backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_weight", [True, False])
def test_fused_rms_norm_backward_matches_autograd(mojo_gpu, dtype, with_weight):
    x = torch.randn(4, 3, 8)
    w = torch.randn(8)
    dy = torch.randn(4, 3, 8)
    eps = 1e-6
    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    xi = x.clone().requires_grad_()
    wi = w.clone().requires_grad_()
    y = xi * torch.rsqrt(xi.pow(2).mean(-1, keepdim=True) + eps)
    if with_weight:
        y = y * wi
    y.backward(dy)

    def dev(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype).to(mojo_gpu)

    with ran("aten::_fused_rms_norm_backward"):
        dx, dw = torch.ops.aten._fused_rms_norm_backward(
            dev(dy),
            dev(x),
            [8],
            rstd.to(mojo_gpu),
            dev(w) if with_weight else None,
            [True, with_weight],
        )
    atol, rtol = (1e-5, 1e-4) if dtype == torch.float32 else (5e-2, 5e-2)
    torch.testing.assert_close(dx.cpu().float(), xi.grad, atol=atol, rtol=rtol)
    if with_weight:
        torch.testing.assert_close(dw.cpu().float(), wi.grad, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# _compute_linear_combination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64])
def test_compute_linear_combination(mojo_gpu, dtype):
    inp = (torch.randn(4, 5, 2) * 3).to(dtype)
    coeff = (torch.randn(3, 4) * 3).to(dtype)
    with ran("aten::_compute_linear_combination"):
        got = torch.ops.aten._compute_linear_combination(
            inp.to(mojo_gpu), coeff.to(mojo_gpu)
        )
    ref = torch.ops.aten._compute_linear_combination(inp, coeff)
    atol, rtol = (2e-2, 2e-2) if dtype == torch.float16 else (None, None)
    torch.testing.assert_close(got.cpu(), ref, atol=atol, rtol=rtol)
    # .out accumulates onto the out's current contents, as ATen's kernel does.
    out = torch.ones(3, 5, 2, dtype=dtype, device=mojo_gpu)
    torch.ops.aten._compute_linear_combination(
        inp.to(mojo_gpu), coeff.to(mojo_gpu), out=out
    )
    torch.testing.assert_close(out.cpu(), ref + 1, atol=atol, rtol=rtol)
    with pytest.raises(RuntimeError, match="Empty tensor not supported"):
        torch.ops.aten._compute_linear_combination(
            torch.empty(0, device=mojo_gpu), coeff.to(mojo_gpu)
        )


# ---------------------------------------------------------------------------
# histogram / histogramdd (CPU semantics: CUDA has no kernel)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bins", [1, 4, 10])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("density", [False, True])
def test_histogram_matches_cpu(mojo_gpu, bins, weighted, density):
    x = torch.randn(200)
    w = torch.rand(200)
    dev_w = w.to(mojo_gpu)
    if not weighted:
        w, dev_w = None, None
    with ran("aten::histogram.bin_ct"):
        got = torch.histogram(x.to(mojo_gpu), bins, weight=dev_w, density=density)
    ref = torch.histogram(x, bins, weight=w, density=density)
    torch.testing.assert_close(got.hist.cpu(), ref.hist)
    torch.testing.assert_close(got.bin_edges.cpu(), ref.bin_edges)
    edges = torch.sort(torch.randn(bins + 1))[0]
    with ran("aten::histogram.bins_tensor"):
        got = torch.histogram(
            x.to(mojo_gpu), edges.to(mojo_gpu), weight=dev_w, density=density
        )
    ref = torch.histogram(x, edges, weight=w, density=density)
    torch.testing.assert_close(got.hist.cpu(), ref.hist)
    torch.testing.assert_close(got.bin_edges.cpu(), ref.bin_edges)


def test_histogram_range_edges_scalar_and_empty(mojo_gpu):
    x = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0, float("nan")])
    got = torch.histogram(x.to(mojo_gpu), 4, range=(-1.0, 1.0))
    ref = torch.histogram(x, 4, range=(-1.0, 1.0))
    torch.testing.assert_close(got.hist.cpu(), ref.hist)
    s = torch.tensor(1.5)
    got = torch.histogram(s.to(mojo_gpu), 3)
    ref = torch.histogram(s, 3)
    torch.testing.assert_close(got.hist.cpu(), ref.hist)
    torch.testing.assert_close(got.bin_edges.cpu(), ref.bin_edges)
    got = torch.histogram(torch.empty(0, device=mojo_gpu), 3)
    torch.testing.assert_close(
        got.bin_edges.cpu(), torch.histogram(torch.empty(0), 3).bin_edges
    )
    with pytest.raises(RuntimeError, match="bins must be > 0"):
        torch.histogram(x.to(mojo_gpu), 0)


@pytest.mark.parametrize("bins", [[2, 3, 4], 3])
@pytest.mark.parametrize("density", [False, True])
def test_histogramdd_matches_cpu(mojo_gpu, bins, density):
    x = torch.randn(40, 3)
    w = torch.rand(40)
    got = torch.histogramdd(
        x.to(mojo_gpu), bins, weight=w.to(mojo_gpu), density=density
    )
    ref = torch.histogramdd(x, bins, weight=w, density=density)
    torch.testing.assert_close(got.hist.cpu(), ref.hist)
    for g, r in zip(got.bin_edges, ref.bin_edges):
        torch.testing.assert_close(g.cpu(), r)
    edges = [torch.sort(torch.randn(k))[0] for k in (3, 5, 2)]
    got = torch.histogramdd(
        x.to(mojo_gpu), [e.to(mojo_gpu) for e in edges], density=density
    )
    ref = torch.histogramdd(x, edges, density=density)
    torch.testing.assert_close(got.hist.cpu(), ref.hist)
    with pytest.raises(RuntimeError, match="size of bins must be equal"):
        torch.histogramdd(x.to(mojo_gpu), [1, 1])


def test_histogramdd_five_dims_density(mojo_gpu):
    x = torch.randn(30, 5)
    got = torch.histogramdd(x.to(mojo_gpu), [2, 1, 2, 1, 2], density=True)
    ref = torch.histogramdd(x, [2, 1, 2, 1, 2], density=True)
    torch.testing.assert_close(got.hist.cpu(), ref.hist)
