"""Ops composed from registered ops through the dispatcher (ops_composed.mojo)."""

import contextlib

import pytest
import torch

from tests.native.conftest import skip_if_metal
from torch_mojo_backend import native
from torch_mojo_backend.native import device_module


@contextlib.contextmanager
def assert_ran(*op_names: str):
    """Assert that each aten op ran as a native boxed kernel in the block."""
    native.op_counting(True)
    before = native.op_counts()
    yield
    after = native.op_counts()
    for name in op_names:
        assert after.get(name, 0) > before.get(name, 0), (
            f"{name} did not run natively (counted: {sorted(after)})"
        )


@pytest.mark.parametrize("act", ["relu", "sigmoid", "tanh"])
def test_activation_backward_composed_through_the_dispatcher(mojo_gpu, act):
    """Backward registrations, including their composed fallbacks, must
    match CPU autograd."""
    torch.manual_seed(0)
    x = torch.randn(4, 7, device=mojo_gpu, requires_grad=True)
    y = getattr(torch, act)(x)
    grad = torch.randn_like(y)
    y.backward(grad)
    ref = x.detach().cpu().requires_grad_(True)
    getattr(torch, act)(ref).backward(grad.cpu())
    assert x.grad is not None and ref.grad is not None
    # two float32 rounding orders (tanh: out*out on device, 1 - out^2 on cpu)
    torch.testing.assert_close(x.grad.cpu(), ref.grad, atol=3e-5, rtol=1e-5)


def _tanh_f32_bits(size: int, seed: int) -> torch.Tensor:
    edges = torch.tensor(
        [
            0,
            0x80000000,
            1,
            0x80000001,
            0x007FFFFF,
            0x807FFFFF,
            0x00800000,
            0x80800000,
            0x3F800000,
            0xBF800000,
            0x3F7FFFFF,
            0xBF7FFFFF,
            0x3F800001,
            0xBF800001,
            0x3F7FFFFE,
            0xBF7FFFFE,
            0x3F800002,
            0xBF800002,
            0x3F000000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00001,
            0x7F800001,
        ],
        dtype=torch.int64,
    )
    indices = torch.arange(size, dtype=torch.int64)
    slots = (indices + seed) % 32
    bits = torch.where(
        slots < edges.numel(),
        edges[slots.clamp_max(edges.numel() - 1)],
        (indices * 2654435761 + seed) & 0xFFFFFFFF,
    )
    return bits.to(torch.int32).view(torch.float32)


def _assert_tanh_f32_bits(actual: torch.Tensor, expected: torch.Tensor):
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    assert torch.equal(actual.view(torch.int32)[~nan], expected.view(torch.int32)[~nan])


@pytest.mark.parametrize("goff", range(4))
@pytest.mark.parametrize("yoff", range(4))
@pytest.mark.parametrize("doff", range(4))
def test_tanh_backward_f32_pointer_residues(mojo_gpu, goff, yoff, doff):
    _check_tanh_backward_f32(mojo_gpu, 1029, goff, yoff, doff, "disjoint")


@pytest.mark.parametrize(
    "size",
    [
        0,
        1,
        2,
        3,
        4,
        5,
        7,
        15,
        16,
        17,
        255,
        256,
        257,
        1023,
        1024,
        1025,
        1026,
        1027,
        1028,
        4095,
        4096,
        4097,
    ],
)
def test_tanh_backward_f32_tails(mojo_gpu, size):
    _check_tanh_backward_f32(mojo_gpu, size, 0, 0, 0, "disjoint")


@pytest.mark.parametrize("offset", range(4))
@pytest.mark.parametrize("alias", ["disjoint", "grad", "output", "inputs", "all"])
def test_tanh_backward_f32_large_aliases(mojo_gpu, offset, alias):
    _check_tanh_backward_f32(mojo_gpu, 357 * 789, offset, offset, offset, alias)


def _check_tanh_backward_f32(
    device: str, size: int, goff: int, yoff: int, doff: int, alias: str
):
    properties = device_module.get_device_properties(device)
    if properties.api != "cuda" or properties.major != 9:
        pytest.skip("CUDA Hopper fused FMA contract")
    gh = _tanh_f32_bits(size + 16, 0)
    yh = _tanh_f32_bits(size + 16, 13)
    gb, yb = gh.to(device), yh.to(device)
    db = torch.full((size + 16,), 17.0, device=device)
    if alias in ("inputs", "all"):
        yb, yh = gb, gh
    if alias in ("grad", "all"):
        db = gb
    elif alias == "output":
        db = yb
    g, y, out = gb[goff : goff + size], yb[yoff : yoff + size], db[doff : doff + size]
    before = db.cpu()
    expected = (
        gh[goff : goff + size] * (1 - yh[yoff : yoff + size].double().square()).float()
    )
    functional = torch.ops.aten.tanh_backward(g, y)
    _assert_tanh_f32_bits(functional.cpu(), expected)
    version = out._version
    result = torch.ops.aten.tanh_backward.grad_input(g, y, grad_input=out)
    assert result is out
    assert out._version == version + 1
    _assert_tanh_f32_bits(out.cpu(), expected)
    after = db.cpu()
    assert torch.equal(after[:doff].view(torch.int32), before[:doff].view(torch.int32))
    assert torch.equal(
        after[doff + size :].view(torch.int32), before[doff + size :].view(torch.int32)
    )
    if alias not in ("grad", "all"):
        assert torch.equal(gb.cpu().view(torch.int32), gh.view(torch.int32))
    if alias not in ("output", "all"):
        assert torch.equal(yb.cpu().view(torch.int32), yh.view(torch.int32))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("layout", ["transpose", "broadcast", "resize", "strided_out"])
def test_tanh_backward_generic_fallback(mojo_gpu, dtype, layout):
    g = torch.linspace(-0.5, 0.5, 35).reshape(5, 7).to(dtype)
    y = torch.linspace(-0.75, 0.75, 35).reshape(5, 7).to(dtype)
    gm, ym = g.to(mojo_gpu), y.to(mojo_gpu)
    if layout == "transpose":
        g, y, gm, ym = g.t(), y.t(), gm.t(), ym.t()
    elif layout == "broadcast":
        g, gm = g[:1], gm[:1]
    # float64 stays on the unchanged composed route (GPU neg accepts it
    # since #499), so every dtype is checked against CPU the same way.
    expected = torch.ops.aten.tanh_backward(g, y)
    result = torch.ops.aten.tanh_backward(gm, ym)
    rtol, atol = (
        (0.02, 0.002) if dtype in (torch.float16, torch.bfloat16) else (None, None)
    )
    torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
    if layout == "resize":
        out = torch.empty(0, dtype=dtype, device=mojo_gpu)
    elif layout == "strided_out":
        out = torch.empty((5, 14), dtype=dtype, device=mojo_gpu)[:, ::2]
    else:
        out = torch.empty_like(result)
    returned = torch.ops.aten.tanh_backward.grad_input(gm, ym, grad_input=out)
    assert returned is out
    torch.testing.assert_close(out.cpu(), expected, rtol=rtol, atol=atol)


def test_relu_module_trains(mojo_gpu):
    layer = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU()).to(mojo_gpu)
    out = layer(torch.randn(3, 8, device=mojo_gpu)).sum()
    out.backward()
    assert layer[0].weight.grad is not None


def test_isneginf_isposinf(mojo_device):
    x = torch.tensor([float("-inf"), -1.0, 0.0, float("inf"), float("nan")]).to(
        mojo_device
    )
    assert torch.isneginf(x).cpu().tolist() == [True, False, False, False, False]
    assert torch.isposinf(x).cpu().tolist() == [False, False, False, True, False]
    out = torch.empty(5, dtype=torch.bool, device=mojo_device)
    torch.isneginf(x, out=out)
    assert out.cpu().tolist() == [True, False, False, False, False]
    assert not torch.isposinf(torch.arange(3, device=mojo_device)).cpu().any()


# ---------------------------------------------------------------------------
# where.self_out
# ---------------------------------------------------------------------------


def test_where_self_out(mojo_gpu):
    cond = torch.rand(4, 5, device=mojo_gpu) > 0.5
    a = torch.randn(4, 5, device=mojo_gpu)
    b = torch.randn(4, 5, device=mojo_gpu)
    out = torch.empty(4, 5, device=mojo_gpu)
    with assert_ran("aten::where.self_out"):
        got = torch.where(cond, a, b, out=out)
    assert got.data_ptr() == out.data_ptr()
    expected = torch.where(cond.cpu(), a.cpu(), b.cpu())
    torch.testing.assert_close(out.cpu(), expected)


def test_where_self_out_resizes_and_broadcasts(mojo_gpu):
    """An `out=` of the wrong shape is resized, exactly like ATen's
    TensorIterator does for an ordinary backend."""
    cond = torch.rand(3, 1, device=mojo_gpu) > 0.5
    a = torch.randn(3, 4, device=mojo_gpu)
    b = torch.randn(1, 4, device=mojo_gpu)
    out = torch.empty(0, device=mojo_gpu)
    with assert_ran("aten::where.self_out"):
        torch.where(cond, a, b, out=out)
    assert tuple(out.shape) == (3, 4)
    torch.testing.assert_close(out.cpu(), torch.where(cond.cpu(), a.cpu(), b.cpu()))


def test_where_self_out_rejects_the_wrong_out_dtype(mojo_gpu):
    cond = torch.rand(4, device=mojo_gpu) > 0.5
    a = torch.randn(4, device=mojo_gpu)
    b = torch.randn(4, device=mojo_gpu)
    out = torch.empty(4, dtype=torch.float64, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="out type"):
        torch.where(cond, a, b, out=out)


# ---------------------------------------------------------------------------
# native_batch_norm_backward
# ---------------------------------------------------------------------------


def _bn_grads(module, x, grad, device=None):
    """One forward + backward of `module` on `x`, returning the output, the
    input gradient and the parameter gradients."""
    if device is not None:
        module = module.to(device)
        x = x.to(device)
        grad = grad.to(device)
    x = x.detach().requires_grad_(True)
    y = module(x)
    y.backward(grad)
    params = [p.grad for p in module.parameters()]
    return y, x.grad, params


def _close(got, want, dtype):
    atol, rtol = (1e-4, 1e-4) if dtype is torch.float32 else (3e-2, 3e-2)
    torch.testing.assert_close(
        got.float().cpu(), want.float().cpu(), atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("track", [True, False])
def test_batch_norm2d_training_step(mojo_gpu, dtype, affine, track):
    """A BatchNorm2d training step: the three gradients and the running-stat
    update must match CPU autograd."""
    torch.manual_seed(0)
    x = torch.randn(4, 3, 5, 6)
    grad = torch.randn(4, 3, 5, 6)
    ref = torch.nn.BatchNorm2d(3, affine=affine, track_running_stats=track)
    if affine:
        with torch.no_grad():
            ref.weight.copy_(torch.linspace(0.5, 1.5, 3))
            ref.bias.copy_(torch.linspace(-0.2, 0.2, 3))
    ours = torch.nn.BatchNorm2d(3, affine=affine, track_running_stats=track)
    ours.load_state_dict(ref.state_dict())

    y_ref, gx_ref, gp_ref = _bn_grads(ref, x, grad)
    with assert_ran("aten::native_batch_norm_backward"):
        y, gx, gp = _bn_grads(ours, x.to(dtype), grad.to(dtype), device=mojo_gpu)

    _close(y, y_ref, dtype)
    _close(gx, gx_ref, dtype)
    assert gx.dtype == dtype
    assert len(gp) == (2 if affine else 0)
    for got, want in zip(gp, gp_ref):
        _close(got, want, dtype)
        assert got.dtype == torch.float32
    if track:
        _close(ours.running_mean, ref.running_mean, dtype)
        _close(ours.running_var, ref.running_var, dtype)
        batches = ours.num_batches_tracked
        assert batches is not None and int(batches.cpu()) == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batch_norm2d_eval_backward(mojo_gpu, dtype):
    """In eval mode the formula reads the running statistics instead of the
    saved ones and drops the two mean-removal terms."""
    torch.manual_seed(1)
    x = torch.randn(2, 4, 3, 3)
    grad = torch.randn(2, 4, 3, 3)
    ref = torch.nn.BatchNorm2d(4)
    with torch.no_grad():
        ref.running_mean = torch.linspace(-1.0, 1.0, 4)
        ref.running_var = torch.linspace(0.5, 2.0, 4)
    ours = torch.nn.BatchNorm2d(4)
    ours.load_state_dict(ref.state_dict())
    ref.eval()
    ours.eval()

    _, gx_ref, gp_ref = _bn_grads(ref, x, grad)
    with assert_ran("aten::native_batch_norm_backward"):
        _, gx, gp = _bn_grads(ours, x.to(dtype), grad.to(dtype), device=mojo_gpu)
    _close(gx, gx_ref, dtype)
    for got, want in zip(gp, gp_ref):
        _close(got, want, dtype)


def test_batch_norm1d_rank3(mojo_gpu):
    """Rank 3 ([N, C, L]): the reduce dims are [0, 2]."""
    torch.manual_seed(2)
    x = torch.randn(6, 5, 7)
    grad = torch.randn(6, 5, 7)
    ref = torch.nn.BatchNorm1d(5)
    ours = torch.nn.BatchNorm1d(5)
    ours.load_state_dict(ref.state_dict())

    _, gx_ref, gp_ref = _bn_grads(ref, x, grad)
    with assert_ran("aten::native_batch_norm_backward"):
        _, gx, gp = _bn_grads(ours, x, grad, device=mojo_gpu)
    _close(gx, gx_ref, torch.float32)
    for got, want in zip(gp, gp_ref):
        _close(got, want, torch.float32)


def test_batch_norm3d_rank5(mojo_gpu):
    """Rank 5 goes through the [N, C, HxW] collapse, since the broadcast
    binary kernels stop at rank 4."""
    torch.manual_seed(3)
    x = torch.randn(2, 3, 2, 3, 4)
    grad = torch.randn(2, 3, 2, 3, 4)
    ref = torch.nn.BatchNorm3d(3)
    ours = torch.nn.BatchNorm3d(3)
    ours.load_state_dict(ref.state_dict())

    _, gx_ref, gp_ref = _bn_grads(ref, x, grad)
    with assert_ran("aten::native_batch_norm_backward"):
        _, gx, gp = _bn_grads(ours, x, grad, device=mojo_gpu)
    assert tuple(gx.shape) == (2, 3, 2, 3, 4)
    _close(gx, gx_ref, torch.float32)
    for got, want in zip(gp, gp_ref):
        _close(got, want, torch.float32)


def test_batch_norm_input_grad_only(mojo_gpu):
    """output_mask = [True, False, False]: the two affine gradients are the
    0-element stand-ins and autograd never reads them."""
    torch.manual_seed(4)
    x = torch.randn(3, 4, 2, 2)
    grad = torch.randn(3, 4, 2, 2)
    ref = torch.nn.BatchNorm2d(4)
    ours = torch.nn.BatchNorm2d(4)
    ours.load_state_dict(ref.state_dict())
    for p in ours.parameters():
        p.requires_grad_(False)
    for p in ref.parameters():
        p.requires_grad_(False)

    _, gx_ref, _ = _bn_grads(ref, x, grad)
    with assert_ran("aten::native_batch_norm_backward"):
        _, gx, _ = _bn_grads(ours, x, grad, device=mojo_gpu)
    _close(gx, gx_ref, torch.float32)


# Every output_mask autograd can ask for (all-False never reaches the op).
_MASKS = [
    (True, True, True),
    (True, False, False),
    (False, True, True),
    (False, True, False),
    (False, False, True),
    (True, True, False),
    (True, False, True),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize(
    "affine,mask",
    [(True, m) for m in _MASKS] + [(False, (True, False, False))],
    ids=str,
)
def test_native_batch_norm_backward_masks(mojo_gpu, dtype, train, affine, mask):
    """Every output_mask, both modes, every half type, on an odd shape (C=7,
    N=1): the AMP layout, a half input with float32 per-channel buffers.
    Without a weight autograd only ever asks for grad_input."""
    torch.manual_seed(7)
    shape = (1, 7, 5, 9)
    x = torch.randn(shape).to(dtype)
    grad = torch.randn(shape).to(dtype)
    weight = torch.randn(7) if affine else None
    running_mean = torch.randn(7)
    running_var = torch.rand(7) + 0.5
    save_mean = x.float().mean(dim=(0, 2, 3))
    save_invstd = torch.rsqrt(x.float().var(dim=(0, 2, 3), unbiased=False) + 1e-5)
    cpu = [grad.float(), x.float(), weight, running_mean, running_var]
    stats = [save_mean, save_invstd]
    want = torch.ops.aten.native_batch_norm_backward(
        *cpu, *stats, train, 1e-5, list(mask)
    )
    dev = [None if t is None else t.to(mojo_gpu) for t in [grad, x, weight]]
    rest = [t.to(mojo_gpu) for t in [running_mean, running_var, *stats]]
    with assert_ran("aten::native_batch_norm_backward"):
        got = torch.ops.aten.native_batch_norm_backward(
            *dev, *rest, train, 1e-5, list(mask)
        )
    for g, w, wanted in zip(got, want, mask):
        if wanted:
            _close(g, w, dtype)
    assert got[0] is None or not mask[0] or got[0].dtype == dtype


# ---------------------------------------------------------------------------
# native_group_norm_backward
# ---------------------------------------------------------------------------


def _gn_close(got, want, dtype):
    # float32: sums over up to 64K elements accumulated in two orders; the
    # half types round the input and grad_out before a float32 reduction.
    atol, rtol = (2e-4, 2e-4) if dtype is torch.float32 else (5e-2, 3e-2)
    torch.testing.assert_close(
        got.float().cpu(), want.float().cpu(), atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "n,c,hxw,group",
    [
        (2, 6, 12, 3),
        (1, 9, 13, 3),  # N = 1, odd channels and spatial
        (2, 10, 33, 1),  # group = 1: layer norm over the whole sample
        (2, 10, 33, 10),  # group = C: instance norm
        (3, 7, 5, 7),
        (1, 32, 4096, 8),  # large spatial extent: K = 16384
    ],
)
@pytest.mark.parametrize("affine", [True, False])
def test_native_group_norm_backward(mojo_gpu, dtype, n, c, hxw, group, affine):
    torch.manual_seed(8)
    x = torch.randn(n, c, hxw).to(dtype)
    grad = torch.randn(n, c, hxw).to(dtype)
    weight = torch.randn(c).to(dtype) if affine else None
    mask = [True, True, True] if affine else [True, False, False]
    wf = None if weight is None else weight.float()
    _, mean, rstd = torch.ops.aten.native_group_norm(
        x.float(), wf, None, n, c, hxw, group, 1e-5
    )
    want = torch.ops.aten.native_group_norm_backward(
        grad.float(), x.float(), mean, rstd, wf, n, c, hxw, group, mask
    )
    dw = None if weight is None else weight.to(mojo_gpu)
    dx = x.to(mojo_gpu)
    _, dmean, drstd = torch.ops.aten.native_group_norm(
        dx, dw, None, n, c, hxw, group, 1e-5
    )
    with assert_ran("aten::native_group_norm_backward"):
        got = torch.ops.aten.native_group_norm_backward(
            grad.to(mojo_gpu), dx, dmean, drstd, dw, n, c, hxw, group, mask
        )
    for g, w, wanted in zip(got, want, mask):
        if wanted:
            assert g.dtype == dtype
            _gn_close(g, w, dtype)
        else:
            assert g is None


@pytest.mark.parametrize("mask", _MASKS, ids=str)
def test_native_group_norm_backward_masks(mojo_gpu, mask):
    """Only the requested gradients come back; the rest are None."""
    torch.manual_seed(9)
    n, c, hxw, group = 2, 6, 10, 3
    x = torch.randn(n, c, hxw)
    grad = torch.randn(n, c, hxw)
    weight = torch.randn(c)
    _, mean, rstd = torch.ops.aten.native_group_norm(
        x, weight, None, n, c, hxw, group, 1e-5
    )
    want = torch.ops.aten.native_group_norm_backward(
        grad, x, mean, rstd, weight, n, c, hxw, group, list(mask)
    )
    dev = [t.to(mojo_gpu) for t in [grad, x, mean, rstd, weight]]
    with assert_ran("aten::native_group_norm_backward"):
        got = torch.ops.aten.native_group_norm_backward(
            *dev, n, c, hxw, group, list(mask)
        )
    assert [g is not None for g in got] == list(mask)
    for g, w, wanted in zip(got, want, mask):
        if wanted:
            _gn_close(g, w, torch.float32)


def test_native_group_norm_backward_strided(mojo_gpu):
    """A non-contiguous grad_out (what a transposed consumer hands back).

    The reference gets a contiguous copy: CPU ATen's kernel reads this
    transposed layout as if it were channels-last and returns wrong values."""
    torch.manual_seed(10)
    n, c, h, w, group = 2, 6, 5, 4, 2
    x = torch.randn(n, c, h, w)
    grad = torch.randn(n, c, w, h).transpose(2, 3)
    weight = torch.randn(c)
    mask = [True, True, True]
    _, mean, rstd = torch.ops.aten.native_group_norm(
        x, weight, None, n, c, h * w, group, 1e-5
    )
    want = torch.ops.aten.native_group_norm_backward(
        grad.contiguous(), x, mean, rstd, weight, n, c, h * w, group, mask
    )
    dev = [t.to(mojo_gpu) for t in [x, mean, rstd, weight]]
    with assert_ran("aten::native_group_norm_backward"):
        got = torch.ops.aten.native_group_norm_backward(
            grad.to(mojo_gpu), *dev[:3], dev[3], n, c, h * w, group, mask
        )
    assert got[0].shape == x.shape
    for g, want_t in zip(got, want):
        _gn_close(g, want_t, torch.float32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("groups", [1, 3, 6])
def test_group_norm_autograd(mojo_gpu, dtype, affine, groups):
    """`F.group_norm` forward then `.backward()`: all three grads vs CPU."""
    torch.manual_seed(11)
    x = torch.randn(2, 6, 5, 7)
    weight = torch.randn(6)
    bias = torch.randn(6)
    grad = torch.randn(2, 6, 5, 7)

    def step(device, dt):
        def leaf_of(t):
            return t.to(dt).to(device).detach().requires_grad_(True)

        leaf = leaf_of(x)
        w = leaf_of(weight) if affine else None
        b = leaf_of(bias) if affine else None
        y = torch.nn.functional.group_norm(leaf, groups, w, b)
        y.backward(grad.to(dt).to(device))
        if w is None or b is None:
            return [leaf.grad]
        return [leaf.grad, w.grad, b.grad]

    want = step("cpu", torch.float32)
    with assert_ran("aten::native_group_norm_backward"):
        got = step(mojo_gpu, dtype)
    for g, w in zip(got, want):
        assert g.dtype == dtype
        _gn_close(g, w, dtype)


def test_group_norm_module_training_step(mojo_gpu):
    """nn.GroupNorm inside a small conv-free block trains like on CPU."""
    torch.manual_seed(12)
    ref = torch.nn.GroupNorm(4, 12)
    ours = torch.nn.GroupNorm(4, 12)
    ours.load_state_dict(ref.state_dict())
    x = torch.randn(3, 12, 9, 11)
    grad = torch.randn(3, 12, 9, 11)
    _, gx_ref, gp_ref = _bn_grads(ref, x, grad)
    with assert_ran("aten::native_group_norm_backward"):
        _, gx, gp = _bn_grads(ours, x, grad, device=mojo_gpu)
    _gn_close(gx, gx_ref, torch.float32)
    for got, want in zip(gp, gp_ref):
        _gn_close(got, want, torch.float32)


# ---------------------------------------------------------------------------
# _softmax_backward_data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_softmax_backward(mojo_gpu, dim):
    torch.manual_seed(5)
    x = torch.randn(3, 4, 5)
    grad = torch.randn(3, 4, 5)
    ref = x.clone().requires_grad_(True)
    torch.softmax(ref, dim=dim).backward(grad)

    ours = x.to(mojo_gpu).requires_grad_(True)
    with assert_ran("aten::_softmax_backward_data"):
        torch.softmax(ours, dim=dim).backward(grad.to(mojo_gpu))
    assert ours.grad is not None
    torch.testing.assert_close(ours.grad.cpu(), ref.grad, atol=1e-5, rtol=1e-5)


def _softmax_backward_cuda(
    grad: torch.Tensor, out: torch.Tensor, dim: int, dtype: torch.dtype
) -> torch.Tensor:
    """SoftMax.cu's softmax_backward_cuda_out, rounding for rounding: `tmp =
    grad * out` in the gradient's dtype, then `tmp - out * sum(tmp)` in float
    (accscalar_t), rounded once to `dtype`."""
    tmp = (grad * out).float()
    total = tmp.sum(dim, keepdim=True)
    return (tmp - out.float() * total).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [0, -1])
def test_softmax_backward_half(mojo_gpu, dtype, dim):
    """The half product `grad * out` is rounded before the float reduction,
    as CUDA's is."""
    torch.manual_seed(6)
    out = torch.softmax(torch.randn(6, 50) * 3, dim).to(dtype)
    grad = torch.randn(6, 50).to(dtype)
    want = _softmax_backward_cuda(grad, out, dim, dtype)
    with assert_ran("aten::_softmax_backward_data"):
        got = torch.ops.aten._softmax_backward_data(
            grad.to(mojo_gpu), out.to(mojo_gpu), dim, dtype
        )
    assert got.dtype == dtype
    torch.testing.assert_close(got.cpu(), want, atol=1e-5, rtol=2e-3)


def test_softmax_backward_half_rounds_the_product(mojo_gpu):
    """bf16 output [0.7, 0.3], grad [3, 3]: in exact math the gradient is 0,
    but CUDA rounds grad * out to bf16 first and gets about +-0.001175."""
    out = torch.tensor([0.7, 0.3], dtype=torch.bfloat16)
    grad = torch.tensor([3.0, 3.0], dtype=torch.bfloat16)
    want = _softmax_backward_cuda(grad, out, 0, torch.bfloat16)
    assert want.abs().max() > 1e-3
    got = torch.ops.aten._softmax_backward_data(
        grad.to(mojo_gpu), out.to(mojo_gpu), 0, torch.bfloat16
    )
    torch.testing.assert_close(got.cpu(), want, atol=0, rtol=0)


def test_softmax_backward_half_to_float(mojo_gpu):
    """half_to_float's backward: a float gradient narrowed to float16, the
    only mismatch CUDA accepts."""
    torch.manual_seed(7)
    out = torch.softmax(torch.randn(4, 30), -1)
    grad = torch.randn(4, 30)
    want = _softmax_backward_cuda(grad, out, -1, torch.float16)
    got = torch.ops.aten._softmax_backward_data(
        grad.to(mojo_gpu), out.to(mojo_gpu), -1, torch.float16
    )
    assert got.dtype == torch.float16
    torch.testing.assert_close(got.cpu(), want, atol=1e-5, rtol=2e-3)
    with pytest.raises(RuntimeError, match="input to be at::Half"):
        torch.ops.aten._softmax_backward_data(
            grad.to(mojo_gpu), out.to(mojo_gpu), -1, torch.bfloat16
        )


# ---------------------------------------------------------------------------
# trace / dot / vdot (and inner, a composite over dot)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool]
)
@pytest.mark.parametrize("shape", [(7, 9), (9, 7), (5, 5), (0, 3)])
def test_trace(mojo_gpu, dtype, shape):
    x = (torch.randn(shape) * 10).to(dtype)
    with assert_ran("aten::trace"):
        got = torch.trace(x.to(mojo_gpu))
    # CUDA's `trace_cuda` is `diagonal().sum()` for every dtype (CPU's trace
    # has no half/bool kernel).
    expected = x.diagonal().sum()
    assert got.dtype == expected.dtype
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-2, atol=1e-2)
    got_t = torch.trace(x.to(mojo_gpu).t()).cpu()
    torch.testing.assert_close(got_t, x.t().diagonal().sum(), rtol=1e-2, atol=1e-2)
    with pytest.raises(RuntimeError, match="expected a matrix"):
        torch.trace(torch.zeros(2, 2, 2, device=mojo_gpu))


@pytest.mark.parametrize("op", ["dot", "vdot", "inner"])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.int32]
)
def test_dot(mojo_gpu, op, dtype):
    fn = getattr(torch, op)
    g = torch.Generator().manual_seed(0)
    a = (torch.randn(37, generator=g) * 4).to(dtype)
    b = (torch.randn(37, generator=g) * 4).to(dtype)
    with assert_ran("aten::vdot" if op == "vdot" else "aten::dot"):
        got = fn(a.to(mojo_gpu), b.to(mojo_gpu))
    expected = fn(a.double(), b.double()).to(dtype)
    assert got.dtype == dtype
    tol = (
        0 if not dtype.is_floating_point else (1e-4 if dtype == torch.float32 else 1e-2)
    )
    torch.testing.assert_close(got.cpu(), expected, rtol=tol, atol=tol)
    # Strided operands: columns of a row-major matrix (stride 4) and a
    # reversed-step slice.
    s = (torch.randn(37, 4, generator=g) * 4).to(dtype).to(mojo_gpu)
    u, v = s[:, 1], s[:, 3]
    assert not u.is_contiguous() and not v.is_contiguous()
    if op == "inner" and dtype == torch.int32:
        return  # strided int32 inner decomposes to an int32 sum, not ours
    torch.testing.assert_close(
        fn(u, v).cpu(),
        fn(u.cpu().double(), v.cpu().double()).to(dtype),
        rtol=tol or 1e-2,
        atol=tol or 1e-2,
    )


def test_dot_errors(mojo_gpu):
    a = torch.ones(3, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="1D tensors expected"):
        torch.dot(torch.ones(1, 1, device=mojo_gpu), a)
    with pytest.raises(RuntimeError, match="expected both vectors to have same dtype"):
        torch.dot(a, a.half())
    with pytest.raises(RuntimeError, match="inconsistent tensor size"):
        torch.dot(torch.ones(9, device=mojo_gpu), a)
    with pytest.raises(RuntimeError, match="not implemented for 'Bool'"):
        torch.dot(a.bool(), a.bool())


# ---------------------------------------------------------------------------
# linalg_cross (and cross, a composite over it)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float64,
        torch.int64,
        torch.int32,
    ],
)
@pytest.mark.parametrize(
    ("xs", "ys", "dim"),
    [
        ((5, 3), (5, 3), -1),
        ((3, 4), (3, 4), 0),
        ((2, 1, 3, 6), (1, 4, 3, 6), 2),  # batch dims broadcast
        ((0, 3), (1, 3), 1),
    ],
)
def test_linalg_cross(mojo_gpu, dtype, xs, ys, dim):
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "Apple GPUs have no float64")
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(xs, generator=g) * 8).to(dtype)
    y = (torch.randn(ys, generator=g) * 8).to(dtype)
    with assert_ran("aten::linalg_cross"):
        got = torch.linalg.cross(x.to(mojo_gpu), y.to(mojo_gpu), dim=dim)
    # CPU's cross computes in the operand type like CUDA's `cross_kernel`
    # (c10::Half / BFloat16 round each product and the difference).
    expected = torch.linalg.cross(x, y, dim=dim)
    assert got.dtype == dtype and got.shape == expected.shape
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-5, atol=1e-4)


def test_linalg_cross_strided_out_and_cross(mojo_gpu):
    g = torch.Generator().manual_seed(1)
    base = torch.randn(4, 3, 10, generator=g)
    x = base[:, :, ::3].transpose(0, 2)  # (4, 3, 4) non-contiguous
    y = torch.randn(3, 4, generator=g).t().unsqueeze(0)  # (1, 4, 3)
    xm, ym = base.to(mojo_gpu)[:, :, ::3].transpose(0, 2), y.to(mojo_gpu)
    expected = torch.linalg.cross(x, y.transpose(1, 2).expand(4, 3, 4), dim=1)
    got = torch.linalg.cross(xm, ym.transpose(1, 2), dim=1)
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-5, atol=1e-5)
    # out=: an empty out is resized, a transposed out of the right shape is
    # written in place
    out = torch.empty(0, device=mojo_gpu)
    with assert_ran("aten::linalg_cross.out"):
        r = torch.linalg.cross(xm, ym.transpose(1, 2), dim=1, out=out)
    assert r is out
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=1e-5)
    out_t = torch.empty(4, 4, 3, device=mojo_gpu).transpose(1, 2)
    torch.linalg.cross(xm, ym.transpose(1, 2), dim=1, out=out_t)
    assert out_t.stride() == (12, 1, 3)
    torch.testing.assert_close(out_t.cpu(), expected, rtol=1e-5, atol=1e-5)
    # torch.cross picks the first dim of size 3 when dim is omitted
    a = torch.randn(3, 5, generator=g)
    torch.testing.assert_close(
        torch.cross(a.to(mojo_gpu), a.flip(1).to(mojo_gpu), dim=0).cpu(),
        torch.cross(a, a.flip(1), dim=0),
    )


def test_linalg_cross_special_values(mojo_gpu):
    x = torch.tensor(
        [[float("inf"), 0.0, 1.0], [float("nan"), 1.0, 2.0], [-0.0, 0.0, 0.0]]
    )
    y = torch.tensor([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0], [0.0, -0.0, 1.0]])
    got = torch.linalg.cross(x.to(mojo_gpu), y.to(mojo_gpu)).cpu()
    expected = torch.linalg.cross(x, y)
    torch.testing.assert_close(got, expected, equal_nan=True)
    # the sign of zeros (a NaN's sign bit is unspecified and differs by GPU)
    real = ~expected.isnan()
    assert torch.equal(torch.signbit(got[real]), torch.signbit(expected[real]))


def test_linalg_cross_errors(mojo_gpu):
    a = torch.ones(4, 3, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="same number of dimensions"):
        torch.linalg.cross(a, torch.ones(2, 4, 3, device=mojo_gpu))
    with pytest.raises(
        RuntimeError, match="dimension 0 must have length 3. Got 4 and 4"
    ):
        torch.linalg.cross(a, a, dim=0)
    with pytest.raises(IndexError, match="Dimension out of range"):
        torch.linalg.cross(a, a, dim=2)
    with pytest.raises(RuntimeError, match="must match the size of tensor b"):
        torch.linalg.cross(a, torch.ones(5, 3, device=mojo_gpu))
    with pytest.raises(RuntimeError, match="Found dtype Double but expected Float"):
        torch.linalg.cross(a, a.double())
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
        torch.linalg.cross(
            a, a, out=torch.empty(4, 3, device=mojo_gpu, dtype=torch.half)
        )
    with pytest.raises(RuntimeError, match="unsupported operation"):
        torch.linalg.cross(a, a, out=a)
    with pytest.raises(NotImplementedError, match="not implemented for 'Bool'"):
        torch.linalg.cross(a.bool(), a.bool())
    # an empty result never reaches the dtype dispatch, as on CUDA
    e = torch.ones(0, 3, device=mojo_gpu, dtype=torch.bool)
    assert torch.linalg.cross(e, e).shape == (0, 3)


# ---------------------------------------------------------------------------
# torch's test-only ops (CPU-only upstream; TestOps.cpp semantics)
# ---------------------------------------------------------------------------


def test_test_optional_lists(mojo_gpu):
    ops = torch.ops.aten
    v = torch.tensor([1, 2, 3, -2147483647], dtype=torch.int32)
    vm = v.to(mojo_gpu)
    with assert_ran("aten::_test_optional_intlist"):
        got = ops._test_optional_intlist(vm, [1, 2, 3, -2])
    assert torch.equal(got.cpu(), ops._test_optional_intlist(v, [1, 2, 3, -2]))
    assert ops._test_optional_intlist(vm, None) is vm
    assert torch.equal(
        ops._test_optional_filled_intlist(vm[:2], 5).cpu(),
        ops._test_optional_filled_intlist(v[:2], 5),
    )
    f = torch.tensor([1.0, 2.5, 3.0, 1e8, 4.0])
    fm = f.to(mojo_gpu)
    for addends in ([0.5, 1e-9, -2.0, 3.0, 7.0], [0.1] * 6):
        assert torch.equal(
            ops._test_optional_floatlist(fm[::2], addends).cpu(),
            ops._test_optional_floatlist(f[::2], addends),
        )
    out = torch.empty(7, dtype=torch.int32, device=mojo_gpu)
    r = ops._test_optional_intlist.out(vm, [1, 1, 1, 1], out=out)
    assert r is out and torch.equal(out.cpu(), v + 1)
    ops._test_optional_intlist.out(vm, None, out=out)
    assert torch.equal(out.cpu(), v)
    with pytest.raises(RuntimeError, match="ArrayRef: invalid index Index = 2"):
        ops._test_optional_intlist(vm, [1, 2])
    with pytest.raises(RuntimeError, match="expected scalar type Int but found Float"):
        ops._test_optional_intlist(fm, [1] * 5)
    with pytest.raises(RuntimeError, match=r"Expected values.dim\(\) == 1"):
        ops._test_optional_intlist(vm.view(2, 2), [1])
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
        ops._test_optional_intlist.out(vm, None, out=out.long())


def test_test_functorch_fallback(mojo_gpu):
    ops = torch.ops.aten
    x = torch.randn(3, 4)
    xm = x.to(mojo_gpu)
    with assert_ran("aten::_test_functorch_fallback"):
        got = ops._test_functorch_fallback(xm.t(), xm)
    assert torch.equal(got.cpu(), x.t()) and got.data_ptr() != xm.data_ptr()
    out = torch.empty(0, device=mojo_gpu)
    assert ops._test_functorch_fallback.out(xm, xm, out=out) is out
    assert torch.equal(out.cpu(), x)
