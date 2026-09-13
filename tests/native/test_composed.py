"""Ops composed from registered ops through the dispatcher (ops_composed.mojo)."""

import contextlib

import pytest
import torch

from torch_mojo_backend import native


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
    """threshold/sigmoid/tanh backward have no kernel of their own; they are
    composed from registered ops and must match CPU autograd."""
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
