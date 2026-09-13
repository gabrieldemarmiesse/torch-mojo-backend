"""Native-backend nn ops: softmax family, normalization, NLL loss, embedding,
2-D pooling and bilinear upsampling.

Everything here goes through public torch APIs on the mojo device and is
compared against the same call on CPU torch. The ops are exercised through
`torch.ops.aten.*` / `torch.nn.functional.*` rather than through autograd,
because the native backend registers ops group by group and a module-level
forward would pull in ops from groups that are not ported yet.
"""

import contextlib
import math

import pytest
import torch

from torch_mojo_backend import (
    aten_functions,
    get_accelerators,
    native,
    register_mojo_devices,
)
from torch_mojo_backend.testing import CallChecker

FLOAT_DTYPES = [torch.float32, torch.bfloat16, torch.float16]


@pytest.fixture(autouse=True)
def _registered():
    """`mojo_device` yields a device string without registering the backend."""
    register_mojo_devices()


@contextlib.contextmanager
def ran(*op_names: str):
    """Assert that at least one of `op_names` ran as a native boxed kernel.

    Used for the ops with no `aten_functions` twin for `CallChecker` to key
    on (the loss and the backward ops).
    """
    native.op_counting(True)
    before = {name: native.op_count(name) for name in op_names}
    yield
    assert any(native.op_count(name) > before[name] for name in op_names), (
        f"none of {op_names} ran natively"
    )


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    """(atol, rtol) for a dtype, as a tuple: `**kwargs` unpacking into
    `assert_close` defeats the type checker."""
    if dtype == torch.float32:
        return (1e-5, 1e-5)
    if dtype == torch.bfloat16:
        return (8e-3, 8e-3)
    return (2e-3, 2e-3)


# ---------------------------------------------------------------------------
# Softmax / log-softmax
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_softmax_every_dim(mojo_device, call_checker: CallChecker, dtype, dim):
    call_checker.register(aten_functions.aten__softmax)
    x = torch.randn(3, 4, 5).to(dtype)
    want = torch.softmax(x, dim)
    got = torch.softmax(x.to(mojo_device), dim)
    assert got.dtype == want.dtype
    assert tuple(got.shape) == tuple(want.shape)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(got.cpu(), want, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_log_softmax_every_dim(mojo_device, call_checker: CallChecker, dtype, dim):
    call_checker.register(aten_functions.aten__log_softmax)
    x = torch.randn(6, 7).to(dtype)
    want = torch.log_softmax(x, dim)
    got = torch.log_softmax(x.to(mojo_device), dim)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(got.cpu(), want, atol=atol, rtol=rtol)


@pytest.mark.parametrize("cols", [1, 3, 32, 257, 1024, 5000])
def test_log_softmax_row_widths(mojo_device, cols):
    x = torch.randn(3, cols)
    torch.testing.assert_close(
        torch.log_softmax(x.to(mojo_device), -1).cpu(),
        torch.log_softmax(x, -1),
        atol=1e-5,
        rtol=1e-5,
    )


def test_softmax_masked_and_infinite_rows(mojo_device):
    x = torch.full((2, 8), float("-inf"))
    x[0, ::2] = 0.0
    torch.testing.assert_close(
        torch.softmax(x.to(mojo_device), -1).cpu(), torch.softmax(x, -1), equal_nan=True
    )
    torch.testing.assert_close(
        torch.log_softmax(x.to(mojo_device), -1).cpu(),
        torch.log_softmax(x, -1),
        equal_nan=True,
    )


def test_softmax_strided_input(mojo_device):
    base = torch.randn(4, 12)
    view = base[:, ::2]
    torch.testing.assert_close(
        torch.softmax(base.to(mojo_device)[:, ::2], -1).cpu(),
        torch.softmax(view, -1),
        atol=1e-5,
        rtol=1e-5,
    )


def test_softmax_with_dtype_argument(mojo_device):
    """`dtype=` makes ATen cast on the host first, so the kernel sees fp32."""
    x = torch.randn(3, 9).to(torch.bfloat16)
    torch.testing.assert_close(
        torch.softmax(x.to(mojo_device), -1, dtype=torch.float32).cpu(),
        torch.softmax(x, -1, dtype=torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )


def test_softmax_rank0(mojo_device):
    x = torch.randn(())
    torch.testing.assert_close(
        torch.softmax(x.to(mojo_device), 0).cpu(), torch.softmax(x, 0)
    )


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_log_softmax_backward(mojo_gpu, call_checker: CallChecker, dtype, dim):
    call_checker.register(aten_functions.aten__log_softmax_backward_data)
    x = torch.randn(5, 11).to(dtype)
    output = torch.log_softmax(x, dim)
    grad = torch.randn(5, 11).to(dtype)
    want = torch.ops.aten._log_softmax_backward_data(grad, output, dim, dtype)
    got = torch.ops.aten._log_softmax_backward_data(
        grad.to(mojo_gpu), output.to(mojo_gpu), dim, dtype
    )
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(got.cpu(), want, atol=atol, rtol=rtol)


def test_log_softmax_backward_offset_view(mojo_gpu):
    """A contiguous but unaligned operand takes the materializing route."""
    backing = torch.randn(4, 65)
    grad = backing[:, 1:]
    output = torch.log_softmax(torch.randn(4, 64), -1)
    want = torch.ops.aten._log_softmax_backward_data(
        grad.contiguous(), output, -1, torch.float32
    )
    got = torch.ops.aten._log_softmax_backward_data(
        backing.to(mojo_gpu)[:, 1:], output.to(mojo_gpu), -1, torch.float32
    )
    torch.testing.assert_close(got.cpu(), want, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Layer norm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("has_weight", "has_bias"),
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize(
    ("shape", "normalized_shape", "eps"),
    [((3, 7), (7,), 1e-5), ((2, 3, 4), (3, 4), 0.5), ((2, 6, 768), (768,), 1e-5)],
)
def test_native_layer_norm(
    mojo_device,
    call_checker: CallChecker,
    has_weight,
    has_bias,
    shape,
    normalized_shape,
    eps,
):
    call_checker.register(aten_functions.aten_native_layer_norm)
    x = torch.randn(shape)
    w = torch.randn(normalized_shape) if has_weight else None
    b = torch.randn(normalized_shape) if has_bias else None
    want = torch.native_layer_norm(x, normalized_shape, w, b, eps)
    got = torch.native_layer_norm(
        x.to(mojo_device),
        normalized_shape,
        None if w is None else w.to(mojo_device),
        None if b is None else b.to(mojo_device),
        eps,
    )
    for g, e in zip(got, want, strict=True):
        assert g.dtype == e.dtype
        assert tuple(g.shape) == tuple(e.shape)
        torch.testing.assert_close(g.cpu(), e, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_layer_norm_reduced_precision(mojo_device, dtype):
    """The ACCELERATOR contract for the statistics, not the CPU one: CUDA
    allocates mean/rstd in `at::toAccumulateType(input.scalar_type(), true)`,
    i.e. float32 for both half types, while CPU returns them in the input
    dtype. The backward that consumes them is an accelerator kernel, so
    float32 is the contract to keep."""
    x = torch.randn(2, 6, 64).to(dtype)
    w = torch.randn(64).to(dtype)
    b = torch.randn(64).to(dtype)
    want = torch.native_layer_norm(x, (64,), w, b, 1e-5)
    got = torch.native_layer_norm(
        x.to(mojo_device), (64,), w.to(mojo_device), b.to(mojo_device), 1e-5
    )
    assert got[0].dtype == dtype
    assert got[1].dtype == torch.float32
    assert got[2].dtype == torch.float32
    torch.testing.assert_close(got[0].cpu(), want[0], atol=2e-2, rtol=2e-2)
    for i in (1, 2):
        torch.testing.assert_close(got[i].cpu(), want[i].float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_layer_norm_refuses_reduced_precision_training(mojo_gpu, dtype):
    """`native_layer_norm_backward` here covers float32 only. Refusing in the
    FORWARD keeps the traceback on the user's own frame instead of burying it
    inside the autograd engine."""
    x = torch.randn(2, 6, 64).to(dtype).to(mojo_gpu).requires_grad_()
    with pytest.raises(NotImplementedError, match="requires grad"):
        torch.native_layer_norm(x, (64,), None, None, 1e-5)
    # Inference is untouched, grad-requiring PARAMETERS included: under
    # torch.no_grad() the activation does not require grad, and the
    # parameters are deliberately not consulted (see the op's comment).
    plain = torch.randn(2, 6, 64).to(dtype).to(mojo_gpu)
    w = torch.randn(64).to(dtype).to(mojo_gpu).requires_grad_()
    b = torch.randn(64).to(dtype).to(mojo_gpu).requires_grad_()
    with torch.no_grad():
        torch.native_layer_norm(plain, (64,), w, b, 1e-5)


def test_layer_norm_module_forward(mojo_device):
    x = torch.randn(2, 6, 768)
    ln = torch.nn.LayerNorm(768).eval()
    with torch.no_grad():
        ln.weight.normal_()
        ln.bias.normal_()
    ln_dev = torch.nn.LayerNorm(768).eval()
    ln_dev.load_state_dict(ln.state_dict())
    ln_dev = ln_dev.to(mojo_device)
    with torch.no_grad():
        torch.testing.assert_close(
            ln_dev(x.to(mojo_device)).cpu(), ln(x), atol=1e-5, rtol=1e-5
        )


def test_native_layer_norm_no_affine_repeats(mojo_device):
    """No weight and no bias: the classic route synthesizes ones/zeros and has
    to order that fill against the kernel that reads it. The bug this catches
    fired about one call in ten, so the check repeats."""
    x = torch.randn(2, 6, 64)
    want = torch.native_layer_norm(x, (64,), None, None, 1e-5)
    device_x = x.to(mojo_device)
    for _ in range(25):
        got = torch.native_layer_norm(device_x, (64,), None, None, 1e-5)
        for g, e in zip(got, want, strict=True):
            torch.testing.assert_close(g.cpu(), e, atol=1e-4, rtol=1e-4)


def test_native_group_norm_no_affine_repeats(mojo_device):
    """The group-norm twin of test_native_layer_norm_no_affine_repeats."""
    x = torch.randn(2, 8, 16)
    want = torch.ops.aten.native_group_norm(x, None, None, 2, 8, 16, 4, 1e-5)
    device_x = x.to(mojo_device)
    for _ in range(25):
        got = torch.ops.aten.native_group_norm(device_x, None, None, 2, 8, 16, 4, 1e-5)
        torch.testing.assert_close(got[0].cpu(), want[0], atol=1e-4, rtol=1e-4)


def test_native_layer_norm_noncontiguous_input(mojo_device):
    base = torch.randn(4, 2, 16)
    x = base.transpose(0, 1)
    want = torch.native_layer_norm(x, (16,), None, None, 1e-5)
    got = torch.native_layer_norm(
        base.to(mojo_device).transpose(0, 1), (16,), None, None, 1e-5
    )
    for g, e in zip(got, want, strict=True):
        torch.testing.assert_close(g.cpu(), e, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "mask",
    [
        (True, True, True),
        (True, False, False),
        (False, True, True),
        (False, False, True),
    ],
)
def test_native_layer_norm_backward_output_masks(mojo_gpu, mask):
    x = torch.randn(5, 9)
    w = torch.randn(9)
    b = torch.randn(9)
    grad = torch.randn(5, 9)
    _, mean, rstd = torch.native_layer_norm(x, (9,), w, b, 1e-5)
    want = torch.ops.aten.native_layer_norm_backward(
        grad, x, [9], mean, rstd, w, b, list(mask)
    )
    with ran("aten::native_layer_norm_backward"):
        got = torch.ops.aten.native_layer_norm_backward(
            grad.to(mojo_gpu),
            x.to(mojo_gpu),
            [9],
            mean.to(mojo_gpu),
            rstd.to(mojo_gpu),
            w.to(mojo_gpu),
            b.to(mojo_gpu),
            list(mask),
        )
    for i, wanted in enumerate(mask):
        if wanted:
            torch.testing.assert_close(got[i].cpu(), want[i], atol=1e-4, rtol=1e-4)
        else:
            # There is no way to build an undefined at::Tensor from Mojo, so a
            # gradient autograd did not ask for comes back empty.
            assert got[i] is None  # an undefined Tensor, as ATen returns it


def test_native_layer_norm_backward_empty_rows(mojo_gpu):
    x = torch.randn(0, 9)
    w = torch.randn(9)
    b = torch.randn(9)
    grad = torch.randn(0, 9)
    mean = torch.randn(0, 1)
    rstd = torch.randn(0, 1)
    got = torch.ops.aten.native_layer_norm_backward(
        grad.to(mojo_gpu),
        x.to(mojo_gpu),
        [9],
        mean.to(mojo_gpu),
        rstd.to(mojo_gpu),
        w.to(mojo_gpu),
        b.to(mojo_gpu),
        [True, True, True],
    )
    assert got[0].shape == (0, 9)
    assert got[1].cpu().tolist() == [0.0] * 9
    assert got[2].cpu().tolist() == [0.0] * 9


# ---------------------------------------------------------------------------
# Batch norm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batch_norm_inference(mojo_gpu, call_checker: CallChecker, dtype):
    """A reduced-precision activation with float32 running statistics is what
    `nn.BatchNorm2d` holds under AMP."""
    call_checker.register(aten_functions.aten__native_batch_norm_legit_no_training)
    torch.manual_seed(0)
    x = torch.randn(3, 8, 5, 7, dtype=dtype)
    weight = torch.randn(8, dtype=dtype)
    bias = torch.randn(8, dtype=dtype)
    running_mean = torch.randn(8)
    running_var = torch.rand(8) + 0.5
    args = (weight, bias, running_mean, running_var)
    want = torch.ops.aten._native_batch_norm_legit_no_training(
        x.float(), *(t.float() for t in args), 0.1, 1e-5
    )
    got = torch.ops.aten._native_batch_norm_legit_no_training(
        x.to(mojo_gpu), *(t.to(mojo_gpu) for t in args), 0.1, 1e-5
    )
    tol = 1e-5 if dtype == torch.float32 else 8e-3
    torch.testing.assert_close(got[0].cpu().float(), want[0], atol=tol, rtol=tol)
    # CPU torch leaves the two saved statistics empty here; the CUDA kernel
    # fills them (Normalization.cu) and so does ours, because the autograd
    # formula forwards them into native_batch_norm_backward.
    torch.testing.assert_close(got[1].cpu(), running_mean, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        got[2].cpu(), torch.rsqrt(running_var + 1e-5), atol=1e-5, rtol=1e-5
    )


def test_batch_norm_inference_uniform_dtype(mojo_device, call_checker: CallChecker):
    """Inference batch norm with every parameter in the input's own dtype.

    That is the shape nn_ops' `BatchNormSpec` takes, which is the MAX CPU
    device's only route (its accelerator twin carries the input, the
    statistics and the affine dtypes apart, as `test_batch_norm_inference`
    covers). The two saved statistics come out of a composition there, not
    out of the kernel, so they are checked on both devices.
    """
    call_checker.register(aten_functions.aten_native_batch_norm)
    torch.manual_seed(0)
    x = torch.randn(3, 8, 5, 7)
    weight = torch.randn(8)
    bias = torch.randn(8)
    running_mean = torch.randn(8)
    running_var = torch.rand(8) + 0.5
    args = (weight, bias, running_mean, running_var)
    want = torch.ops.aten.native_batch_norm(x, *args, False, 0.1, 1e-5)
    dev = [t.to(mojo_device) for t in args]
    got = torch.ops.aten.native_batch_norm(x.to(mojo_device), *dev, False, 0.1, 1e-5)
    torch.testing.assert_close(got[0].cpu(), want[0], atol=1e-5, rtol=1e-5)
    # CPU torch returns the two saved statistics empty for an inference batch
    # norm; the CUDA kernel fills them (ATen/native/cuda/Normalization.cu) and
    # so must we, because the autograd formula of the frozen op forwards them
    # into native_batch_norm_backward.
    torch.testing.assert_close(got[1].cpu(), running_mean, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        got[2].cpu(), torch.rsqrt(running_var + 1e-5), atol=1e-5, rtol=1e-5
    )
    # training=False never touches the running statistics.
    torch.testing.assert_close(dev[2].cpu(), running_mean, atol=0, rtol=0)
    torch.testing.assert_close(dev[3].cpu(), running_var, atol=0, rtol=0)


def test_batch_norm_module_inference(mojo_gpu):
    x = torch.randn(2, 64, 14, 14)
    bn = torch.nn.BatchNorm2d(64).eval()
    assert bn.running_mean is not None and bn.running_var is not None
    bn.running_mean.normal_()
    bn.running_var.uniform_(0.5, 2.0)
    bn_dev = torch.nn.BatchNorm2d(64).eval()
    bn_dev.load_state_dict(bn.state_dict())
    bn_dev = bn_dev.to(mojo_gpu)
    with torch.no_grad():
        torch.testing.assert_close(
            bn_dev(x.to(mojo_gpu)).cpu(), bn(x), atol=1e-5, rtol=1e-5
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 8, 5, 7), (2, 5, 13), (4, 3, 8, 8, 2)])
def test_batch_norm_training(mojo_device, dtype, shape):
    """The accelerator runs one fused kernel; the MAX CPU device has none and
    composes the forward through the dispatcher, so both are checked here.

    `ran` rather than the `call_checker` fixture, whose teardown asserts even
    for the parametrizations skipped below."""
    if len(shape) > 4 and mojo_device == f"mojo:{len(list(get_accelerators())) - 1}":
        pytest.skip(
            "rank > 4 training batch norm is accelerator-only: the CPU route "
            "composes through the rank-4 broadcast binary kernels"
        )
    torch.manual_seed(0)
    channels = shape[1]
    x = torch.randn(shape, dtype=dtype)
    weight = torch.randn(channels, dtype=dtype)
    bias = torch.randn(channels, dtype=dtype)
    running_mean = torch.randn(channels)
    running_var = torch.rand(channels) + 0.5

    ref_mean, ref_var = running_mean.clone(), running_var.clone()
    want = torch.ops.aten.native_batch_norm(
        x.float(), weight.float(), bias.float(), ref_mean, ref_var, True, 0.1, 1e-5
    )
    dev_mean = running_mean.to(mojo_device)
    dev_var = running_var.to(mojo_device)
    with ran("aten::native_batch_norm"):
        got = torch.ops.aten.native_batch_norm(
            x.to(mojo_device),
            weight.to(mojo_device),
            bias.to(mojo_device),
            dev_mean,
            dev_var,
            True,
            0.1,
            1e-5,
        )
    tol = 1e-4 if dtype == torch.float32 else 5e-2
    torch.testing.assert_close(got[0].cpu().float(), want[0], atol=tol, rtol=tol)
    torch.testing.assert_close(got[1].cpu(), want[1], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(got[2].cpu(), want[2], atol=1e-4, rtol=1e-4)
    # The running statistics are updated in place, exactly once.
    torch.testing.assert_close(dev_mean.cpu(), ref_mean, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(dev_var.cpu(), ref_var, atol=1e-4, rtol=1e-4)


def test_batch_norm_training_without_running_stats(mojo_device):
    x = torch.randn(4, 6, 3, 3)
    want = torch.ops.aten.native_batch_norm(x, None, None, None, None, True, 0.1, 1e-5)
    got = torch.ops.aten.native_batch_norm(
        x.to(mojo_device), None, None, None, None, True, 0.1, 1e-5
    )
    for g, e in zip(got, want, strict=True):
        torch.testing.assert_close(g.cpu(), e, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Group norm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "c", "hxw", "groups"), [(2, 8, 16, 4), (1, 6, 5, 3), (3, 4, 1, 2)]
)
@pytest.mark.parametrize("affine", [True, False])
def test_native_group_norm(
    mojo_device, call_checker: CallChecker, n, c, hxw, groups, affine
):
    call_checker.register(aten_functions.aten_native_group_norm)
    x = torch.randn(n, c, hxw)
    w = torch.randn(c) if affine else None
    b = torch.randn(c) if affine else None
    want = torch.ops.aten.native_group_norm(x, w, b, n, c, hxw, groups, 1e-5)
    got = torch.ops.aten.native_group_norm(
        x.to(mojo_device),
        None if w is None else w.to(mojo_device),
        None if b is None else b.to(mojo_device),
        n,
        c,
        hxw,
        groups,
        1e-5,
    )
    torch.testing.assert_close(got[0].cpu(), want[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        got[1].cpu().float(), want[1].float(), atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        got[2].cpu().float(), want[2].float(), atol=1e-4, rtol=1e-4
    )


def test_group_norm_module_forward(mojo_device):
    x = torch.randn(2, 8, 4, 4)
    gn = torch.nn.GroupNorm(4, 8).eval()
    with torch.no_grad():
        gn.weight.normal_()
        gn.bias.normal_()
    gn_dev = torch.nn.GroupNorm(4, 8).eval()
    gn_dev.load_state_dict(gn.state_dict())
    gn_dev = gn_dev.to(mojo_device)
    with torch.no_grad():
        torch.testing.assert_close(
            gn_dev(x.to(mojo_device)).cpu(), gn(x), atol=1e-5, rtol=1e-5
        )


# ---------------------------------------------------------------------------
# NLL loss (`out=` ABI)
# ---------------------------------------------------------------------------


def _nll_reference(log_probs, target, reduction, grad_output):
    output = torch.empty((log_probs.shape[0],) if reduction == 0 else ())
    total_weight = torch.empty(())
    torch.ops.aten.nll_loss_forward.output(
        log_probs, target, None, reduction, -1, output=output, total_weight=total_weight
    )
    grad_input = torch.empty_like(log_probs)
    torch.ops.aten.nll_loss_backward.grad_input(
        grad_output,
        log_probs,
        target,
        None,
        reduction,
        -1,
        total_weight,
        grad_input=grad_input,
    )
    return output, total_weight, grad_input


@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_nll_loss_forward_and_backward_out(mojo_gpu, reduction):
    generator = torch.Generator().manual_seed(20260718)
    rows, classes = 17, 13
    log_probs = torch.log_softmax(
        torch.randn(rows, classes, generator=generator), dim=-1
    )
    target = torch.arange(rows, dtype=torch.int64) % classes
    target[::5] = -1
    grad_output = torch.randn((rows,) if reduction == 0 else (), generator=generator)
    want_out, want_tw, want_gi = _nll_reference(
        log_probs, target, reduction, grad_output
    )

    device_log_probs = log_probs.to(mojo_gpu)
    # A strided target exercises the materializing route.
    backing = torch.zeros(rows * 2, dtype=torch.int64)
    backing[::2] = target
    device_target = backing.to(mojo_gpu)[::2]
    device_output = torch.empty((rows,) if reduction == 0 else (), device=mojo_gpu)
    device_total_weight = torch.empty((), device=mojo_gpu)
    with ran("aten::nll_loss_forward.output", "aten::nll_loss_forward"):
        out, tw = torch.ops.aten.nll_loss_forward.output(
            device_log_probs,
            device_target,
            None,
            reduction,
            -1,
            output=device_output,
            total_weight=device_total_weight,
        )
    device_grad_input = torch.empty_like(device_log_probs)
    with ran("aten::nll_loss_backward.grad_input", "aten::nll_loss_backward"):
        gi = torch.ops.aten.nll_loss_backward.grad_input(
            grad_output.to(mojo_gpu),
            device_log_probs,
            device_target,
            None,
            reduction,
            -1,
            device_total_weight,
            grad_input=device_grad_input,
        )
    assert out is device_output
    assert tw is device_total_weight
    assert gi is device_grad_input
    torch.testing.assert_close(device_output.cpu(), want_out)
    torch.testing.assert_close(device_total_weight.cpu(), want_tw)
    torch.testing.assert_close(device_grad_input.cpu(), want_gi)


@pytest.mark.parametrize(("rows", "classes"), [(1, 2), (257, 65), (3, 5000)])
def test_nll_loss_shape_regimes(mojo_gpu, rows, classes):
    generator = torch.Generator().manual_seed(7)
    log_probs = torch.log_softmax(
        torch.randn(rows, classes, generator=generator), dim=-1
    )
    target = torch.arange(rows, dtype=torch.int64) % classes
    grad_output = torch.randn((), generator=generator)
    want_out, want_tw, want_gi = _nll_reference(log_probs, target, 1, grad_output)
    device_output = torch.empty((), device=mojo_gpu)
    device_total_weight = torch.empty((), device=mojo_gpu)
    torch.ops.aten.nll_loss_forward.output(
        log_probs.to(mojo_gpu),
        target.to(mojo_gpu),
        None,
        1,
        -1,
        output=device_output,
        total_weight=device_total_weight,
    )
    device_grad_input = torch.empty(rows, classes, device=mojo_gpu)
    torch.ops.aten.nll_loss_backward.grad_input(
        grad_output.to(mojo_gpu),
        log_probs.to(mojo_gpu),
        target.to(mojo_gpu),
        None,
        1,
        -1,
        device_total_weight,
        grad_input=device_grad_input,
    )
    torch.testing.assert_close(device_output.cpu(), want_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(device_total_weight.cpu(), want_tw)
    torch.testing.assert_close(device_grad_input.cpu(), want_gi, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_nll_loss_all_labels_ignored(mojo_gpu, reduction):
    rows, classes = 4, 6
    log_probs = torch.log_softmax(torch.randn(rows, classes), dim=-1)
    target = torch.full((rows,), -1, dtype=torch.int64)
    grad_output = torch.randn((rows,) if reduction == 0 else ())
    want_out, want_tw, want_gi = _nll_reference(
        log_probs, target, reduction, grad_output
    )
    device_output = torch.empty((rows,) if reduction == 0 else (), device=mojo_gpu)
    device_total_weight = torch.empty((), device=mojo_gpu)
    torch.ops.aten.nll_loss_forward.output(
        log_probs.to(mojo_gpu),
        target.to(mojo_gpu),
        None,
        reduction,
        -1,
        output=device_output,
        total_weight=device_total_weight,
    )
    device_grad_input = torch.empty(rows, classes, device=mojo_gpu)
    torch.ops.aten.nll_loss_backward.grad_input(
        grad_output.to(mojo_gpu),
        log_probs.to(mojo_gpu),
        target.to(mojo_gpu),
        None,
        reduction,
        -1,
        device_total_weight,
        grad_input=device_grad_input,
    )
    torch.testing.assert_close(
        device_output.cpu(), want_out, equal_nan=True, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(device_total_weight.cpu(), want_tw)
    torch.testing.assert_close(device_grad_input.cpu(), want_gi)


def test_nll_loss_functional_variant(mojo_gpu):
    """The functional `nll_loss_forward` torch generates from the `out=` one."""
    log_probs = torch.log_softmax(torch.randn(9, 4), dim=-1)
    target = torch.arange(9, dtype=torch.int64) % 4
    want = torch.ops.aten.nll_loss_forward(log_probs, target, None, 1, -100)
    got = torch.ops.aten.nll_loss_forward(
        log_probs.to(mojo_gpu), target.to(mojo_gpu), None, 1, -100
    )
    torch.testing.assert_close(got[0].cpu(), want[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(got[1].cpu(), want[1])


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index_shape", [(1,), (5,), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_embedding(mojo_device, call_checker: CallChecker, index_shape, dtype):
    call_checker.register(aten_functions.aten_embedding)
    table = torch.randn(10, 7).to(dtype)
    idx = torch.randint(0, 10, index_shape, dtype=torch.int64)
    want = torch.nn.functional.embedding(idx, table)
    got = torch.nn.functional.embedding(idx.to(mojo_device), table.to(mojo_device))
    assert tuple(got.shape) == tuple(want.shape)
    torch.testing.assert_close(got.cpu(), want)


def test_embedding_with_padding_idx(mojo_device):
    table = torch.randn(6, 3)
    idx = torch.tensor([0, 2, 5, 2], dtype=torch.int64)
    torch.testing.assert_close(
        torch.nn.functional.embedding(
            idx.to(mojo_device), table.to(mojo_device), padding_idx=2
        ).cpu(),
        torch.nn.functional.embedding(idx, table, padding_idx=2),
    )


def test_embedding_out_of_range_index_stays_in_bounds(mojo_gpu):
    """A vocabulary overflow must not read device memory outside the table.
    The kernel receives the row count and clamps: the value for an invalid
    index is unspecified, the access is not (there is no portable device
    assert, and a host-visible error flag would synchronize the device on
    every embedding lookup)."""
    table = torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3)
    idx = torch.tensor([0, 6, 1 << 30, -7], dtype=torch.int64)
    got = torch.nn.functional.embedding(idx.to(mojo_gpu), table.to(mojo_gpu)).cpu()
    assert tuple(got.shape) == (4, 3)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got[0], table[0])
    rows = table.tolist()
    for i in (1, 2, 3):
        assert got[i].tolist() in rows


def test_embedding_strided_indices(mojo_device):
    table = torch.randn(8, 5)
    backing = torch.randint(0, 8, (12,), dtype=torch.int64)
    torch.testing.assert_close(
        torch.nn.functional.embedding(
            backing.to(mojo_device)[::3], table.to(mojo_device)
        ).cpu(),
        torch.nn.functional.embedding(backing[::3], table),
    )


@pytest.mark.parametrize("padding_idx", [-1, 0, 3])
def test_embedding_dense_backward(mojo_gpu, padding_idx):
    grad = torch.randn(6, 4)
    idx = torch.tensor([0, 3, 3, 1, 7, 0], dtype=torch.int64)
    want = torch.ops.aten.embedding_dense_backward(grad, idx, 8, padding_idx, False)
    with ran("aten::embedding_dense_backward"):
        got = torch.ops.aten.embedding_dense_backward(
            grad.to(mojo_gpu), idx.to(mojo_gpu), 8, padding_idx, False
        )
    assert tuple(got.shape) == (8, 4)
    torch.testing.assert_close(got.cpu(), want, atol=1e-5, rtol=1e-5)


def test_embedding_dense_backward_declines_scale_grad_by_freq(mojo_gpu):
    grad = torch.randn(3, 2).to(mojo_gpu)
    idx = torch.tensor([0, 1, 1], dtype=torch.int64).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        torch.ops.aten.embedding_dense_backward(grad, idx, 4, -1, True)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kernel", "stride", "padding", "dilation"),
    [(2, None, 0, 1), (3, 2, 1, 1), ((2, 3), (2, 1), (1, 1), 1), (2, 2, 0, 2)],
)
def test_max_pool2d(
    mojo_device, call_checker: CallChecker, kernel, stride, padding, dilation
):
    call_checker.register(aten_functions.aten_max_pool2d_with_indices)
    x = torch.randn(2, 3, 9, 11)
    want, want_idx = torch.nn.functional.max_pool2d(
        x, kernel, stride, padding, dilation, return_indices=True
    )
    got, got_idx = torch.nn.functional.max_pool2d(
        x.to(mojo_device), kernel, stride, padding, dilation, return_indices=True
    )
    torch.testing.assert_close(got.cpu(), want)
    torch.testing.assert_close(got_idx.cpu(), want_idx)


def test_max_pool2d_ceil_mode_declines(mojo_device):
    x = torch.randn(1, 1, 5, 5).to(mojo_device)
    with pytest.raises(NotImplementedError):
        torch.nn.functional.max_pool2d(x, 2, 2, 0, 1, ceil_mode=True)


@pytest.mark.parametrize(
    ("kernel", "stride", "padding", "count_include_pad", "divisor_override"),
    [
        (2, None, 0, True, None),
        (3, 2, 1, True, None),
        (3, 2, 1, False, None),
        ((2, 3), (2, 1), 0, True, 5),
    ],
)
def test_avg_pool2d(
    mojo_device,
    call_checker: CallChecker,
    kernel,
    stride,
    padding,
    count_include_pad,
    divisor_override,
):
    call_checker.register(aten_functions.aten_avg_pool2d)
    x = torch.randn(2, 3, 8, 10)
    want = torch.nn.functional.avg_pool2d(
        x,
        kernel,
        stride,
        padding,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )
    got = torch.nn.functional.avg_pool2d(
        x.to(mojo_device),
        kernel,
        stride,
        padding,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )
    torch.testing.assert_close(got.cpu(), want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("output_size", [(1, 1), (3, 3), (2, 5), (7, 7)])
def test_adaptive_avg_pool2d(mojo_device, call_checker: CallChecker, output_size):
    """Called through the aten op, not `F.adaptive_avg_pool2d`: ATen's
    composite rewrites a (1, 1) output into `mean.dim`, which belongs to the
    reductions group."""
    call_checker.register(aten_functions.aten__adaptive_avg_pool2d)
    x = torch.randn(2, 4, 7, 9)
    torch.testing.assert_close(
        torch.ops.aten._adaptive_avg_pool2d(x.to(mojo_device), list(output_size)).cpu(),
        torch.ops.aten._adaptive_avg_pool2d(x, list(output_size)),
        atol=1e-5,
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# Bilinear upsampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("output_size", [(4, 6), (16, 16), (3, 3)])
def test_upsample_bilinear2d(
    mojo_device, call_checker: CallChecker, align_corners, output_size
):
    call_checker.register(aten_functions.aten_upsample_bilinear2d)
    x = torch.randn(2, 3, 8, 8)
    want = torch.ops.aten.upsample_bilinear2d(x, list(output_size), align_corners)
    got = torch.ops.aten.upsample_bilinear2d(
        x.to(mojo_device), list(output_size), align_corners
    )
    torch.testing.assert_close(got.cpu(), want, atol=1e-5, rtol=1e-5)


def test_upsample_bilinear2d_explicit_scales(mojo_device):
    x = torch.randn(1, 2, 4, 5)
    want = torch.ops.aten.upsample_bilinear2d(x, [8, 10], False, 2.0, 2.0)
    got = torch.ops.aten.upsample_bilinear2d(
        x.to(mojo_device), [8, 10], False, 2.0, 2.0
    )
    torch.testing.assert_close(got.cpu(), want, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# End to end: the two ops F.cross_entropy composes
# ---------------------------------------------------------------------------


def test_cross_entropy_forward(mojo_gpu):
    """`log_softmax` then `nll_loss_forward` through the public functional."""
    logits = torch.randn(12, 30)
    target = torch.arange(12, dtype=torch.int64) % 30
    want = torch.nn.functional.cross_entropy(logits, target)
    with ran("aten::_log_softmax"):
        got = torch.nn.functional.cross_entropy(
            logits.to(mojo_gpu), target.to(mojo_gpu)
        )
    torch.testing.assert_close(got.cpu(), want, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Launch-route regimes.
#
# The shapes above all sit inside one launch route. The counts below straddle
# every route boundary: a scalar head/tail, the register-cached ladder, the
# streaming kernel past it, and the grid the L2 budget stops binding.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cols", [7, 789, 1024, 4096, 20000])
def test_native_layer_norm_row_widths(mojo_gpu, dtype, cols):
    """789 is not a whole number of 16-byte vectors; 1024/4096 are the
    register-cached ladder; 20000 falls to the streaming kernel."""
    x = torch.randn(9, cols).to(dtype)
    w = torch.randn(cols).to(dtype)
    b = torch.randn(cols).to(dtype)
    want = torch.native_layer_norm(x.float(), (cols,), w.float(), b.float(), 1e-5)
    got = torch.native_layer_norm(
        x.to(mojo_gpu), (cols,), w.to(mojo_gpu), b.to(mojo_gpu), 1e-5
    )
    out_tol = 1e-5 if dtype == torch.float32 else 1e-2
    stat_tol = (1e-4, 1e-3) if dtype == torch.float32 else (2e-2, 2e-2)
    torch.testing.assert_close(
        got[0].cpu().float(), want[0].to(dtype).float(), atol=out_tol, rtol=out_tol
    )
    torch.testing.assert_close(
        got[1].cpu().float(),
        want[1].to(dtype).float(),
        atol=stat_tol[0],
        rtol=stat_tol[0],
    )
    torch.testing.assert_close(
        got[2].cpu().float(),
        want[2].to(dtype).float(),
        atol=stat_tol[1],
        rtol=stat_tol[1],
    )


@pytest.mark.parametrize("offset", [1, 2, 3])
def test_native_layer_norm_offset_view_row_phase(mojo_gpu, offset):
    """Whether the cached route may store where it loaded depends on the
    row's phase, which a storage offset changes."""
    flat = torch.randn(5 * 64 + offset)
    w = torch.randn(64)
    b = torch.randn(64)
    x_cpu = flat[offset:].view(5, 64)
    x = flat.to(mojo_gpu)[offset:].view(5, 64)
    want = torch.native_layer_norm(x_cpu, (64,), w, b, 1e-5)
    got = torch.native_layer_norm(x, (64,), w.to(mojo_gpu), b.to(mojo_gpu), 1e-5)
    for g, e in zip(got, want, strict=True):
        torch.testing.assert_close(g.cpu(), e, atol=1e-5, rtol=1e-5)


def test_native_layer_norm_large_mean_needs_the_moment_repass(mojo_gpu):
    """Past the cached ladder the statistics are taken about the row's first
    element; when that element is the outlier, the variance loses most of its
    significand unless the kernel re-reads."""
    cols = 40000
    x = torch.randn(4, cols)
    x[:, 0] = 1e6
    want = torch.native_layer_norm(x.double(), (cols,), None, None, 1e-5)
    got = torch.native_layer_norm(x.to(mojo_gpu), (cols,), None, None, 1e-5)
    torch.testing.assert_close(
        got[2].cpu().double(), want[2], atol=0, rtol=1e-4
    )  # rstd
    torch.testing.assert_close(
        got[0].cpu().double(), want[0], atol=1e-4, rtol=1e-4
    )  # output


def test_native_layer_norm_noncontiguous_weight_and_bias(mojo_gpu):
    """The test above covers a non-contiguous INPUT; the affine parameters
    have their own read path."""
    x_base = torch.randn(3, 2, 4)
    w_base = torch.randn(4, 3)
    b_base = torch.randn(4, 3)
    x_cpu, w_cpu, b_cpu = x_base.transpose(0, 1), w_base.t(), b_base.t()
    x = x_base.to(mojo_gpu).transpose(0, 1)
    w = w_base.to(mojo_gpu).t()
    b = b_base.to(mojo_gpu).t()
    want = torch.native_layer_norm(x_cpu, (3, 4), w_cpu, b_cpu, 1e-5)
    got = torch.native_layer_norm(x, (3, 4), w, b, 1e-5)
    assert got[0].is_contiguous()
    assert tuple(got[1].shape) == (2, 1, 1)
    torch.testing.assert_close(got[0].cpu(), want[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got[1].cpu(), want[1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got[2].cpu(), want[2], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("cols", [768, 769])
def test_native_layer_norm_nonfinite_rows(mojo_gpu, cols, value):
    """A row that is entirely non-finite: output, mean AND rstd are all NaN."""
    x = torch.full((2, cols), value)
    got = torch.native_layer_norm(x.to(mojo_gpu), (cols,), None, None, 1e-5)
    for part in got:
        assert bool(part.cpu().isnan().all()), part


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("n", "c", "hxw", "groups"), [(2, 8, 5, 4), (3, 12, 49, 3), (2, 6, 4096, 2)]
)
def test_native_group_norm_spatial_regimes(mojo_gpu, dtype, n, c, hxw, groups):
    """hxw = 4096 is a different launch regime from the <= 16 cases above."""
    x = torch.randn(n, c, hxw).to(dtype)
    w = torch.randn(c).to(dtype)
    b = torch.randn(c).to(dtype)
    want = torch.ops.aten.native_group_norm(
        x.float(), w.float(), b.float(), n, c, hxw, groups, 1e-5
    )
    got = torch.ops.aten.native_group_norm(
        x.to(mojo_gpu), w.to(mojo_gpu), b.to(mojo_gpu), n, c, hxw, groups, 1e-5
    )
    out_tol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(
        got[0].cpu().float(), want[0].to(dtype).float(), atol=out_tol, rtol=out_tol
    )
    torch.testing.assert_close(
        got[1].cpu().float(), want[1].float(), atol=1e-3, rtol=1e-3
    )


@pytest.mark.parametrize("channels", [3, 200])
@pytest.mark.parametrize("magnitude", [1e6, -1e6])
def test_native_batch_norm_training_outlier_first_element(
    mojo_gpu, channels, magnitude
):
    """Element (0, c, 0) is the assumed mean, i.e. the worst available shift.
    3 channels take the split workspace + merge; 200 take the fused
    one-block-per-channel path that re-scans in place."""
    x = torch.randn(6, channels, 4, 9)
    x[0, :, 0, 0] = magnitude
    running_mean_cpu = torch.zeros(channels, dtype=torch.float64)
    running_var_cpu = torch.ones(channels, dtype=torch.float64)
    want = torch.native_batch_norm(
        x.double(), None, None, running_mean_cpu, running_var_cpu, True, 0.1, 1e-5
    )
    running_mean = torch.zeros(channels).to(mojo_gpu)
    running_var = torch.ones(channels).to(mojo_gpu)
    got = torch.native_batch_norm(
        x.to(mojo_gpu), None, None, running_mean, running_var, True, 0.1, 1e-5
    )
    torch.testing.assert_close(
        got[1].cpu().double(), want[1], atol=0, rtol=1e-5
    )  # save_mean
    torch.testing.assert_close(
        got[2].cpu().double(), want[2], atol=0, rtol=1e-5
    )  # save_invstd
    torch.testing.assert_close(
        running_var.cpu().double(), running_var_cpu, atol=0, rtol=1e-5
    )


# ---------------------------------------------------------------------------
# log_softmax: wide rows, non-finite rows, and the backward's own regimes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cols", [5, 128, 257, 1023, 1024, 2047, 4096])
def test_log_softmax_narrow_rows_match_cpu(mojo_gpu, dtype, cols):
    """Rows narrower than one block pass leave threads with no elements;
    their -inf running max must not NaN the sum."""
    torch.manual_seed(cols)
    x = torch.randn(8, cols).to(dtype)
    got = torch.log_softmax(x.to(mojo_gpu), dim=-1)
    # bfloat16 outputs: the kernel and CPU torch round a float32 result to
    # bfloat16 from slightly different sums, up to a few bf16 ulps apart
    tol = {"rtol": 2e-2, "atol": 1e-2} if dtype == torch.bfloat16 else {}
    torch.testing.assert_close(got.cpu(), torch.log_softmax(x, dim=-1), **tol)


def test_log_softmax_positive_inf_rows_match_cpu(mojo_gpu):
    """torch's denominator is NaN here, so the whole row is NaN. Guarding
    `exp(a-b)` with an `a == b` select turns `exp(inf-inf)` into 1.0 and
    returns -inf for the finite entries instead."""
    x = torch.randn(4, 512)
    x[:, 7] = float("inf")
    got = torch.log_softmax(x.to(mojo_gpu), dim=-1)
    torch.testing.assert_close(got.cpu(), torch.log_softmax(x, dim=-1), equal_nan=True)


@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "cols", [12_501, 16_384, 16_385, 25_000, 33_000, 50_304, 70_001]
)
def test_log_softmax_wide_rows_match_cpu(mojo_gpu, cols, dtype, offset):
    """These counts straddle the point where the L2 budget stops binding the
    grid. The offset makes each row base unaligned, so the per-row scalar
    head and tail run too."""
    flat = torch.randn(3 * cols + offset).to(dtype)
    x_cpu = flat[offset:].view(3, cols)
    x = flat.to(mojo_gpu)[offset:].view(3, cols)
    got = torch.log_softmax(x, dim=-1).cpu()
    assert not bool(got.isnan().any())
    tol = 2e-5 if dtype == torch.float32 else 3e-2
    torch.testing.assert_close(
        got.float(), torch.log_softmax(x_cpu.float(), dim=-1), atol=tol, rtol=tol
    )


@pytest.mark.parametrize("cols", [16_385, 50_304])
def test_log_softmax_wide_rows_stay_finite(mojo_gpu, cols):
    """A block max seeded with Float32.MIN (= -inf) turns an idle thread's
    `0 * exp(m-m)` into a NaN the block reduction spreads over the row."""
    spiked = torch.full((1, cols), -3.0)
    spiked[0, cols // 3] = 60.0
    got = torch.log_softmax(spiked.to(mojo_gpu), dim=-1).cpu()
    assert not bool(got.isnan().any())
    torch.testing.assert_close(
        got, torch.log_softmax(spiked, dim=-1), atol=3e-2, rtol=3e-2
    )

    constant = torch.full((1, cols), 0.25)
    got_constant = torch.log_softmax(constant.to(mojo_gpu), dim=-1).cpu()
    torch.testing.assert_close(
        got_constant, torch.full((1, cols), -math.log(cols)), atol=3e-2, rtol=3e-2
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape,dim", [((33, 257), -1), ((16, 4096), -1), ((2, 3, 129), 2)]
)
def test_log_softmax_backward_regimes(mojo_gpu, shape, dim, dtype):
    """Odd cols exercise the head/tail; 4096 is the pure aligned body; the
    rank-3 case reduces over a non-final-but-trailing dim."""
    out = torch.log_softmax(torch.randn(shape), dim=dim).to(dtype)
    grad = torch.randn(shape).to(dtype)
    expected = (
        grad.float() - out.float().exp() * grad.float().sum(dim, keepdim=True)
    ).to(dtype)
    got = torch.ops.aten._log_softmax_backward_data(
        grad.to(mojo_gpu), out.to(mojo_gpu), dim, dtype
    )
    tol = 2e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(got.cpu().float(), expected.float(), atol=tol, rtol=tol)


@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cols", [3, 33_000, 50_304, 70_001])
def test_log_softmax_backward_wide_rows(mojo_gpu, cols, dtype, offset):
    """cols = 3 is below one 16-byte vector, so the whole row is head/tail
    scalars; 70001 reaches the re-read fallback."""
    out_flat = torch.randn(2 * cols + offset).to(dtype)
    grad_flat = torch.randn(2 * cols + offset).to(dtype)
    out_cpu = torch.log_softmax(out_flat[offset:].view(2, cols).float(), dim=-1).to(
        dtype
    )
    grad_cpu = grad_flat[offset:].view(2, cols)
    expected = (
        grad_cpu.float()
        - out_cpu.float().exp() * grad_cpu.float().sum(-1, keepdim=True)
    ).to(dtype)
    got = torch.ops.aten._log_softmax_backward_data(
        grad_flat.to(mojo_gpu)[offset:].view(2, cols), out_cpu.to(mojo_gpu), -1, dtype
    )
    tol = 2e-5 if dtype == torch.float32 else 3e-2
    torch.testing.assert_close(got.cpu().float(), expected.float(), atol=tol, rtol=tol)


def test_log_softmax_backward_non_trailing_dim(mojo_gpu):
    out = torch.log_softmax(torch.randn(6, 33, 5), dim=1)
    grad = torch.randn(6, 33, 5)
    expected = torch.ops.aten._log_softmax_backward_data(grad, out, 1, torch.float32)
    got = torch.ops.aten._log_softmax_backward_data(
        grad.to(mojo_gpu), out.to(mojo_gpu), 1, torch.float32
    )
    torch.testing.assert_close(got.cpu(), expected, atol=2e-5, rtol=2e-5)


@pytest.mark.xfail(
    strict=True,
    reason="the native op declines an input_dtype different from the "
    "gradient's ('_log_softmax_backward_data with an input_dtype different "
    "from the gradient's'); the old eager path served it. AOTAutograd emits "
    "exactly this call for an autocast log_softmax, so it is a real gap, not "
    "an exotic overload. Drop the marker when the op accepts it.",
)
def test_log_softmax_backward_promotes_to_the_requested_dtype(mojo_gpu):
    """`input_dtype=float16` with float32 operands: the CPU op REJECTS this
    promotion, so the reference is built by hand."""
    out = torch.log_softmax(torch.randn(4, 7), dim=-1)
    grad = torch.randn(4, 7)
    expected = (grad - out.exp() * grad.sum(-1, keepdim=True)).to(torch.float16)
    got = torch.ops.aten._log_softmax_backward_data(
        grad.to(mojo_gpu), out.to(mojo_gpu), -1, torch.float16
    )
    assert got.dtype == torch.float16
    torch.testing.assert_close(got.cpu(), expected, atol=2e-3, rtol=2e-3)
