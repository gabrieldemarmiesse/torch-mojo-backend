"""Native-backend pooling group: max / average / adaptive pooling in 2-D and
3-D with their backwards, max unpooling, and im2col / col2im (F.unfold /
F.fold).

Every case runs through the public torch API on the mojo device and is
compared against the same call on CPU torch. Backwards run through autograd
(the input's `.grad`), which reaches the aten backward ops natively.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.native.conftest import ran, skip_if_metal
from torch_mojo_backend import aten_functions
from torch_mojo_backend.testing import CallChecker

FLOAT_DTYPES = [torch.float32, torch.bfloat16, torch.float16, torch.float64]


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    """(atol, rtol) for a dtype."""
    if dtype == torch.float64:
        return (1e-10, 1e-10)
    if dtype == torch.float32:
        return (1e-5, 1e-5)
    if dtype == torch.bfloat16:
        return (1.6e-2, 1.6e-2)
    return (2e-3, 2e-3)


def _skip_f64(device: str, dtype: torch.dtype):
    if dtype == torch.float64:
        skip_if_metal(device, "Apple GPUs have no float64")


def _close(got: torch.Tensor, want: torch.Tensor, dtype: torch.dtype):
    atol, rtol = _tol(dtype)
    assert got.dtype == want.dtype
    torch.testing.assert_close(got.cpu(), want, atol=atol, rtol=rtol, equal_nan=True)


def _grad_pair(fn, x: torch.Tensor, device: str, seed: int = 0):
    """(input grad on CPU, input grad on the device) of sum(fn(x) * g)."""
    torch.manual_seed(seed)
    xc = x.clone().requires_grad_(True)
    out = fn(xc)
    out = out[0] if isinstance(out, tuple) else out
    g = torch.randn(out.shape).to(x.dtype)
    out.backward(g)
    xd = x.to(device).requires_grad_(True)
    outd = fn(xd)
    outd = outd[0] if isinstance(outd, tuple) else outd
    outd.backward(g.to(device))
    assert xc.grad is not None and xd.grad is not None
    return xc.grad, xd.grad


# ---------------------------------------------------------------------------
# Max pooling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kernel", "stride", "padding", "dilation", "ceil_mode"),
    [
        (2, None, 0, 1, False),
        (3, 2, 1, 1, False),
        ((2, 3), (2, 1), (1, 1), 1, False),
        (2, 2, 0, 2, False),
        (3, 2, 1, 1, True),
        (2, 3, 1, (2, 1), True),
    ],
)
def test_max_pool2d(
    mojo_device, call_checker: CallChecker, kernel, stride, padding, dilation, ceil_mode
):
    call_checker.register(aten_functions.aten_max_pool2d_with_indices)
    x = torch.randn(2, 3, 9, 11)
    args = (kernel, stride, padding, dilation)
    want, want_idx = F.max_pool2d(x, *args, ceil_mode=ceil_mode, return_indices=True)
    got, got_idx = F.max_pool2d(
        x.to(mojo_device), *args, ceil_mode=ceil_mode, return_indices=True
    )
    torch.testing.assert_close(got.cpu(), want)
    torch.testing.assert_close(got_idx.cpu(), want_idx)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_pool2d_dtypes_layouts(mojo_device, dtype):
    """Every float dtype; an unbatched (C, H, W) input; a channels_last one."""
    _skip_f64(mojo_device, dtype)
    x = torch.randn(2, 5, 12, 7).to(dtype)
    for inp in (x, x[0], x.to(memory_format=torch.channels_last)):
        want, want_idx = F.max_pool2d(inp, 3, 2, 1, return_indices=True)
        got, got_idx = F.max_pool2d(inp.to(mojo_device), 3, 2, 1, return_indices=True)
        _close(got, want, dtype)
        torch.testing.assert_close(got_idx.cpu(), want_idx)


def test_max_pool_nan_wins(mojo_device):
    """CUDA's `val > max || isnan(val)`: a NaN in the window is the max, and
    the last NaN's index is the one kept; -inf-only windows keep their first
    element's index."""
    x = torch.randn(1, 1, 6, 6)
    x[0, 0, 0, 1] = float("nan")
    x[0, 0, 1, 0] = float("nan")
    x[0, 0, 4:6, 4:6] = float("-inf")
    want, want_idx = F.max_pool2d(x, 2, return_indices=True)
    got, got_idx = F.max_pool2d(x.to(mojo_device), 2, return_indices=True)
    torch.testing.assert_close(got.cpu(), want, equal_nan=True)
    torch.testing.assert_close(got_idx.cpu(), want_idx)
    x3 = torch.randn(1, 2, 4, 4, 4)
    x3[0, 1, 1, 2, 3] = float("nan")
    want, want_idx = F.max_pool3d(x3, 2, return_indices=True)
    got, got_idx = F.max_pool3d(x3.to(mojo_device), 2, return_indices=True)
    torch.testing.assert_close(got.cpu(), want, equal_nan=True)
    torch.testing.assert_close(got_idx.cpu(), want_idx)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    ("kernel", "stride", "padding", "dilation", "ceil_mode"),
    [
        (2, None, 0, 1, False),
        ((3, 2, 2), (2, 1, 2), (1, 0, 1), 1, True),
        (2, 1, 0, (1, 2, 2), False),
    ],
)
def test_max_pool3d(mojo_device, dtype, kernel, stride, padding, dilation, ceil_mode):
    _skip_f64(mojo_device, dtype)
    x = torch.randn(2, 3, 5, 7, 6).to(dtype)
    args = (kernel, stride, padding, dilation)
    with ran("aten::max_pool3d_with_indices"):
        got, got_idx = F.max_pool3d(
            x.to(mojo_device), *args, ceil_mode=ceil_mode, return_indices=True
        )
    want, want_idx = F.max_pool3d(
        x.float(), *args, ceil_mode=ceil_mode, return_indices=True
    )
    _close(got, want.to(dtype), dtype)
    torch.testing.assert_close(got_idx.cpu(), want_idx)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("ceil_mode", [False, True])
def test_max_pool_backward(mojo_device, dtype, ceil_mode):
    """Overlapping windows (stride < kernel) so one input collects several
    outputs' gradients."""
    x = torch.randn(2, 3, 9, 10).to(dtype)
    with ran("aten::max_pool2d_with_indices_backward"):
        want, got = _grad_pair(
            lambda t: F.max_pool2d(t, 3, 2, 1, (1, 2), ceil_mode=ceil_mode),
            x,
            mojo_device,
        )
    _close(got, want, dtype)
    x3 = torch.randn(2, 2, 5, 6, 7).to(torch.float32)
    with ran("aten::max_pool3d_with_indices_backward"):
        want, got = _grad_pair(
            lambda t: F.max_pool3d(t, 3, 2, 1, ceil_mode=ceil_mode), x3, mojo_device
        )
    _close(got, want, torch.float32)


def test_max_pool_out_variants(mojo_device):
    x = torch.randn(2, 3, 8, 8)
    want, want_idx = torch.ops.aten.max_pool2d_with_indices(x, [3], [2], [1])
    out = torch.empty(0).to(mojo_device)
    idx = torch.empty(0, dtype=torch.int64).to(mojo_device)
    torch.ops.aten.max_pool2d_with_indices.out(
        x.to(mojo_device), [3], [2], [1], out=out, indices=idx
    )
    torch.testing.assert_close(out.cpu(), want)
    torch.testing.assert_close(idx.cpu(), want_idx)
    g = torch.randn(want.shape)
    gin_want = torch.ops.aten.max_pool2d_with_indices_backward(
        g, x, [3], [2], [1], [1], False, want_idx
    )
    gin = torch.empty(0).to(mojo_device)
    torch.ops.aten.max_pool2d_with_indices_backward.grad_input(
        g.to(mojo_device),
        x.to(mojo_device),
        [3],
        [2],
        [1],
        [1],
        False,
        idx,
        grad_input=gin,
    )
    torch.testing.assert_close(gin.cpu(), gin_want)


def test_max_pool_empty_batch(mojo_device):
    x = torch.randn(0, 3, 6, 6)
    got, idx = F.max_pool2d(x.to(mojo_device), 2, return_indices=True)
    assert tuple(got.shape) == (0, 3, 3, 3) and tuple(idx.shape) == (0, 3, 3, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kernel_size": 2, "padding": 2},
        {"kernel_size": 2, "padding": -1},
        {"kernel_size": 2, "stride": 0},
        {"kernel_size": 7},
    ],
)
def test_max_pool_rejects_bad_arguments(mojo_device, kwargs):
    x = torch.randn(1, 2, 5, 5).to(mojo_device)
    with pytest.raises(RuntimeError) as info:
        F.max_pool2d(x, return_indices=True, **kwargs)
    assert not isinstance(info.value, NotImplementedError)


def test_pool_declines_int64(mojo_device):
    """CUDA has no integer pooling kernel (AT_DISPATCH_FLOATING_TYPES)."""
    x = torch.arange(16).reshape(1, 1, 4, 4).to(mojo_device)
    with pytest.raises(NotImplementedError):
        F.max_pool2d(x, 2)


# ---------------------------------------------------------------------------
# Average pooling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kernel", "stride", "padding", "ceil_mode", "count_include_pad", "divisor"),
    [
        (2, None, 0, False, True, None),
        (3, 2, 1, False, True, None),
        (3, 2, 1, False, False, None),
        ((2, 3), (2, 1), 0, False, True, 5),
        (3, 2, 1, True, True, None),
        (3, 2, 1, True, False, None),
        (4, 3, 2, True, False, 3),
    ],
)
def test_avg_pool2d(
    mojo_device,
    call_checker: CallChecker,
    kernel,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor,
):
    call_checker.register(aten_functions.aten_avg_pool2d)
    x = torch.randn(2, 3, 8, 10)
    kwargs = {
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor,
    }
    want = F.avg_pool2d(x, kernel, stride, padding, **kwargs)
    got = F.avg_pool2d(x.to(mojo_device), kernel, stride, padding, **kwargs)
    torch.testing.assert_close(got.cpu(), want, atol=1e-5, rtol=1e-5)
    want_g, got_g = _grad_pair(
        lambda t: F.avg_pool2d(t, kernel, stride, padding, **kwargs), x, mojo_device
    )
    torch.testing.assert_close(got_g.cpu(), want_g, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_avg_pool_dtypes_unbatched_1d(mojo_device, dtype):
    _skip_f64(mojo_device, dtype)
    x = torch.randn(3, 9, 11).to(dtype)
    _close(
        F.avg_pool2d(x.to(mojo_device), 3, 2, 1, ceil_mode=True),
        F.avg_pool2d(x.double(), 3, 2, 1, ceil_mode=True).to(dtype),
        dtype,
    )
    _close(
        F.avg_pool1d(x.to(mojo_device), 4, 3, 2, ceil_mode=True),
        F.avg_pool1d(x.double(), 4, 3, 2, ceil_mode=True).to(dtype),
        dtype,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("kernel", "stride", "padding", "ceil_mode", "count_include_pad", "divisor"),
    [
        (2, None, 0, False, True, None),
        ((3, 2, 3), (2, 1, 2), (1, 0, 1), True, False, None),
        (3, 2, 1, False, True, 7),
    ],
)
def test_avg_pool3d(
    mojo_device, dtype, kernel, stride, padding, ceil_mode, count_include_pad, divisor
):
    x = torch.randn(2, 3, 6, 7, 8).to(dtype)
    kwargs = {
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor,
    }
    with ran("aten::avg_pool3d"):
        got = F.avg_pool3d(x.to(mojo_device), kernel, stride, padding, **kwargs)
    want = F.avg_pool3d(x.double(), kernel, stride, padding, **kwargs).to(dtype)
    _close(got, want, dtype)
    with ran("aten::avg_pool3d_backward"):
        want_g, got_g = _grad_pair(
            lambda t: F.avg_pool3d(t, kernel, stride, padding, **kwargs),
            x.float(),
            mojo_device,
        )
    _close(got_g, want_g, torch.float32)


def test_avg_pool_out_variants(mojo_device):
    x = torch.randn(2, 3, 7, 7)
    want = F.avg_pool2d(x, 3, 2, 1)
    out = torch.empty(5).to(mojo_device)
    torch.ops.aten.avg_pool2d.out(x.to(mojo_device), [3], [2], [1], out=out)
    torch.testing.assert_close(out.cpu(), want)
    x3 = torch.randn(1, 2, 4, 5, 6)
    want3 = F.avg_pool3d(x3, 2)
    out3 = torch.empty(0).to(mojo_device)
    torch.ops.aten.avg_pool3d.out(x3.to(mojo_device), [2], out=out3)
    torch.testing.assert_close(out3.cpu(), want3)


def test_avg_pool_rejects_bad_arguments(mojo_device):
    x = torch.randn(1, 2, 5, 5).to(mojo_device)
    with pytest.raises(RuntimeError, match="divisor must be not zero"):
        F.avg_pool2d(x, 2, divisor_override=0)
    with pytest.raises(RuntimeError, match="pad should be at most half"):
        F.avg_pool2d(x, 2, padding=2)
    with pytest.raises(RuntimeError, match="smaller than kernel size"):
        F.avg_pool3d(torch.randn(1, 1, 2, 5, 5).to(mojo_device), 3, padding=1)


# ---------------------------------------------------------------------------
# Adaptive pooling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("output_size", [(1, 1), (3, 3), (2, 5), (7, 7), (4, 13)])
def test_adaptive_avg_pool2d(mojo_device, call_checker: CallChecker, output_size):
    """Called through the aten op, not `F.adaptive_avg_pool2d`: ATen's
    composite rewrites a (1, 1) output into `mean.dim`."""
    call_checker.register(aten_functions.aten__adaptive_avg_pool2d)
    x = torch.randn(2, 4, 7, 9)
    torch.testing.assert_close(
        torch.ops.aten._adaptive_avg_pool2d(x.to(mojo_device), list(output_size)).cpu(),
        torch.ops.aten._adaptive_avg_pool2d(x, list(output_size)),
        atol=1e-5,
        rtol=1e-5,
    )
    with ran("aten::_adaptive_avg_pool2d_backward"):
        want, got = _grad_pair(
            lambda t: torch.ops.aten._adaptive_avg_pool2d(t, list(output_size)),
            x,
            mojo_device,
        )
    torch.testing.assert_close(got.cpu(), want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("output_size", [(2, 3, 4), (5, 1, 7), (6, 7, 8)])
def test_adaptive_avg_pool3d(mojo_device, dtype, output_size):
    _skip_f64(mojo_device, dtype)
    x = torch.randn(2, 3, 5, 6, 7).to(dtype)
    with ran("aten::_adaptive_avg_pool3d"):
        got = torch.ops.aten._adaptive_avg_pool3d(x.to(mojo_device), list(output_size))
    want = torch.ops.aten._adaptive_avg_pool3d(x.double(), list(output_size))
    _close(got, want.to(dtype), dtype)
    with ran("aten::_adaptive_avg_pool3d_backward"):
        want_g, got_g = _grad_pair(
            lambda t: torch.ops.aten._adaptive_avg_pool3d(t, list(output_size)),
            x.float(),
            mojo_device,
        )
    _close(got_g, want_g, torch.float32)
    out = torch.empty(0, dtype=dtype).to(mojo_device)
    torch.ops.aten.adaptive_avg_pool3d.out(x.to(mojo_device), list(output_size), out=out)
    _close(out, want.to(dtype), dtype)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("output_size", [(1, 1), (3, 4), (5, 9)])
def test_adaptive_max_pool2d(mojo_device, dtype, output_size):
    _skip_f64(mojo_device, dtype)
    x = torch.randn(2, 3, 7, 9).to(dtype)
    with ran("aten::adaptive_max_pool2d"):
        got, got_idx = F.adaptive_max_pool2d(
            x.to(mojo_device), output_size, return_indices=True
        )
    want, want_idx = F.adaptive_max_pool2d(x.float(), output_size, return_indices=True)
    _close(got, want.to(dtype), dtype)
    torch.testing.assert_close(got_idx.cpu(), want_idx)
    with ran("aten::adaptive_max_pool2d_backward"):
        want_g, got_g = _grad_pair(
            lambda t: F.adaptive_max_pool2d(t, output_size), x.float(), mojo_device
        )
    _close(got_g, want_g, torch.float32)


@pytest.mark.parametrize("output_size", [(1, 1, 1), (2, 3, 4), (5, 6, 7)])
def test_adaptive_max_pool3d(mojo_device, output_size):
    x = torch.randn(2, 3, 5, 6, 7)
    x[0, 1, 2, 3, 4] = float("nan")
    with ran("aten::adaptive_max_pool3d"):
        got, got_idx = F.adaptive_max_pool3d(
            x.to(mojo_device), output_size, return_indices=True
        )
    want, want_idx = F.adaptive_max_pool3d(x, output_size, return_indices=True)
    torch.testing.assert_close(got.cpu(), want, equal_nan=True)
    torch.testing.assert_close(got_idx.cpu(), want_idx)
    x = torch.randn(3, 5, 6, 7)  # unbatched
    with ran("aten::adaptive_max_pool3d_backward"):
        want_g, got_g = _grad_pair(
            lambda t: F.adaptive_max_pool3d(t, output_size), x, mojo_device
        )
    torch.testing.assert_close(got_g.cpu(), want_g)


def test_interpolate_area(mojo_device):
    x = torch.randn(2, 3, 11, 13)
    torch.testing.assert_close(
        F.interpolate(x.to(mojo_device), size=(4, 5), mode="area").cpu(),
        F.interpolate(x, size=(4, 5), mode="area"),
    )
    x1 = torch.randn(2, 3, 17)
    torch.testing.assert_close(
        F.interpolate(x1.to(mojo_device), size=6, mode="area").cpu(),
        F.interpolate(x1, size=6, mode="area"),
    )


# ---------------------------------------------------------------------------
# Max unpooling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_unpool2d(mojo_device, dtype):
    _skip_f64(mojo_device, dtype)
    x = torch.randn(2, 3, 9, 8)
    pooled, idx = F.max_pool2d(x, 3, 2, 1, return_indices=True)
    pooled = pooled.to(dtype)
    want = F.max_unpool2d(pooled.float(), idx, (3, 3), (2, 2), (1, 1), (9, 8))
    with ran("aten::max_unpool2d"):
        got = F.max_unpool2d(
            pooled.to(mojo_device), idx.to(mojo_device), (3, 3), (2, 2), (1, 1), (9, 8)
        )
    _close(got, want.to(dtype), dtype)
    # Unbatched, and the backward (a gather at the indices).
    want_g, got_g = _grad_pair(
        lambda t: F.max_unpool2d(t, idx[0].to(t.device), (3, 3), (2, 2), (1, 1), (9, 8)),
        pooled[0].float(),
        mojo_device,
    )
    torch.testing.assert_close(got_g.cpu(), want_g)


def test_max_unpool1d_3d(mojo_device):
    x = torch.randn(2, 3, 17)
    p, i = F.max_pool1d(x, 2, return_indices=True)
    torch.testing.assert_close(
        F.max_unpool1d(p.to(mojo_device), i.to(mojo_device), (2,)).cpu(),
        F.max_unpool1d(p, i, (2,)),
    )
    x3 = torch.randn(2, 2, 6, 5, 7)
    p3, i3 = F.max_pool3d(x3, 2, return_indices=True)
    with ran("aten::max_unpool3d"):
        got = F.max_unpool3d(
            p3.to(mojo_device), i3.to(mojo_device), (2, 2, 2), None, (0, 0, 0), (6, 5, 7)
        )
    torch.testing.assert_close(got.cpu(), F.max_unpool3d(p3, i3, (2, 2, 2), None, (0, 0, 0), (6, 5, 7)))
    out = torch.empty(0).to(mojo_device)
    torch.ops.aten.max_unpool2d.out(
        p.unsqueeze(2).to(mojo_device), i.unsqueeze(2).to(mojo_device), [1, 16], out=out
    )
    torch.testing.assert_close(
        out.cpu(), torch.ops.aten.max_unpool2d(p.unsqueeze(2), i.unsqueeze(2), [1, 16])
    )


def test_max_unpool_rejects_bad_indices_shape(mojo_device):
    x = torch.randn(1, 1, 4, 4).to(mojo_device)
    i = torch.zeros(1, 1, 4, 1, dtype=torch.int64).to(mojo_device)
    with pytest.raises(RuntimeError, match="Expected shape of indices"):
        F.max_unpool2d(x, i, (3, 3), (2, 2))


# ---------------------------------------------------------------------------
# im2col / col2im (F.unfold / F.fold)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [*FLOAT_DTYPES, torch.bool])
@pytest.mark.parametrize(
    ("kernel", "dilation", "padding", "stride"),
    [(2, 1, 0, 1), ((3, 2), (1, 2), (1, 0), (2, 1)), (3, 2, 2, 3)],
)
def test_unfold_fold(mojo_device, dtype, kernel, dilation, padding, stride):
    _skip_f64(mojo_device, dtype)
    x = torch.randn(2, 3, 9, 10)
    x = (x > 0) if dtype == torch.bool else x.to(dtype)
    args = (kernel, dilation, padding, stride)
    with ran("aten::im2col"):
        got = F.unfold(x.to(mojo_device), *args)
    want = F.unfold(x, *args)
    torch.testing.assert_close(got.cpu(), want)
    # Unbatched input keeps no batch dim.
    torch.testing.assert_close(F.unfold(x[0].to(mojo_device), *args).cpu(), F.unfold(x[0], *args))
    with ran("aten::col2im"):
        folded = F.fold(want.to(mojo_device), (9, 10), *args)
    if dtype == torch.bool:
        # CPU folds bool as a sum cast to bool: "any".
        torch.testing.assert_close(folded.cpu(), F.fold(want.float(), (9, 10), *args) != 0)
    else:
        _close(folded, F.fold(want.double(), (9, 10), *args).to(dtype), dtype)


def test_unfold_backward_is_fold(mojo_device):
    x = torch.randn(2, 3, 8, 7)
    want, got = _grad_pair(lambda t: F.unfold(t, 3, 1, 1, 2), x, mojo_device)
    torch.testing.assert_close(got.cpu(), want)


def test_fold_out_variants_and_errors(mojo_device):
    cols = torch.randn(2, 12, 16)
    out = torch.empty(0).to(mojo_device)
    torch.ops.aten.col2im.out(cols.to(mojo_device), [5, 5], [2, 2], [1, 1], [0, 0], [1, 1], out=out)
    torch.testing.assert_close(
        out.cpu(), torch.ops.aten.col2im(cols, [5, 5], [2, 2], [1, 1], [0, 0], [1, 1])
    )
    with pytest.raises(RuntimeError, match="divisible by the product of kernel_size"):
        F.fold(torch.randn(2, 10, 16).to(mojo_device), (5, 5), 2)
    with pytest.raises(RuntimeError, match="sliding blocks"):
        F.fold(torch.randn(2, 12, 15).to(mojo_device), (5, 5), 2)
    with pytest.raises(RuntimeError, match="must be at least one"):
        F.unfold(torch.randn(1, 1, 2, 2).to(mojo_device), 3)
