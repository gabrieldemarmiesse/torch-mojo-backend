"""Native-backend pooling group: max / average / adaptive pooling in 2-D and
3-D with their backwards, max unpooling, and im2col / col2im (F.unfold /
F.fold).

Every case runs through the public torch API on the mojo device and is
compared against the same call on CPU torch. Backwards run through autograd
(the input's `.grad`), which reaches the aten backward ops natively.
"""

import itertools
import math

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
    torch.ops.aten.adaptive_avg_pool3d.out(
        x.to(mojo_device), list(output_size), out=out
    )
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
# Fractional max pooling and the indices-free max_pool2d_backward
# ---------------------------------------------------------------------------


def _frac(n: int):
    return F.fractional_max_pool2d if n == 2 else F.fractional_max_pool3d


def _frac_reference(n: int, x: torch.Tensor, k, out, samples: torch.Tensor):
    """CPU torch's result, in float32 for the half types: CPU builds the
    window sequence in the dtype itself (half arithmetic), CUDA -- and this
    backend -- in float (`acc_type`); the max of half values is exact in
    float32."""
    if x.dtype in (torch.float16, torch.bfloat16):
        y, idx = _frac(n)(
            x.float(),
            k,
            output_size=out,
            return_indices=True,
            _random_samples=samples.float(),
        )
        return y.to(x.dtype), idx
    return _frac(n)(x, k, output_size=out, return_indices=True, _random_samples=samples)


_FRAC_CASES = [
    # (n, input shape, kernel, output size): batched and unbatched, a window
    # as large as the input, an output of one, awkward sizes.
    (2, (2, 3, 11, 9), (3, 2), (5, 6)),
    (2, (4, 10, 10), 2, (7, 1)),
    (2, (1, 2, 7, 7), 7, 1),
    (2, (3, 5, 37, 23), (4, 3), (13, 17)),
    (3, (2, 2, 9, 8, 7), (2, 3, 2), (4, 3, 5)),
    (3, (3, 6, 6, 6), 3, (3, 2, 1)),
]


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(("n", "shape", "kernel", "out"), _FRAC_CASES)
def test_fractional_max_pool(mojo_device, dtype, n, shape, kernel, out):
    _skip_f64(mojo_device, dtype)
    torch.manual_seed(0)
    x = torch.randn(shape).to(dtype)
    batch = shape[0] if len(shape) == n + 2 else 1
    channels = shape[-n - 1]
    samples = torch.rand(batch, channels, n).to(dtype)
    want, want_idx = _frac_reference(n, x, kernel, out, samples)
    with ran(f"aten::fractional_max_pool{n}d"):
        got, got_idx = _frac(n)(
            x.to(mojo_device),
            kernel,
            output_size=out,
            return_indices=True,
            _random_samples=samples.to(mojo_device),
        )
    torch.testing.assert_close(got_idx.cpu(), want_idx)
    torch.testing.assert_close(got.cpu(), want, atol=0, rtol=0)
    # Backward through autograd: the gradient lands on the saved indices.
    g = torch.randn(want.shape).to(dtype)
    xd = x.to(mojo_device).requires_grad_(True)
    with ran(f"aten::fractional_max_pool{n}d_backward"):
        _frac(n)(
            xd, kernel, output_size=out, _random_samples=samples.to(mojo_device)
        ).backward(g.to(mojo_device))
    planes = math.prod(shape[:-n])
    flat = want_idx.reshape(planes, -1)
    want_g = torch.zeros(planes, math.prod(shape[-n:]), dtype=torch.float64)
    want_g.scatter_add_(1, flat, g.double().reshape(flat.shape))
    assert xd.grad is not None
    _close(xd.grad, want_g.reshape(shape).to(dtype), dtype)


def test_fractional_max_pool_nan_wins(mojo_device):
    """`val > max || isnan(val)`, like the other max pools."""
    x = torch.randn(1, 1, 8, 8)
    x[0, 0, 0, 0] = float("nan")
    x[0, 0, 5, 6] = float("nan")
    samples = torch.rand(1, 1, 2)
    want, want_idx = F.fractional_max_pool2d(
        x, 3, output_size=4, return_indices=True, _random_samples=samples
    )
    got, got_idx = F.fractional_max_pool2d(
        x.to(mojo_device),
        3,
        output_size=4,
        return_indices=True,
        _random_samples=samples.to(mojo_device),
    )
    torch.testing.assert_close(got.cpu(), want, equal_nan=True)
    torch.testing.assert_close(got_idx.cpu(), want_idx)


def test_fractional_max_pool_out_variants(mojo_device):
    x = torch.randn(2, 3, 10, 9)
    samples = torch.rand(2, 3, 2)
    want, want_idx = torch.ops.aten.fractional_max_pool2d(x, [3, 2], [4, 5], samples)
    out = torch.empty(0, device=mojo_device)
    idx = torch.empty(0, dtype=torch.long, device=mojo_device)
    with ran("aten::fractional_max_pool2d.output"):
        torch.ops.aten.fractional_max_pool2d.output(
            x.to(mojo_device),
            [3, 2],
            [4, 5],
            samples.to(mojo_device),
            output=out,
            indices=idx,
        )
    torch.testing.assert_close(out.cpu(), want, atol=0, rtol=0)
    torch.testing.assert_close(idx.cpu(), want_idx)
    g = torch.randn(want.shape)
    want_g = torch.ops.aten.fractional_max_pool2d_backward(
        g, x, [3, 2], [4, 5], want_idx
    )
    gin = torch.empty(2, 3, 9, 10, device=mojo_device).transpose(-1, -2)
    with ran("aten::fractional_max_pool2d_backward.grad_input"):
        torch.ops.aten.fractional_max_pool2d_backward.grad_input(
            g.to(mojo_device), x.to(mojo_device), [3, 2], [4, 5], idx, grad_input=gin
        )
    torch.testing.assert_close(gin.cpu(), want_g)

    x3 = torch.randn(3, 7, 6, 8)
    s3 = torch.rand(1, 3, 3)
    want3, want3_idx = torch.ops.aten.fractional_max_pool3d(
        x3, [2, 2, 3], [3, 4, 2], s3
    )
    out3 = torch.empty(0, device=mojo_device)
    idx3 = torch.empty(0, dtype=torch.long, device=mojo_device)
    torch.ops.aten.fractional_max_pool3d.output(
        x3.to(mojo_device),
        [2, 2, 3],
        [3, 4, 2],
        s3.to(mojo_device),
        output=out3,
        indices=idx3,
    )
    torch.testing.assert_close(out3.cpu(), want3, atol=0, rtol=0)
    torch.testing.assert_close(idx3.cpu(), want3_idx)
    g3 = torch.randn(want3.shape)
    want3_g = torch.ops.aten.fractional_max_pool3d_backward(
        g3, x3, [2, 2, 3], [3, 4, 2], want3_idx
    )
    gin3 = torch.empty(0, device=mojo_device)
    with ran("aten::fractional_max_pool3d_backward.grad_input"):
        torch.ops.aten.fractional_max_pool3d_backward.grad_input(
            g3.to(mojo_device),
            x3.to(mojo_device),
            [2, 2, 3],
            [3, 4, 2],
            idx3,
            grad_input=gin3,
        )
    torch.testing.assert_close(gin3.cpu(), want3_g)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (((1, 2, 6, 6), [7, 2], [1, 2], (1, 2, 2)), "pool height 7 too large"),
        (((1, 2, 6, 6), [2, 2], [3, 6], (1, 2, 2)), "pool width 2 too large"),
        (
            ((1, 2, 6, 6), [0, 2], [3, 3], (1, 2, 2)),
            "kernel size should be greater than zero",
        ),
        (((2, 2, 6, 6), [2, 2], [3, 3], (1, 2, 2)), r"size\(0\) no less then"),
        (
            ((1, 2, 6, 6), [2, 2], [3, 3], (1, 3, 2)),
            r"size\(1\) equals to input channel",
        ),
        (((1, 2, 6, 6), [2, 2], [3, 3], (1, 2, 3)), r"size\(2\) equals to 2; got 3"),
        (
            ((1, 2, 6, 6), [2, 2], [3, 3], (1, 2)),
            "Expect _random_samples to have 3 dimensions",
        ),
        (((2, 6, 6), [2], [3, 3], (1, 2, 2)), "kernel_size must either be"),
        (((1, 2, 0, 6), [2, 2], [3, 3], (1, 2, 2)), "non-zero size for non-batch"),
    ],
)
def test_fractional_max_pool2d_rejects_bad_arguments(mojo_device, args, match):
    shape, k, out, sshape = args
    x = torch.randn(shape).to(mojo_device)
    s = torch.rand(sshape).to(mojo_device)
    with pytest.raises(RuntimeError, match=match):
        torch.ops.aten.fractional_max_pool2d(x, k, out, s)


def test_fractional_max_pool_rejects_bad_samples_dtype_and_3d_sizes(mojo_device):
    x = torch.randn(1, 2, 6, 6, 6).to(mojo_device)
    s = torch.rand(1, 2, 3).to(mojo_device)
    with pytest.raises(RuntimeError, match="same dtype as input"):
        torch.ops.aten.fractional_max_pool3d(x, [2, 2, 2], [3, 3, 3], s.double())
    # 3-D asks out + pool - 1 < in strictly.
    with pytest.raises(
        RuntimeError, match="pool time 2 too large relative to input time 6"
    ):
        torch.ops.aten.fractional_max_pool3d(x, [2, 2, 2], [5, 3, 3], s)


def test_fractional_max_pool3d_backward_rejects_more_planes_than_self(mojo_device):
    """A grad with more batches than `self` would scatter past grad_input."""
    x = torch.randn(1, 2, 5, 5, 5, device=mojo_device)
    grad = torch.randn(4, 2, 2, 2, 2, device=mojo_device)
    idx = torch.zeros(4, 2, 2, 2, 2, dtype=torch.int64, device=mojo_device)
    with pytest.raises(RuntimeError, match="gradOutput sizes unexpected"):
        torch.ops.aten.fractional_max_pool3d_backward(
            grad, x, [2, 2, 2], [2, 2, 2], idx
        )


def test_fractional_max_pool_backward_follows_the_determinism_policy(mojo_device):
    x = torch.randn(1, 2, 6, 6).to(mojo_device)
    s = torch.rand(1, 2, 2).to(mojo_device)
    y, idx = torch.ops.aten.fractional_max_pool2d(x, [2, 2], [3, 3], s)
    torch.use_deterministic_algorithms(True)
    try:
        with pytest.raises(RuntimeError, match="does not have a deterministic"):
            torch.ops.aten.fractional_max_pool2d_backward(y, x, [2, 2], [3, 3], idx)
    finally:
        torch.use_deterministic_algorithms(False)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    ("kernel", "stride", "padding", "dilation", "ceil_mode"),
    [((3, 2), (2, 2), (1, 1), (1, 1), False), (3, 2, 1, 2, True), (2, [], 0, 1, False)],
)
def test_max_pool2d_backward_without_indices(
    mojo_device, dtype, kernel, stride, padding, dilation, ceil_mode
):
    """aten::max_pool2d_backward (an MPS-only kernel upstream) recomputes the
    argmax from `self`: the same gradient as the indexed backward."""
    _skip_f64(mojo_device, dtype)
    as2 = lambda v: [v, v] if isinstance(v, int) else list(v)  # noqa: E731
    args = (
        as2(kernel),
        as2(stride) if stride != [] else [],
        as2(padding),
        as2(dilation),
        ceil_mode,
    )
    x = torch.randn(2, 3, 9, 8).to(dtype)
    y, idx = torch.ops.aten.max_pool2d_with_indices(x, *args)
    g = torch.randn(y.shape).to(dtype)
    want = torch.ops.aten.max_pool2d_with_indices_backward(g, x, *args, idx)
    with ran("aten::max_pool2d_backward"):
        got = torch.ops.aten.max_pool2d_backward(
            g.to(mojo_device), x.to(mojo_device), *args
        )
    _close(got, want, dtype)
    out = torch.empty(0, dtype=dtype, device=mojo_device)
    with ran("aten::max_pool2d_backward.out"):
        torch.ops.aten.max_pool2d_backward.out(
            g.to(mojo_device), x.to(mojo_device), *args, out=out
        )
    _close(out, want, dtype)


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
        lambda t: F.max_unpool2d(
            t, idx[0].to(t.device), (3, 3), (2, 2), (1, 1), (9, 8)
        ),
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
            p3.to(mojo_device),
            i3.to(mojo_device),
            (2, 2, 2),
            None,
            (0, 0, 0),
            (6, 5, 7),
        )
    torch.testing.assert_close(
        got.cpu(), F.max_unpool3d(p3, i3, (2, 2, 2), None, (0, 0, 0), (6, 5, 7))
    )
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
    torch.testing.assert_close(
        F.unfold(x[0].to(mojo_device), *args).cpu(), F.unfold(x[0], *args)
    )
    with ran("aten::col2im"):
        folded = F.fold(want.to(mojo_device), (9, 10), *args)
    if dtype == torch.bool:
        # CPU folds bool as a sum cast to bool: "any".
        torch.testing.assert_close(
            folded.cpu(), F.fold(want.float(), (9, 10), *args) != 0
        )
    else:
        _close(folded, F.fold(want.double(), (9, 10), *args).to(dtype), dtype)


def test_unfold_backward_is_fold(mojo_device):
    x = torch.randn(2, 3, 8, 7)
    want, got = _grad_pair(lambda t: F.unfold(t, 3, 1, 1, 2), x, mojo_device)
    torch.testing.assert_close(got.cpu(), want)


def test_fold_out_variants_and_errors(mojo_device):
    cols = torch.randn(2, 12, 16)
    out = torch.empty(0).to(mojo_device)
    torch.ops.aten.col2im.out(
        cols.to(mojo_device), [5, 5], [2, 2], [1, 1], [0, 0], [1, 1], out=out
    )
    torch.testing.assert_close(
        out.cpu(), torch.ops.aten.col2im(cols, [5, 5], [2, 2], [1, 1], [0, 0], [1, 1])
    )
    with pytest.raises(RuntimeError, match="divisible by the product of kernel_size"):
        F.fold(torch.randn(2, 10, 16).to(mojo_device), (5, 5), 2)
    with pytest.raises(RuntimeError, match="sliding blocks"):
        F.fold(torch.randn(2, 12, 15).to(mojo_device), (5, 5), 2)
    with pytest.raises(RuntimeError, match="must be at least one"):
        F.unfold(torch.randn(1, 1, 2, 2).to(mojo_device), 3)


# ---------------------------------------------------------------------------
# out= contract and supplied indices
# ---------------------------------------------------------------------------


def test_out_is_written_in_place_when_it_can_be(mojo_device):
    """A contiguous, right-shaped out= is computed into directly; a wrongly
    shaped one is resized; a strided one still gets the right values."""
    x = torch.randn(2, 3, 8, 8)
    want = F.avg_pool2d(x, 2)
    out = torch.empty(2, 3, 4, 4).to(mojo_device)
    ptr = out.data_ptr()
    torch.ops.aten.avg_pool2d.out(x.to(mojo_device), [2], out=out)
    assert out.data_ptr() == ptr
    torch.testing.assert_close(out.cpu(), want)
    strided = torch.empty(2, 3, 4, 8).to(mojo_device)[..., ::2]
    torch.ops.aten.avg_pool2d.out(x.to(mojo_device), [2], out=strided)
    torch.testing.assert_close(strided.cpu(), want)


def test_out_rejects_internal_overlap(mojo_device):
    x = torch.randn(1, 1, 4, 4).to(mojo_device)
    out = torch.empty(1).to(mojo_device).expand(1, 1, 2, 2)
    with pytest.raises(RuntimeError, match="more than one element"):
        torch.ops.aten.avg_pool2d.out(x, [2], out=out)


def test_indices_out_wrong_device_or_dtype(mojo_device):
    x = torch.randn(1, 1, 4, 4).to(mojo_device)
    out = torch.empty(0).to(mojo_device)
    with pytest.raises(RuntimeError, match="Expected out tensor to have device"):
        torch.ops.aten.max_pool2d_with_indices.out(
            x, [2], out=out, indices=torch.empty(0, dtype=torch.int64)
        )
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
        torch.ops.aten.adaptive_max_pool2d.out(
            x, [2, 2], out=out, indices=torch.empty(0).to(mojo_device)
        )


def _scatter_reference(g, idx, in_shape, n):
    """CUDA's atomic max-pool backward, one add at a time in the dtype."""
    flat_g = g.reshape(-1, math.prod(g.shape[-n:]))
    flat_i = idx.reshape(flat_g.shape)
    in_plane = math.prod(in_shape[-n:])
    out = torch.zeros(flat_g.shape[0], in_plane, dtype=g.dtype)
    for j in range(flat_g.shape[1]):
        rows = torch.arange(flat_g.shape[0])
        out[rows, flat_i[:, j]] = (
            out[rows, flat_i[:, j]].float() + flat_g[:, j].float()
        ).to(g.dtype)
    return out.reshape(in_shape)


def test_scatter_backwards_take_arbitrary_indices(mojo_device):
    """The 3-D and adaptive max-pool backwards scatter to whatever index
    they are given, as torch's CPU and CUDA kernels do."""
    x = torch.randn(1, 1, 4, 4)
    g = torch.ones(1, 1, 2, 2)
    idx = torch.zeros(1, 1, 2, 2, dtype=torch.int64)
    want = torch.ops.aten.adaptive_max_pool2d_backward(g, x, idx)
    got = torch.ops.aten.adaptive_max_pool2d_backward(
        g.to(mojo_device), x.to(mojo_device), idx.to(mojo_device)
    )
    torch.testing.assert_close(got.cpu(), want)
    assert got.cpu()[0, 0, 0, 0] == 4
    x3 = torch.randn(1, 2, 4, 4, 4).half()
    g3 = torch.randn(1, 2, 2, 2, 2).half()
    idx3 = torch.randint(0, 64, (1, 2, 2, 2, 2))
    args = ([2] * 3, [2] * 3, [0] * 3, [1] * 3, False)
    got = torch.ops.aten.max_pool3d_with_indices_backward(
        g3.to(mojo_device), x3.to(mojo_device), *args, idx3.to(mojo_device)
    )
    want = _scatter_reference(g3, idx3, x3.shape, 3)
    torch.testing.assert_close(got.cpu(), want, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize(
    ("dtype", "size", "want"),
    [(torch.bfloat16, 32, 256.0), (torch.float16, 64, 2048.0)],
)
def test_scatter_backward_accumulates_in_dtype(mojo_device, dtype, size, want):
    """CUDA's atomicAdd rounds a half accumulator on every add: every unit
    gradient of a (size, size) adaptive max pool landing on one input
    saturates where a float accumulator would not (bf16 1024 -> 256,
    fp16 4096 -> 2048)."""
    x = torch.zeros(1, 1, 1, 1, dtype=dtype)
    g = torch.ones(1, 1, size, size, dtype=dtype)
    idx = torch.zeros(1, 1, size, size, dtype=torch.int64)
    got = torch.ops.aten.adaptive_max_pool2d_backward(
        g.to(mojo_device), x.to(mojo_device), idx.to(mojo_device)
    )
    assert got.cpu().item() == want


def _cuda_adaptive_avg_backward(g: torch.Tensor, in_shape) -> torch.Tensor:
    """AdaptiveAveragePooling{,3d}.cu's backward, adds in output order, each
    rounded in the dtype. 2-D: grad / kW / kH in the dtype. 3-D: grad / kT /
    kH / kW in the dtype when a size does not divide, else one float
    division rounded to the dtype."""
    dt = g.dtype
    n = 3 if len(in_shape) == 5 else 2
    isz = in_shape[-n:]
    osz = g.shape[-n:]

    def window(o, os_, is_):
        return (o * is_) // os_, -((-(o + 1) * is_) // os_)

    divisible = all(i % o == 0 for i, o in zip(isz, osz))
    gin = torch.zeros(in_shape, dtype=dt)
    for out in itertools.product(*[range(o) for o in osz]):
        wins = [window(o, os_, is_) for o, os_, is_ in zip(out, osz, isz)]
        ks = [e - b for b, e in wins]
        v = g[(..., *out)]
        if n == 2:
            delta = ((v.float() / ks[1]).to(dt).float() / ks[0]).to(dt)
        elif not divisible:
            delta = v
            for k in ks:
                delta = (delta.float() / k).to(dt)
        else:
            delta = (v.float() / (ks[0] * ks[1] * ks[2])).to(dt)
        sl = (..., *[slice(b, e) for b, e in wins])
        gin[sl] = (gin[sl].float() + delta.float()[(...,) + (None,) * n]).to(dt)
    return gin


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("in_shape", "out_size"),
    [
        ((2, 3, 7, 9), (3, 4)),
        ((2, 3, 8, 8), (4, 2)),
        ((2, 2, 5, 6, 7), (2, 3, 4)),
        ((2, 2, 4, 6, 8), (2, 3, 4)),
    ],
)
def test_adaptive_avg_backward_rounds_like_cuda(mojo_device, dtype, in_shape, out_size):
    torch.manual_seed(0)
    g = (torch.randn(*in_shape[: -len(out_size)], *out_size) * 30).to(dtype)
    x = torch.randn(in_shape).to(dtype)
    op = (
        torch.ops.aten._adaptive_avg_pool2d_backward
        if len(out_size) == 2
        else torch.ops.aten._adaptive_avg_pool3d_backward
    )
    got = op(g.to(mojo_device), x.to(mojo_device)).cpu()
    torch.testing.assert_close(
        got, _cuda_adaptive_avg_backward(g, in_shape), atol=0, rtol=0
    )


def test_max_unpool_invalid_index_raises(mojo_device):
    x = torch.randn(1, 1, 2, 2).to(mojo_device)
    idx = torch.tensor([[[[0, 1], [2, 99]]]]).to(mojo_device)
    with pytest.raises(RuntimeError, match="Found an invalid max index: 99"):
        torch.ops.aten.max_unpool2d(x, idx, [4, 4])


def test_adaptive_avg_pool3d_rejects_zero_channels(mojo_device):
    with pytest.raises(RuntimeError, match="non-zero size for non-batch"):
        torch.ops.aten._adaptive_avg_pool3d(
            torch.randn(1, 0, 2, 2, 2).to(mojo_device), [1, 1, 1]
        )
    # 2-D checks only the spatial dims, like torch.
    got = torch.ops.aten._adaptive_avg_pool2d(
        torch.randn(1, 0, 2, 2).to(mojo_device), [1, 1]
    )
    assert tuple(got.shape) == (1, 0, 1, 1)


def test_out_growing_the_inputs_storage(mojo_device):
    """An out= past the end of the input's own storage is resized, which
    moves that storage: the input must be read from where it now lives."""
    x = torch.randn(2, 3, 8, 8)
    want = F.avg_pool2d(x, 2)
    xd = x.to(mojo_device)
    out = xd.flatten()[xd.numel() :]
    torch.ops.aten.avg_pool2d.out(xd, [2], out=out)
    torch.testing.assert_close(out.cpu(), want)
    torch.testing.assert_close(xd.cpu(), x)


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.uint8])
def test_max_unpool_integer_dtypes(mojo_device, dtype):
    """CUDA unpools every non-bool type (CPU torch only floats), so the
    reference is the scatter it amounts to."""
    x = torch.randint(0, 100, (2, 3, 4, 4)).to(dtype)
    idx = torch.randperm(64)[:16].reshape(1, 1, 4, 4).expand(2, 3, 4, 4).contiguous()
    want = (
        torch.zeros(6, 64, dtype=dtype)
        .scatter_(1, idx.reshape(6, 16), x.reshape(6, 16))
        .reshape(2, 3, 8, 8)
    )
    got = torch.ops.aten.max_unpool2d(x.to(mojo_device), idx.to(mojo_device), [8, 8])
    torch.testing.assert_close(got.cpu(), want)


@pytest.mark.parametrize("adaptive", [False, True])
def test_out_and_indices_share_a_storage(mojo_device, adaptive):
    """`indices` past `out` in one buffer: resizing it moves the storage
    `out` lives in, as the CPU accepts."""
    x = torch.randn(1, 1, 4, 4)
    buf = torch.zeros(4).to(mojo_device)
    out = buf.view(1, 1, 2, 2)
    indices = buf.view(torch.int64)[2:]
    if adaptive:
        want, want_idx = torch.ops.aten.adaptive_max_pool2d(x, [2, 2])
        torch.ops.aten.adaptive_max_pool2d.out(
            x.to(mojo_device), [2, 2], out=out, indices=indices
        )
    else:
        want, want_idx = torch.ops.aten.max_pool2d_with_indices(x, [2])
        torch.ops.aten.max_pool2d_with_indices.out(
            x.to(mojo_device), [2], out=out, indices=indices
        )
    torch.testing.assert_close(out.cpu(), want)
    torch.testing.assert_close(indices.cpu(), want_idx)


def _cuda_avg_pool3d_backward(g, in_shape, k, s, p, cip, divisor):
    """AveragePool3d.cu's backward in the dtype: stride 1 and no padding
    sums every window's grad in float and scales by 1 / divisor once;
    otherwise each window adds `dtype(float(grad) / divisor)` to its inputs
    in the dtype, windows in output order."""
    dt = g.dtype
    isz = in_shape[-3:]
    gin = torch.zeros(in_shape, dtype=dt)
    stride1 = s == [1, 1, 1] and p == [0, 0, 0]
    wide = torch.zeros(in_shape, dtype=torch.float32)
    for out in itertools.product(*[range(o) for o in g.shape[-3:]]):
        wins, sizes, clamped = [], 1, 1
        for d in range(3):
            b = out[d] * s[d] - p[d]
            e = min(b + k[d], isz[d] + p[d])
            sizes *= e - b
            b, e = max(b, 0), min(e, isz[d])
            clamped *= max(e - b, 0)
            wins.append(slice(b, e))
        if clamped == 0:
            continue
        div = divisor or (sizes if cip else clamped)
        v = g[(..., *out)]
        sl = (..., *wins)
        if stride1:
            wide[sl] += v.float()[..., None, None, None]
        else:
            delta = (v.float() / div).to(dt).float()[..., None, None, None]
            gin[sl] = (gin[sl].float() + delta).to(dt)
    if stride1:
        div = divisor or k[0] * k[1] * k[2]
        gin = (wide * (1.0 / div)).to(dt)
    return gin


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("k", "s", "p", "cip", "divisor"),
    [
        ([2, 2, 2], [1, 1, 1], [0, 0, 0], True, None),
        ([3, 2, 2], [1, 1, 1], [0, 0, 0], True, 5),
        ([3, 3, 3], [2, 1, 2], [1, 1, 0], False, None),
        ([2, 3, 2], [1, 2, 1], [1, 1, 1], True, None),
        ([3, 3, 3], [3, 3, 3], [1, 1, 1], True, 7),
    ],
)
def test_avg_pool3d_backward_rounds_like_cuda(
    mojo_device, dtype, k, s, p, cip, divisor
):
    torch.manual_seed(0)
    in_shape = (2, 2, 5, 6, 7)
    x = torch.randn(in_shape).to(dtype)
    out_shape = F.avg_pool3d(
        x.float(), k, s, p, count_include_pad=cip, divisor_override=divisor
    ).shape
    g = (torch.randn(out_shape) * 30).to(dtype)
    got = torch.ops.aten.avg_pool3d_backward(
        g.to(mojo_device), x.to(mojo_device), k, s, p, False, cip, divisor
    ).cpu()
    want = _cuda_avg_pool3d_backward(g, in_shape, k, s, p, cip, divisor)
    torch.testing.assert_close(got, want, atol=0, rtol=0)
