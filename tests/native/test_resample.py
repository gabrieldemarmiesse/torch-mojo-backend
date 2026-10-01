"""Reflection / replication padding and nearest / linear / cubic upsampling
on the mojo device (tmb/ops/resample.mojo), forward and backward, against
CPU torch."""

import pytest
import torch
import torch.nn.functional as F

from tests.native.conftest import ran, skip_if_metal

_FLOATS = [torch.float32, torch.float16, torch.bfloat16]


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    """(atol, rtol) of a forward result."""
    if dtype == torch.float16:
        return 2e-3, 2e-3
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    if dtype == torch.float64:
        return 1e-12, 1e-12
    return 1e-5, 1e-5


def _grad_tol(dtype: torch.dtype) -> tuple[float, float]:
    """A gradient sums up to a few dozen terms; the half types round each
    term on CPU (and on CUDA, whose scatter adds in the input dtype) where
    this backend sums in float32 and rounds once."""
    if dtype == torch.float16:
        return 2e-2, 1e-2
    if dtype == torch.bfloat16:
        return 1.5e-1, 5e-2
    if dtype == torch.float64:
        return 1e-12, 1e-12
    return 1e-5, 1e-5


def _check_fwd_bwd(fn, x: torch.Tensor, device: str, op: str):
    """fn on CPU vs on the device, value and input gradient, with the named
    native forward op required to run."""
    dtype = x.dtype
    xc = x.clone().requires_grad_(True)
    xm = x.to(device).requires_grad_(True)
    with ran(op):
        ym = fn(xm)
    yc = fn(xc)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(ym.cpu(), yc, atol=atol, rtol=rtol)
    g = torch.randn(yc.shape).to(dtype)
    with ran(op + "_backward"):
        ym.backward(g.to(device))
    yc.backward(g)
    assert xm.grad is not None and xc.grad is not None
    atol, rtol = _grad_tol(dtype)
    torch.testing.assert_close(xm.grad.cpu(), xc.grad, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

_PAD_CASES = [
    # (shape, padding) -- 1-d, 2-d and 3-d, batched and not, asymmetric,
    # zero on one side, the largest reflection (n - 1), negative (cropping).
    ((2, 3, 7), (2, 3)),
    ((3, 7), (0, 6)),
    ((2, 3, 7), (-2, 3)),
    ((2, 3, 6, 5), (1, 4, 0, 5)),
    ((3, 6, 5), (4, 2, 3, 1)),
    ((2, 3, 6, 5), (2, -1, -3, 2)),
    ((2, 2, 4, 5, 6), (1, 2, 3, 0, 2, 3)),
    ((2, 4, 5, 6), (5, 0, 1, 4, 3, 1)),
    ((1, 2, 4, 5, 6), (-1, 2, 1, -2, 0, 1)),
]


def _pad_op(mode: str, rank: int) -> str:
    base = "reflection" if mode == "reflect" else "replication"
    return f"aten::{base}_pad{rank}d"


@pytest.mark.parametrize("dtype", [*_FLOATS, torch.int64, torch.uint8])
@pytest.mark.parametrize("mode", ["reflect", "replicate"])
@pytest.mark.parametrize(("shape", "padding"), _PAD_CASES)
def test_pad_forward(mojo_device, mode, shape, padding, dtype):
    x = (torch.randn(shape) * 50).to(dtype)
    with ran(_pad_op(mode, len(padding) // 2)):
        got = F.pad(x.to(mojo_device), padding, mode=mode)
    torch.testing.assert_close(got.cpu(), F.pad(x, padding, mode=mode), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", _FLOATS)
@pytest.mark.parametrize("mode", ["reflect", "replicate"])
@pytest.mark.parametrize(("shape", "padding"), _PAD_CASES)
def test_pad_backward(mojo_device, mode, shape, padding, dtype):
    x = torch.randn(shape).to(dtype)
    _check_fwd_bwd(
        lambda t: F.pad(t, padding, mode=mode),
        x,
        mojo_device,
        _pad_op(mode, len(padding) // 2),
    )


@pytest.mark.parametrize("mode", ["reflect", "replicate"])
def test_pad_float64(mojo_device, mode):
    skip_if_metal(mojo_device, "Apple GPUs have no float64")
    x = torch.randn(2, 3, 5, 6, dtype=torch.float64)
    _check_fwd_bwd(
        lambda t: F.pad(t, (2, 1, 4, 3), mode=mode), x, mojo_device, _pad_op(mode, 2)
    )


def test_replicate_pad_far_past_the_input(mojo_device):
    """Replication has no upper bound: the edge rows sum many gradients."""
    x = torch.randn(2, 3, 3, 2)
    _check_fwd_bwd(
        lambda t: F.pad(t, (9, 7, 11, 6), mode="replicate"),
        x,
        mojo_device,
        "aten::replication_pad2d",
    )


@pytest.mark.parametrize("mode", ["reflect", "replicate"])
def test_pad_strided_input(mojo_device, mode):
    x = torch.randn(2, 3, 9, 8).transpose(-1, -2)
    got = F.pad(x.to(mojo_device), (2, 3, 1, 4), mode=mode)
    torch.testing.assert_close(got.cpu(), F.pad(x, (2, 3, 1, 4), mode=mode))


def test_pad_empty_batch(mojo_device):
    x = torch.randn(0, 3, 5)
    got = F.pad(x.to(mojo_device), (2, 1), mode="reflect")
    assert got.shape == (0, 3, 8)


@pytest.mark.parametrize("contiguous", [True, False])
def test_pad_out_variants(mojo_device, contiguous):
    x = torch.randn(2, 3, 4, 5)
    want = torch.ops.aten.reflection_pad2d(x, [1, 2, 3, 0])
    if contiguous:
        out = torch.empty(0, device=mojo_device)
    else:
        out = torch.empty(2, 3, 8, 7, device=mojo_device).transpose(-1, -2)
        assert out.shape == want.shape and not out.is_contiguous()
    with ran("aten::reflection_pad2d.out"):
        res = torch.ops.aten.reflection_pad2d.out(
            x.to(mojo_device), [1, 2, 3, 0], out=out
        )
    assert res is out
    torch.testing.assert_close(out.cpu(), want)

    g = torch.randn(want.shape)
    want_grad = torch.ops.aten.replication_pad2d_backward(g, x, [1, 2, 3, 0])
    grad_input = torch.empty(0, device=mojo_device)
    with ran("aten::replication_pad2d_backward.grad_input"):
        torch.ops.aten.replication_pad2d_backward.grad_input(
            g.to(mojo_device), x.to(mojo_device), [1, 2, 3, 0], grad_input=grad_input
        )
    torch.testing.assert_close(grad_input.cpu(), want_grad)


def test_reflection_pad_too_large_raises(mojo_device):
    x = torch.randn(2, 3, 4).to(mojo_device)
    with pytest.raises(RuntimeError, match="Padding size should be less than"):
        F.pad(x, (4, 0), mode="reflect")


def test_pad_bad_rank_raises(mojo_device):
    x = torch.randn(2, 3, 4, 5, 6).to(mojo_device)
    with pytest.raises(RuntimeError, match="Expected 2D or 3D"):
        torch.ops.aten.replication_pad1d(x, [1, 1])


# ---------------------------------------------------------------------------
# Upsampling
# ---------------------------------------------------------------------------

_UP_CASES = [
    # (mode, input shape, interpolate kwargs, native op)
    ("nearest", (2, 3, 5), {"size": (8,)}, "upsample_nearest1d"),
    ("nearest", (2, 3, 7), {"scale_factor": 0.6}, "upsample_nearest1d"),
    ("nearest", (2, 3, 5, 4), {"size": (7, 9)}, "upsample_nearest2d"),
    ("nearest", (1, 2, 5, 7), {"scale_factor": 1.7}, "upsample_nearest2d"),
    ("nearest", (2, 3, 5, 4, 3), {"size": (7, 2, 5)}, "upsample_nearest3d"),
    ("nearest-exact", (2, 3, 5), {"size": (8,)}, "_upsample_nearest_exact1d"),
    ("nearest-exact", (2, 3, 5, 4), {"size": (3, 9)}, "_upsample_nearest_exact2d"),
    (
        "nearest-exact",
        (2, 3, 4, 4, 4),
        {"scale_factor": 1.7},
        "_upsample_nearest_exact3d",
    ),
    ("linear", (2, 3, 5), {"size": (8,)}, "upsample_linear1d"),
    ("linear", (2, 3, 9), {"size": (4,), "align_corners": True}, "upsample_linear1d"),
    ("linear", (2, 3, 5), {"scale_factor": 1.7}, "upsample_linear1d"),
    ("bilinear", (2, 3, 5, 4), {"size": (7, 9)}, "upsample_bilinear2d"),
    (
        "bilinear",
        (2, 3, 5, 4),
        {"size": (3, 9), "align_corners": True},
        "upsample_bilinear2d",
    ),
    ("bilinear", (2, 3, 8, 8), {"scale_factor": 0.6}, "upsample_bilinear2d"),
    ("bicubic", (2, 3, 5, 4), {"size": (7, 9)}, "upsample_bicubic2d"),
    (
        "bicubic",
        (2, 3, 5, 4),
        {"size": (3, 2), "align_corners": True},
        "upsample_bicubic2d",
    ),
    ("bicubic", (2, 3, 4, 4), {"scale_factor": 1.7}, "upsample_bicubic2d"),
    ("bicubic", (1, 2, 9, 7), {"size": (4, 3)}, "upsample_bicubic2d"),
    ("trilinear", (2, 3, 5, 4, 3), {"size": (7, 2, 5)}, "upsample_trilinear3d"),
    (
        "trilinear",
        (2, 3, 5, 4, 3),
        {"size": (7, 2, 5), "align_corners": True},
        "upsample_trilinear3d",
    ),
    ("trilinear", (1, 2, 4, 4, 4), {"scale_factor": 0.6}, "upsample_trilinear3d"),
    (
        "bilinear",
        (2, 3, 10, 20),
        {"size": (3, 7), "antialias": True},
        "_upsample_bilinear2d_aa",
    ),
    (
        "bilinear",
        (1, 2, 5, 6),
        {"scale_factor": (1.7, 0.9), "antialias": True},
        "_upsample_bilinear2d_aa",
    ),
    (
        "bicubic",
        (2, 3, 10, 20),
        {"size": (3, 7), "antialias": True},
        "_upsample_bicubic2d_aa",
    ),
    (
        "bicubic",
        (1, 2, 5, 6),
        {"scale_factor": (1.7, 0.9), "antialias": True},
        "_upsample_bicubic2d_aa",
    ),
    (
        "bicubic",
        (1, 2, 13, 11),
        {"size": (4, 17), "align_corners": True, "antialias": True},
        "_upsample_bicubic2d_aa",
    ),
    # Same size: the copy special case of the 1-d / 3-d / cubic kernels.
    ("linear", (2, 3, 5), {"size": (5,)}, "upsample_linear1d"),
    ("bicubic", (2, 3, 5, 4), {"size": (5, 4)}, "upsample_bicubic2d"),
]


def _interp(mode: str, kwargs):
    return lambda t: F.interpolate(t, mode=mode, **kwargs)


@pytest.mark.parametrize("dtype", _FLOATS)
@pytest.mark.parametrize(
    ("mode", "shape", "kwargs", "op"),
    _UP_CASES,
    ids=[f"{m}-{s}-{k}" for m, s, k, _ in _UP_CASES],
)
def test_upsample_forward_backward(mojo_device, mode, shape, kwargs, op, dtype):
    if kwargs.get("antialias") and dtype != torch.float32:
        pytest.skip(
            "CPU torch has no half antialiased kernel to compare against "
            "(CUDA parity of the half types is checked by hand)"
        )
    x = torch.randn(shape).to(dtype)
    _check_fwd_bwd(_interp(mode, kwargs), x, mojo_device, "aten::" + op)


@pytest.mark.parametrize(
    ("mode", "shape", "kwargs", "op"),
    _UP_CASES,
    ids=[f"{m}-{s}-{k}" for m, s, k, _ in _UP_CASES],
)
def test_upsample_float64(mojo_device, mode, shape, kwargs, op):
    skip_if_metal(mojo_device, "Apple GPUs have no float64")
    x = torch.randn(shape, dtype=torch.float64)
    _check_fwd_bwd(_interp(mode, kwargs), x, mojo_device, "aten::" + op)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
def test_upsample_antialias_half(mojo_device, mode, dtype):
    """CPU torch has no half antialiased kernel: compare with its float32
    result, rounded (the weights and the row pass round to the dtype on the
    device, as on CUDA)."""
    x = torch.randn(2, 3, 19, 11).to(dtype)
    want = F.interpolate(x.float(), size=(6, 13), mode=mode, antialias=True)
    with ran(f"aten::_upsample_{mode}2d_aa"):
        got = F.interpolate(x.to(mojo_device), size=(6, 13), mode=mode, antialias=True)
    atol, rtol = _grad_tol(dtype)
    torch.testing.assert_close(got.cpu().float(), want, atol=atol, rtol=rtol)


_HAS_LANCZOS = hasattr(torch.ops.aten, "_upsample_lanczos2d_aa")


@pytest.mark.skipif(not _HAS_LANCZOS, reason="lanczos interpolation is torch 2.14+")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("shape", "kwargs"),
    [
        ((2, 3, 10, 20), {"size": (3, 7)}),
        ((1, 2, 5, 6), {"size": (9, 5)}),
        ((1, 2, 13, 11), {"size": (4, 17), "align_corners": True}),
        ((1, 1, 6, 7), {"size": (6, 3)}),  # an unchanged axis is skipped
        ((1, 2, 8, 9), {"scale_factor": (2.0, 0.5)}),
    ],
)
def test_upsample_lanczos(mojo_device, shape, kwargs, dtype):
    """Lanczos-3 (CPU-only upstream): forward and backward against CPU."""
    if dtype == torch.float64:
        skip_if_metal(mojo_device, "Apple GPUs have no float64")
    x = torch.randn(shape, dtype=dtype)
    _check_fwd_bwd(
        lambda t: F.interpolate(t, mode="lanczos", antialias=True, **kwargs),
        x,
        mojo_device,
        "aten::_upsample_lanczos2d_aa",
    )


@pytest.mark.skipif(not _HAS_LANCZOS, reason="lanczos interpolation is torch 2.14+")
def test_upsample_lanczos_same_size_copies(mojo_device):
    x = torch.randn(1, 2, 5, 6)
    got = torch.ops.aten._upsample_lanczos2d_aa(x.to(mojo_device), [5, 6], False)
    torch.testing.assert_close(got.cpu(), x, atol=0, rtol=0)


def test_upsample_aa_out_variants(mojo_device):
    x = torch.randn(2, 3, 9, 8)
    want = torch.ops.aten._upsample_bicubic2d_aa(x, [4, 5], False)
    out = torch.empty(0, device=mojo_device)
    with ran("aten::_upsample_bicubic2d_aa.out"):
        torch.ops.aten._upsample_bicubic2d_aa.out(
            x.to(mojo_device), [4, 5], False, out=out
        )
    torch.testing.assert_close(out.cpu(), want, atol=1e-5, rtol=1e-5)
    g = torch.randn(2, 3, 4, 5)
    want_g = torch.ops.aten._upsample_bicubic2d_aa_backward(
        g, [4, 5], [2, 3, 9, 8], False
    )
    gin = torch.empty(2, 3, 8, 9, device=mojo_device).transpose(-1, -2)
    with ran("aten::_upsample_bicubic2d_aa_backward.grad_input"):
        torch.ops.aten._upsample_bicubic2d_aa_backward.grad_input(
            g.to(mojo_device), [4, 5], [2, 3, 9, 8], False, grad_input=gin
        )
    torch.testing.assert_close(gin.cpu(), want_g, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mode", ["nearest", "nearest-exact"])
def test_upsample_nearest_is_a_gather(mojo_device, mode):
    """Nearest copies elements: bit-identical, uint8 included."""
    for dtype in (torch.float32, torch.bfloat16, torch.uint8):
        x = (torch.randn(2, 3, 5, 7) * 50).to(dtype)
        got = F.interpolate(x.to(mojo_device), size=(9, 4), mode=mode)
        torch.testing.assert_close(
            got.cpu(), F.interpolate(x, size=(9, 4), mode=mode), atol=0, rtol=0
        )


def test_upsample_strided_input(mojo_device):
    x = torch.randn(2, 3, 6, 5).transpose(-1, -2)
    got = F.interpolate(x.to(mojo_device), size=(9, 4), mode="bicubic")
    torch.testing.assert_close(
        got.cpu(), F.interpolate(x, size=(9, 4), mode="bicubic"), atol=1e-5, rtol=1e-5
    )


def test_upsample_empty_batch(mojo_device):
    x = torch.randn(0, 3, 5, 4)
    got = F.interpolate(x.to(mojo_device), size=(7, 9), mode="bilinear")
    assert got.shape == (0, 3, 7, 9)


def test_upsample_explicit_scales(mojo_device):
    """A scale that disagrees with the sizes is what the source index uses."""
    x = torch.randn(1, 2, 4, 5)
    want = torch.ops.aten.upsample_bicubic2d(x, [8, 10], False, 2.5, 1.5)
    got = torch.ops.aten.upsample_bicubic2d(x.to(mojo_device), [8, 10], False, 2.5, 1.5)
    torch.testing.assert_close(got.cpu(), want, atol=1e-5, rtol=1e-5)
    want = torch.ops.aten.upsample_linear1d(x[0], [8], False, 3.0)
    got = torch.ops.aten.upsample_linear1d(x[0].to(mojo_device), [8], False, 3.0)
    torch.testing.assert_close(got.cpu(), want, atol=1e-5, rtol=1e-5)


def test_upsample_out_variants(mojo_device):
    x = torch.randn(2, 3, 4, 5)
    want = torch.ops.aten.upsample_trilinear3d(x[None], [3, 7, 2], True)
    out = torch.empty(0, device=mojo_device)
    with ran("aten::upsample_trilinear3d.out"):
        torch.ops.aten.upsample_trilinear3d.out(
            x[None].to(mojo_device), [3, 7, 2], True, out=out
        )
    torch.testing.assert_close(out.cpu(), want, atol=1e-5, rtol=1e-5)

    g = torch.randn(2, 3, 9, 7)
    want_grad = torch.ops.aten.upsample_nearest2d_backward(g, [9, 7], [2, 3, 4, 5])
    grad_input = torch.empty(2, 3, 5, 4, device=mojo_device).transpose(-1, -2)
    with ran("aten::upsample_nearest2d_backward.grad_input"):
        torch.ops.aten.upsample_nearest2d_backward.grad_input(
            g.to(mojo_device), [9, 7], [2, 3, 4, 5], grad_input=grad_input
        )
    torch.testing.assert_close(grad_input.cpu(), want_grad)


def test_upsample_bad_output_size_raises(mojo_device):
    x = torch.randn(2, 3, 4, 5).to(mojo_device)
    with pytest.raises(RuntimeError, match="Input and output sizes should be greater"):
        torch.ops.aten.upsample_bilinear2d(x, [0, 3], False)
    with pytest.raises(RuntimeError, match="It is expected output_size equals to 2"):
        torch.ops.aten.upsample_nearest2d(x, [3])


@pytest.mark.parametrize("mode", ["nearest", "nearest-exact"])
def test_upsample_nearest2d_keeps_an_unresized_axis(mojo_device, mode):
    """upsample_nearest2d copies an axis whose size does not change, whatever
    its scale: rows stay rows here although 1.4 would duplicate row 0."""
    x = torch.arange(6.0).reshape(1, 1, 2, 3)
    got = F.interpolate(x.to(mojo_device), scale_factor=(1.4, 2), mode=mode)
    want = F.interpolate(x, scale_factor=(1.4, 2), mode=mode)
    torch.testing.assert_close(got.cpu(), want, atol=0, rtol=0)


def test_upsample_backward_zero_weight_tap_of_inf_is_nan(mojo_device):
    """CUDA scatters `0 * grad` for a zero-weight tap: NaN for an inf grad."""
    g = torch.zeros(1, 1, 3, 3)
    g[0, 0, 0, 0] = float("inf")
    want = torch.ops.aten.upsample_bilinear2d_backward(g, [3, 3], [1, 1, 2, 2], False)
    got = torch.ops.aten.upsample_bilinear2d_backward(
        g.to(mojo_device), [3, 3], [1, 1, 2, 2], False
    )
    torch.testing.assert_close(got.cpu(), want, equal_nan=True)


def test_upsample_nearest_backward_uint8(mojo_device):
    """CUDA sums a uint8 gradient in int64 and wraps it back to uint8."""
    g = (torch.arange(36).reshape(1, 1, 6, 6) * 9).to(torch.uint8)
    want = torch.ops.aten.upsample_nearest2d_backward(g.double(), [6, 6], [1, 1, 4, 4])
    got = torch.ops.aten.upsample_nearest2d_backward(
        g.to(mojo_device), [6, 6], [1, 1, 4, 4]
    )
    torch.testing.assert_close(got.cpu(), want.long().to(torch.uint8))


def test_reflection_pad_backward_checks_the_padding(mojo_device):
    x = torch.randn(1, 3).to(mojo_device)
    g = torch.randn(1, 7).to(mojo_device)
    with pytest.raises(RuntimeError, match="Padding size should be less than"):
        torch.ops.aten.reflection_pad1d_backward(g, x, [3, 1])


def test_out_overlap_follows_torch(mojo_device):
    """torch's pad and upsample kernels check no overlap with the input
    (upsample's out still refuses internal overlap). An overlapping out gets
    the result computed as if the input were read first."""
    x = torch.randn(1, 1, 4, 4)
    xm = x.to(mojo_device)
    # Pad: an expanded out is accepted (torch writes it racily; the values
    # are not checked).
    expanded = torch.empty(1, 1, 1, 1, device=mojo_device).expand(1, 1, 6, 6)
    torch.ops.aten.replication_pad2d.out(xm, [1, 1, 1, 1], out=expanded)
    # Upsample: an expanded out still raises.
    with pytest.raises(RuntimeError, match="more than one element"):
        torch.ops.aten.upsample_nearest2d.out(xm, [6, 6], None, None, out=expanded)
    # A strided (non-dense) out is TooHard for ATen's overlap check: allowed.
    torch.ops.aten.upsample_nearest1d.out(xm[0], [2], None, out=xm[0][..., :2])
    # A partially overlapping dense out: accepted. The kernel reads the
    # input while writing the out (a race on CUDA too), so only what does
    # not depend on that order is checked: the shape, and the storage
    # outside the out left untouched.
    for op, extra in (
        (torch.ops.aten.reflection_pad2d.out, ([1, 1, 1, 1],)),
        (torch.ops.aten.upsample_nearest2d.out, ([6, 6], None, None)),
    ):
        buf = torch.arange(64.0)
        dev = buf.to(mojo_device)
        res = op(dev[:16].view(1, 1, 4, 4), *extra, out=dev[8:44].view(1, 1, 6, 6))
        assert res.shape == (1, 1, 6, 6) and res.dtype == torch.float32
        got = dev.cpu()
        torch.testing.assert_close(got[:8], buf[:8], atol=0, rtol=0)
        torch.testing.assert_close(got[44:], buf[44:], atol=0, rtol=0)


def test_same_size_upsample_out_partial_overlap_raises(mojo_device):
    """CUDA's same-size shortcut is `output.copy_(input)`: a partially
    overlapping out raises, forward and backward; pad has no such check."""
    buf = torch.zeros(64, device=mojo_device)
    src = buf[:16].view(1, 1, 4, 4)
    dst = buf[8:24].view(1, 1, 4, 4)
    with pytest.raises(RuntimeError, match="single memory location"):
        torch.ops.aten.upsample_nearest2d.out(src, [4, 4], None, None, out=dst)
    with pytest.raises(RuntimeError, match="single memory location"):
        torch.ops.aten.upsample_bilinear2d.out(src, [4, 4], False, None, None, out=dst)
    with pytest.raises(RuntimeError, match="single memory location"):
        torch.ops.aten.upsample_nearest2d_backward.grad_input(
            src, [4, 4], [1, 1, 4, 4], None, None, grad_input=dst
        )
    torch.ops.aten.reflection_pad2d.out(src, [0, 0, 0, 0], out=dst)


def test_unchanged_size_out_is_the_input(mojo_device):
    """CUDA's unchanged-size shortcut is `out.copy_(input)`: with `out` the
    input itself it is a no-op, not an overlap error."""
    x = torch.randn(1, 1, 4, 4)
    xm = x.to(mojo_device)
    res = torch.ops.aten.upsample_nearest2d.out(xm, [4, 4], None, None, out=xm)
    assert res is xm
    torch.testing.assert_close(xm.cpu(), x, atol=0, rtol=0)


def test_bilinear2d_unchanged_size_copies_despite_the_scale(mojo_device):
    """upsample_bilinear2d copies an unchanged-size input whatever the scale."""
    x = torch.randn(1, 2, 4, 5)
    got = torch.ops.aten.upsample_bilinear2d(x.to(mojo_device), [4, 5], False, 3.0, 1.5)
    torch.testing.assert_close(got.cpu(), x, atol=0, rtol=0)


def test_out_sharing_storage_is_resized_after_the_kernel(mojo_device):
    """An empty `out` over the input's storage is resized (possibly moving
    the storage) only after the input was read."""
    buf = torch.arange(16.0)
    want = F.pad(buf.view(1, 1, 4, 4), (1, 1, 1, 1), mode="replicate")
    bm = buf.to(mojo_device)
    out = bm[16:16]
    torch.ops.aten.replication_pad2d.out(bm.view(1, 1, 4, 4), [1, 1, 1, 1], out=out)
    torch.testing.assert_close(out.cpu(), want, atol=0, rtol=0)


def test_bicubic_backward_inf_grad_matches_cuda(mojo_device):
    """CUDA scatters every (output, tap) product: an inf gradient meets
    coefficients of both signs and zeros, giving [[nan, nan], [nan, inf]]."""
    g = torch.zeros(1, 1, 3, 3)
    g[0, 0, 0, 0] = float("inf")
    got = torch.ops.aten.upsample_bicubic2d_backward(
        g.to(mojo_device), [3, 3], [1, 1, 2, 2], False
    )
    want = torch.tensor(
        [[[[float("nan"), float("nan")], [float("nan"), float("inf")]]]]
    )
    torch.testing.assert_close(got.cpu(), want, equal_nan=True)


@pytest.mark.parametrize("op", ["upsample_nearest1d", "_upsample_nearest_exact1d"])
def test_unchanged_size_nearest1d_out_is_the_input(mojo_device, op):
    x = torch.randn(1, 2, 5)
    xm = x.to(mojo_device)
    res = getattr(torch.ops.aten, op).out(xm, [5], None, out=xm)
    assert res is xm
    torch.testing.assert_close(xm.cpu(), x, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("op", "extra"),
    [
        ("upsample_bilinear2d_backward", [False]),
        ("upsample_bicubic2d_backward", [False]),
        ("_upsample_bilinear2d_aa_backward", [False]),
        ("upsample_nearest2d_backward", []),
    ],
)
def test_backward_grad_input_is_grad_output(mojo_device, op, extra):
    """CUDA zeroes grad_input before copying an unchanged-size grad_output
    into it for the interpolating kernels (so an aliased call returns
    zeros); nearest2d copies with no zeroing (a no-op)."""
    g = torch.randn(1, 2, 3, 4) + 1
    gm = g.to(mojo_device)
    getattr(torch.ops.aten, op).grad_input(
        gm, [3, 4], [1, 2, 3, 4], *extra, grad_input=gm
    )
    want = g if "nearest" in op else torch.zeros_like(g)
    torch.testing.assert_close(gm.cpu(), want, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("op", "extra", "survives"),
    [
        ("upsample_linear1d_backward", [False], True),
        ("upsample_bicubic2d_backward", [False], True),
        ("_upsample_bilinear2d_aa_backward", [False], True),
        ("upsample_bilinear2d_backward", [False], False),
    ],
)
def test_backward_strided_grad_input_is_grad_output(mojo_device, op, extra, survives):
    """With a strided aliased grad_output, the kernels that read a
    `.contiguous()` copy of it keep its values; bilinear2d copies from the
    zeroed tensor itself and returns zeros."""
    if "linear1d" in op:
        base = torch.randn(1, 2, 5, 2) + 1
        g, osize, isize = base[..., 0], [5], [1, 2, 5]
    else:
        base = torch.randn(1, 2, 4, 3) + 1
        g, osize, isize = base.transpose(-1, -2), [3, 4], [1, 2, 3, 4]
    gm = base.to(mojo_device)
    gm = gm[..., 0] if "linear1d" in op else gm.transpose(-1, -2)
    assert not gm.is_contiguous()
    getattr(torch.ops.aten, op).grad_input(gm, osize, isize, *extra, grad_input=gm)
    want = g if survives else torch.zeros_like(g)
    torch.testing.assert_close(gm.cpu(), want, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# grid_sampler_2d / grid_sampler_3d (F.grid_sample)
# ---------------------------------------------------------------------------

_GRID_MODES_2D = ["bilinear", "nearest", "bicubic"]
_GRID_PADS = ["zeros", "border", "reflection"]


def _grid_inputs(shape, out_spatial, dtype, spread=1.4, seed=0):
    """An input and a grid reaching past [-1, 1] by `spread`, in `dtype`."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=gen).to(dtype)
    rank = len(shape) - 2
    grid = (torch.rand(shape[0], *out_spatial, rank, generator=gen) * 2 - 1) * spread
    return x, grid.to(dtype)


def _exact_grid(shape, out_spatial, align, seed=0):
    """A grid whose source coordinates are a quarter off a pixel (some
    outside the image), with every value exact in bfloat16 for the extents
    the tests use (W = 8 unaligned, W = 9 aligned, likewise H and D): the
    CUDA backward computes the source index in the 16-bit type itself, so
    only such a grid has one answer to compare a float64 reference with."""
    gen = torch.Generator().manual_seed(seed)
    rank = len(shape) - 2
    sizes = list(reversed(shape[2:]))  # x, y, z extents
    coords = []
    for k in range(rank):
        n = sizes[k]
        pix = torch.randint(-2, n + 2, (shape[0], *out_spatial), generator=gen)
        frac = torch.where(torch.rand(pix.shape, generator=gen) < 0.5, 0.25, 0.75)
        ix = pix + frac
        coords.append(ix * 2 / (n - 1) - 1 if align else (2 * ix + 1) / n - 1)
    return torch.stack(coords, -1)


def _grid_check(x, grid, device, mode, pad, align, fwd_tol, grad_tol):
    """Forward value and both gradients on the device against CPU float64
    (the CUDA kernels compute half forwards in float32 and half backwards
    in the 16-bit type; a float64 reference bounds both)."""
    op = "aten::grid_sampler_2d" if x.dim() == 4 else "aten::grid_sampler_3d"
    x, grid = x.detach(), grid.detach()
    xm = x.to(device).requires_grad_(True)
    gm = grid.to(device).requires_grad_(True)
    xc = x.double().clone().requires_grad_(True)
    gc = grid.double().clone().requires_grad_(True)
    kw = dict(mode=mode, padding_mode=pad, align_corners=align)
    with ran(op):
        ym = F.grid_sample(xm, gm, **kw)
    yc = F.grid_sample(xc, gc, **kw)
    assert ym.dtype == x.dtype
    torch.testing.assert_close(ym.cpu().double(), yc, atol=fwd_tol, rtol=fwd_tol)
    go = torch.randn(yc.shape, generator=torch.Generator().manual_seed(1))
    go = go.to(x.dtype)
    with ran(op + "_backward"):
        ym.backward(go.to(device))
    yc.backward(go.double())
    torch.testing.assert_close(
        xm.grad.cpu().double(), xc.grad, atol=grad_tol, rtol=grad_tol
    )
    torch.testing.assert_close(
        gm.grad.cpu().double(), gc.grad, atol=grad_tol * 10, rtol=grad_tol * 10
    )


_GRID_TOL = {
    torch.float32: (1e-5, 1e-4),
    torch.float16: (4e-3, 3e-2),
    torch.bfloat16: (3e-2, 2.5e-1),
}


@pytest.mark.parametrize("dtype", _FLOATS)
@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("pad", _GRID_PADS)
@pytest.mark.parametrize("mode", _GRID_MODES_2D)
def test_grid_sample_2d(mojo_device, mode, pad, align, dtype):
    n = 9 if align else 8
    x, grid = _grid_inputs((2, 3, n, n), (4, 6), dtype)
    if dtype != torch.float32:
        grid = _exact_grid(x.shape, (4, 6), align).to(dtype)
    fwd, grad = _GRID_TOL[dtype]
    if mode == "bicubic":
        fwd, grad = fwd * 4, grad * 4
    _grid_check(x, grid, mojo_device, mode, pad, align, fwd, grad)


@pytest.mark.parametrize("dtype", _FLOATS)
@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("pad", _GRID_PADS)
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
def test_grid_sample_3d(mojo_device, mode, pad, align, dtype):
    shape = (2, 3, 5, 9, 9) if align else (2, 3, 4, 8, 8)
    x, grid = _grid_inputs(shape, (3, 4, 2), dtype)
    if dtype != torch.float32:
        grid = _exact_grid(x.shape, (3, 4, 2), align).to(dtype)
    fwd, grad = _GRID_TOL[dtype]
    _grid_check(x, grid, mojo_device, mode, pad, align, fwd, grad)


@pytest.mark.parametrize("mode", _GRID_MODES_2D)
def test_grid_sample_float64(mojo_device, mode):
    skip_if_metal(mojo_device, "Apple GPUs have no float64")
    x, grid = _grid_inputs((2, 3, 6, 5), (7, 3), torch.float64, spread=3.0)
    for pad in _GRID_PADS:
        _grid_check(x, grid, mojo_device, mode, pad, False, 1e-12, 1e-11)
    if mode != "bicubic":
        x, grid = _grid_inputs((1, 2, 3, 4, 5), (2, 3, 4), torch.float64, spread=3.0)
        _grid_check(x, grid, mojo_device, mode, "reflection", True, 1e-12, 1e-11)


def test_grid_sample_far_coordinates(mojo_device):
    """Grid values many spans away: the reflection's fmod / flip count and
    the border clip; a nearest tap at exact half-way points rounds to
    even, as nearbyint does."""
    x, _ = _grid_inputs((1, 2, 4, 5), (1, 1), torch.float32)
    grid = torch.tensor([[[[37.3, -11.9], [-123.6, 4.2], [0.25, -0.375], [1e6, -7.5]]]])
    for mode in _GRID_MODES_2D:
        for pad in _GRID_PADS:
            for align in (False, True):
                _grid_check(x, grid, mojo_device, mode, pad, align, 1e-4, 1e-3)


def test_grid_sample_nonfinite_grid_samples_zero(mojo_device):
    """NaN / inf coordinates fall outside every image (CUDA's
    safe_downgrade_to_int_range) under zero padding."""
    x = torch.randn(1, 2, 3, 3)
    grid = torch.tensor(
        [[[[float("nan"), 0.0], [float("inf"), 0.0], [0.0, -float("inf")]]]]
    )
    for mode in _GRID_MODES_2D:
        y = F.grid_sample(
            x.to(mojo_device), grid.to(mojo_device), mode=mode, align_corners=False
        )
        if mode == "bicubic":
            # bicubic's unnormalized NaN poisons its weights
            continue
        assert torch.equal(y.cpu(), torch.zeros(1, 2, 1, 3))


@pytest.mark.parametrize("pad", ["border", "reflection"])
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
def test_grid_sample_nan_grid_clips_to_the_origin(mojo_device, mode, pad):
    """CUDA's clip_coordinates is fmin/fmax, which drop a NaN: a NaN
    coordinate under border / reflection padding samples pixel 0 (CPU's
    std::min/max keeps the other bound, so CPU is not the reference)."""
    x = torch.randn(1, 2, 3, 4)
    grid = torch.full((1, 1, 2, 2), float("nan"))
    grid[0, 0, 1] = torch.tensor([float("nan"), -1.0])
    y = F.grid_sample(
        x.to(mojo_device),
        grid.to(mojo_device),
        mode=mode,
        padding_mode=pad,
        align_corners=True,
    ).cpu()
    torch.testing.assert_close(y[..., 0], x[:, :, 0, 0][..., None])
    torch.testing.assert_close(y[..., 1], x[:, :, 0, 0][..., None])


def test_grid_sample_mojo_grid_with_cpu_input_raises(mojo_device):
    x = torch.randn(1, 2, 3, 4)
    grid = torch.zeros(1, 2, 2, 2, device=mojo_device)
    with pytest.raises(RuntimeError, match="expected input and grid to be on same"):
        torch.ops.aten.grid_sampler_2d(x, grid, 0, 0, False)


def test_grid_sample_strided_operands(mojo_device):
    x, grid = _grid_inputs((3, 4, 6, 8), (5, 7), torch.float32)
    xs = x.to(mojo_device).transpose(2, 3).contiguous().transpose(2, 3)[:, ::2]
    gs = grid.to(mojo_device).permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
    assert not xs.is_contiguous() and not gs.is_contiguous()
    for mode in _GRID_MODES_2D:
        want = F.grid_sample(x[:, ::2], grid, mode=mode, align_corners=False)
        got = F.grid_sample(xs, gs, mode=mode, align_corners=False)
        torch.testing.assert_close(got.cpu(), want, atol=1e-5, rtol=1e-5)
        go = torch.randn(want.shape)
        want_bw = torch.ops.aten.grid_sampler_2d_backward(
            go, x[:, ::2], grid, _GRID_MODES_2D.index(mode), 0, False, [True, True]
        )
        got_bw = torch.ops.aten.grid_sampler_2d_backward(
            go.to(mojo_device).transpose(2, 3).contiguous().transpose(2, 3),
            xs,
            gs,
            _GRID_MODES_2D.index(mode),
            0,
            False,
            [True, True],
        )
        for g, w in zip(got_bw, want_bw):
            torch.testing.assert_close(g.cpu(), w, atol=1e-4, rtol=1e-4)


def test_grid_sample_backward_output_mask(mojo_device):
    x, grid = _grid_inputs((2, 3, 5, 4), (3, 3), torch.float32)
    go = torch.randn(2, 3, 3, 3)
    with ran("aten::grid_sampler_2d_backward"):
        gi, gg = torch.ops.aten.grid_sampler_2d_backward(
            go.to(mojo_device),
            x.to(mojo_device),
            grid.to(mojo_device),
            0,
            0,
            False,
            [False, True],
        )
    assert gi is None
    want = torch.ops.aten.grid_sampler_2d_backward(
        go, x, grid, 0, 0, False, [True, True]
    )
    torch.testing.assert_close(gg.cpu(), want[1], atol=1e-5, rtol=1e-5)
    # Only the grid requires grad: F.grid_sample's autograd asks for it alone.
    gm = grid.to(mojo_device).requires_grad_(True)
    F.grid_sample(x.to(mojo_device), gm, align_corners=False).backward(
        go.to(mojo_device)
    )
    torch.testing.assert_close(gm.grad.cpu(), want[1], atol=1e-5, rtol=1e-5)


def test_grid_sample_empty(mojo_device):
    for shape, spatial in [
        ((0, 3, 4, 5), (2, 3)),
        ((2, 0, 4, 5), (2, 3)),
        ((2, 3, 4, 5), (0, 3)),
    ]:
        x, grid = _grid_inputs(shape, spatial, torch.float32)
        _grid_check(x, grid, mojo_device, "bilinear", "zeros", False, 0, 0)
    x, grid = _grid_inputs((2, 3, 4, 5, 2), (0, 3, 1), torch.float32)
    _grid_check(x, grid, mojo_device, "bilinear", "border", False, 0, 0)


def test_grid_sample_errors(mojo_device):
    x = torch.randn(2, 3, 4, 5, device=mojo_device)
    grid = torch.zeros(2, 3, 3, 2, device=mojo_device)
    with pytest.raises(RuntimeError, match="same batch size"):
        torch.grid_sampler(x, grid[:1], 0, 0, False)
    with pytest.raises(RuntimeError, match="to have size 2 in last dimension"):
        torch.grid_sampler(x, torch.zeros(2, 3, 3, 3, device=mojo_device), 0, 0, False)
    with pytest.raises(RuntimeError, match="non-empty spatial dimensions"):
        torch.grid_sampler(x[:, :, :0], grid, 0, 0, False)
    with pytest.raises(RuntimeError, match="expected 4D input and grid"):
        torch.ops.aten.grid_sampler_2d(
            x[None], torch.zeros(1, 2, 3, 3, 3, device=mojo_device), 0, 0, False
        )
    x5 = torch.randn(2, 3, 4, 5, 6, device=mojo_device)
    g5 = torch.zeros(2, 1, 2, 2, 3, device=mojo_device)
    with pytest.raises(RuntimeError, match="bicubic interpolation only supports 4D"):
        torch.ops.aten.grid_sampler_3d(x5, g5, 2, 0, False)
    with pytest.raises(
        RuntimeError, match="expected input and grid to be on same device"
    ):
        torch.ops.aten.grid_sampler_2d(x, grid.cpu(), 0, 0, False)
    with pytest.raises(RuntimeError, match="expected scalar type Float but found Half"):
        torch.ops.aten.grid_sampler_2d(x, grid.half(), 0, 0, False)
    with pytest.raises(NotImplementedError, match="not implemented for 'Long'"):
        torch.ops.aten.grid_sampler_2d(x.long(), grid.long(), 0, 0, False)
    with pytest.raises(RuntimeError, match="expected grad_output to have sizes"):
        torch.ops.aten.grid_sampler_2d_backward(
            torch.zeros(2, 3, 3, 4, device=mojo_device),
            x,
            grid,
            0,
            0,
            False,
            [True, True],
        )


def test_grid_sample_backward_alerts_nondeterminism(mojo_device):
    x, grid = _grid_inputs((1, 2, 4, 4), (2, 2), torch.float32)
    xm = x.to(mojo_device).requires_grad_(True)
    y = F.grid_sample(xm, grid.to(mojo_device), align_corners=False)
    with _deterministic():
        with pytest.raises(RuntimeError, match="grid_sampler_2d_backward_cuda"):
            y.sum().backward()


class _deterministic:
    def __enter__(self):
        self.prev = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)

    def __exit__(self, *exc):
        torch.use_deterministic_algorithms(self.prev)


def test_grid_sample_out_variants(mojo_device):
    x, grid = _grid_inputs((2, 3, 4, 5), (3, 2), torch.float32)
    want = torch.ops.aten.grid_sampler_2d(x, grid, 1, 2, True)
    out = torch.empty(0, device=mojo_device)
    with ran("aten::grid_sampler_2d.out"):
        torch.ops.aten.grid_sampler_2d.out(
            x.to(mojo_device), grid.to(mojo_device), 1, 2, True, out=out
        )
    torch.testing.assert_close(out.cpu(), want)
    go = torch.randn(want.shape)
    want_bw = torch.ops.aten.grid_sampler_2d_backward(
        go, x, grid, 0, 1, False, [True, True]
    )
    out0 = torch.empty(2, 3, 5, 4, device=mojo_device).transpose(-1, -2)
    out1 = torch.empty(0, device=mojo_device)
    with ran("aten::grid_sampler_2d_backward.out"):
        torch.ops.aten.grid_sampler_2d_backward.out(
            go.to(mojo_device),
            x.to(mojo_device),
            grid.to(mojo_device),
            0,
            1,
            False,
            [True, True],
            out0=out0,
            out1=out1,
        )
    torch.testing.assert_close(out0.cpu(), want_bw[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out1.cpu(), want_bw[1], atol=1e-5, rtol=1e-5)
    # The generated `copy_arg`: the exact result dtype, no cast; and an out
    # that overlaps itself is refused.
    args = (x.to(mojo_device), grid.to(mojo_device), 1, 2, True)
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
        torch.ops.aten.grid_sampler_2d.out(
            *args, out=torch.empty(0, device=mojo_device, dtype=torch.float16)
        )
    overlapping = torch.empty(1, device=mojo_device).expand(want.shape)
    with pytest.raises(RuntimeError, match="unsupported operation"):
        torch.ops.aten.grid_sampler_2d.out(*args, out=overlapping)


@pytest.mark.parametrize("op", ["_upsample_bilinear2d_aa", "_upsample_bicubic2d_aa"])
def test_upsample_aa_keeps_channels_last(mojo_device, op):
    """The antialiased meta allocates in the input's suggested format."""
    x = torch.randn(2, 3, 9, 8).to(memory_format=torch.channels_last)
    fn = getattr(torch.ops.aten, op)
    want = fn(x, [5, 6], False)
    got = fn(x.to(mojo_device), [5, 6], False)
    assert got.stride() == want.stride()
    torch.testing.assert_close(got.cpu(), want, atol=1e-5, rtol=1e-5)
