"""Native backend: indexing group (torch_mojo_backend/mojo/tmb/ops/indexing.mojo).

flip / roll / unfold / channel_shuffle / take / put_ / index_fill_ /
index_copy / masked_scatter_ / repeat_interleave.Tensor, plus the ATen
composites that reach them (fliplr, flipud, rot90, fft_fftshift,
fft_ifftshift, unfold_copy, put, index_fill, masked_scatter, the
repeat_interleave self overloads). Public torch API only, compared against
CPU torch; `ran` confirms the native kernel is what ran.
"""

import pytest
import torch

from tests.native.conftest import ran

DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool]


def _make(shape: tuple[int, ...], dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    """Deterministic values that differ element to element (bool: a mix)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g) * 10
    if dtype == torch.bool:
        return x > 0
    return x.to(dtype)


def _check(result: torch.Tensor, expected: torch.Tensor):
    assert result.device.type == "mojo"
    assert result.dtype == expected.dtype
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# flip and its composites
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dims", [(0,), (1,), (-1,), (0, 2), (0, 1, 2), ()])
def test_flip(mojo_device, dtype, dims):
    x = _make((4, 5, 6), dtype)
    with ran("aten::flip"):
        _check(torch.flip(x.to(mojo_device), dims), torch.flip(x, dims))


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_flip_strided_and_edge_shapes(mojo_device, dtype):
    x = _make((4, 5, 6), dtype)
    xt = x.to(mojo_device).transpose(0, 2)
    _check(torch.flip(xt, (0, 1)), torch.flip(x.transpose(0, 2), (0, 1)))
    _check(torch.flip(xt[::2], (0,)), torch.flip(x.transpose(0, 2)[::2], (0,)))
    s = _make((), dtype)
    _check(torch.flip(s.to(mojo_device), (0,)), torch.flip(s, (0,)))
    e = _make((3, 0, 2), dtype)
    _check(torch.flip(e.to(mojo_device), (0, 1)), torch.flip(e, (0, 1)))
    with pytest.raises(RuntimeError, match="appears multiple times"):
        torch.flip(x.to(mojo_device), (0, -3))


@pytest.mark.parametrize("dtype", DTYPES)
def test_flip_composites(mojo_device, dtype):
    x = _make((4, 5, 6), dtype)
    d = x.to(mojo_device)
    _check(torch.fliplr(d), torch.fliplr(x))
    _check(torch.flipud(d), torch.flipud(x))
    for k in (-1, 1, 2, 3):
        _check(torch.rot90(d, k, (1, 2)), torch.rot90(x, k, (1, 2)))


# ---------------------------------------------------------------------------
# roll and fftshift
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("shifts", "dims"),
    [((1,), (0,)), ((-7,), (1,)), ((2, 3), (0, 2)), ((5,), ()), ((1, 1), (0, 0))],
)
def test_roll(mojo_device, dtype, shifts, dims):
    x = _make((4, 5, 6), dtype)
    with ran("aten::roll"):
        got = torch.roll(x.to(mojo_device), shifts, dims)
    _check(got, torch.roll(x, shifts, dims))


def test_roll_strided_empty_scalar(mojo_device):
    x = _make((4, 5, 6), torch.float32)
    xt = x.transpose(0, 2)
    dt = x.to(mojo_device).transpose(0, 2)
    _check(torch.roll(dt, (3, -2, 1), (0, 1, 2)), torch.roll(xt, (3, -2, 1), (0, 1, 2)))
    _check(torch.roll(dt, 7), torch.roll(xt, 7))
    s = torch.tensor(3.0)
    _check(torch.roll(s.to(mojo_device), 1), torch.roll(s, 1))
    e = torch.zeros(3, 0, 2)
    _check(torch.roll(e.to(mojo_device), 1, 0), torch.roll(e, 1, 0))
    with pytest.raises(RuntimeError, match="shifts and dimensions must align"):
        torch.roll(x.to(mojo_device), (1, 2), (0,))


@pytest.mark.parametrize("dtype", DTYPES)
def test_fftshift(mojo_device, dtype):
    x = _make((5, 6, 7), dtype)
    d = x.to(mojo_device)
    _check(torch.fft.fftshift(d), torch.fft.fftshift(x))
    _check(torch.fft.ifftshift(d, dim=(0, 2)), torch.fft.ifftshift(x, dim=(0, 2)))


# ---------------------------------------------------------------------------
# unfold (a view) and unfold_copy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("dim", "size", "step"), [(0, 2, 1), (1, 3, 2), (2, 6, 1), (-1, 1, 5), (2, 0, 1)]
)
def test_unfold(mojo_device, dtype, dim, size, step):
    x = _make((4, 5, 6), dtype)
    d = x.to(mojo_device)
    with ran("aten::unfold"):
        got = d.unfold(dim, size, step)
    _check(got, x.unfold(dim, size, step))
    # A strided self keeps its extents (4, 6, 5 -> transposed (4, 6, 5)).
    xt = x.transpose(1, 2).contiguous().transpose(1, 2)
    dt = xt.to(mojo_device).transpose(1, 2).contiguous().transpose(1, 2)
    _check(dt.unfold(dim, size, step), xt.unfold(dim, size, step))


def test_unfold_is_a_view(mojo_device):
    x = torch.arange(10.0)
    d = x.to(mojo_device)
    view = d.unfold(0, 3, 2)
    assert view._base is d
    view[1, 0] = -1.0
    assert d.cpu()[2].item() == -1.0
    s = torch.tensor(4.0)
    _check(s.to(mojo_device).unfold(0, 1, 1), s.unfold(0, 1, 1))
    copy = torch.ops.aten.unfold_copy(d, 0, 4, 3)
    _check(copy, torch.ops.aten.unfold_copy(d.cpu(), 0, 4, 3))
    with pytest.raises(RuntimeError, match="maximum size for tensor at dimension"):
        d.unfold(0, 11, 1)
    with pytest.raises(RuntimeError, match="step is 0 but must be > 0"):
        d.unfold(0, 2, 0)


@pytest.mark.parametrize(
    ("shape", "dim", "size", "step"),
    [((10,), 0, 4, 3), ((10,), 0, 3, 3), ((3, 11, 2), 1, 4, 1), ((3, 11, 2), -2, 5, 2), ((), 0, 1, 1)],
)
def test_unfold_backward(mojo_device, shape, dim, size, step):
    x = torch.randn(shape)
    grad = torch.randn(x.unfold(dim, size, step).shape)
    expected = torch.ops.aten.unfold_backward(grad, list(shape), dim, size, step)
    with ran("aten::unfold_backward"):
        got = torch.ops.aten.unfold_backward(grad.to(mojo_device), list(shape), dim, size, step)
    torch.testing.assert_close(got.cpu(), expected)
    d = x.to(mojo_device).requires_grad_()
    d.unfold(dim, size, step).backward(grad.to(mojo_device))
    assert d.grad is not None
    torch.testing.assert_close(d.grad.cpu(), expected)


# ---------------------------------------------------------------------------
# channel_shuffle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("shape", "groups"),
    [((1, 4, 10, 10), 2), ((2, 6, 8, 8), 3), ((2, 8, 5), 4), ((3, 6, 1, 2, 2), 6)],
)
def test_channel_shuffle(mojo_device, dtype, shape, groups):
    x = _make(shape, dtype)
    with ran("aten::channel_shuffle"):
        got = torch.nn.functional.channel_shuffle(x.to(mojo_device), groups)
    _check(got, torch.nn.functional.channel_shuffle(x, groups))


def test_channel_shuffle_errors(mojo_device):
    d = torch.zeros(2, 6, 3, device=mojo_device)
    with pytest.raises(RuntimeError, match="divisible by groups"):
        torch.nn.functional.channel_shuffle(d, 4)
    with pytest.raises(RuntimeError, match="must be positive"):
        torch.nn.functional.channel_shuffle(d, 0)
    with pytest.raises(RuntimeError, match="at least 3 dimensions"):
        torch.nn.functional.channel_shuffle(torch.zeros(2, 6, device=mojo_device), 2)


# ---------------------------------------------------------------------------
# take / put_
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_take(mojo_device, dtype):
    x = _make((4, 5, 6), dtype)
    idx = torch.tensor([[0, 5], [-1, 119], [-120, 7]])
    di = idx.to(mojo_device)
    with ran("aten::take"):
        got = torch.take(x.to(mojo_device), di)
    _check(got, torch.take(x, idx))
    xt = x.transpose(0, 2)
    _check(torch.take(xt.to(mojo_device), di), torch.take(xt, idx))
    s = _make((), dtype)
    i0 = torch.tensor(0)
    _check(torch.take(s.to(mojo_device), i0.to(mojo_device)), torch.take(s, i0))
    empty = torch.zeros(0, dtype=torch.int64)
    _check(torch.take(x.to(mojo_device), empty.to(mojo_device)), torch.take(x, empty))


def test_take_out_and_errors(mojo_device):
    x = _make((3, 4), torch.float32)
    idx = torch.tensor([1, -1, 5])
    d, di = x.to(mojo_device), idx.to(mojo_device)
    out = torch.empty(0, device=mojo_device)
    with ran("aten::take.out"):
        torch.take(d, di, out=out)
    _check(out, torch.take(x, idx))
    with pytest.raises(RuntimeError, match="Expected a long tensor for index"):
        torch.take(d, di.int())
    flat = d.flatten()
    with pytest.raises(RuntimeError, match="unsupported operation"):
        torch.take(flat, di, out=flat[:3])
    with pytest.raises(RuntimeError, match="tried to take from an empty tensor"):
        torch.take(torch.zeros(0, device=mojo_device), di)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("accumulate", [False, True])
def test_put(mojo_device, dtype, accumulate):
    x = _make((4, 5), dtype)
    idx = torch.tensor([3, -2, 7, 19])
    src = _make((2, 2), dtype, seed=1)
    di, ds = idx.to(mojo_device), src.to(mojo_device)
    d = x.to(mojo_device)
    with ran("aten::put_"):
        d.put_(di, ds, accumulate=accumulate)
    _check(d, x.clone().put_(idx, src, accumulate=accumulate))
    # A strided self is written where it lives.
    dt = x.to(mojo_device).t()
    dt.put_(di, ds, accumulate=accumulate)
    _check(dt, x.clone().t().put_(idx, src, accumulate=accumulate))
    got = torch.put(x.to(mojo_device), di, ds, accumulate)
    _check(got, torch.put(x, idx, src, accumulate))


def test_put_accumulate_duplicates_and_errors(mojo_device):
    x = torch.zeros(5)
    idx = torch.tensor([1, 1, 1, 4])
    src = torch.tensor([1.0, 2.0, 3.0, 4.0])
    d = x.to(mojo_device)
    d.put_(idx.to(mojo_device), src.to(mojo_device), accumulate=True)
    _check(d, x.clone().put_(idx, src, accumulate=True))
    with pytest.raises(RuntimeError, match="same number of elements"):
        d.put_(idx.to(mojo_device), src[:2].to(mojo_device))
    with pytest.raises(RuntimeError, match="index out of range"):
        d.put_(torch.tensor([5], device=mojo_device), torch.ones(1, device=mojo_device))


# ---------------------------------------------------------------------------
# index_fill_ / index_copy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_index_fill(mojo_device, dtype, dim):
    x = _make((4, 5, 6), dtype)
    idx = torch.tensor([0, 2, -1])
    di = idx.to(mojo_device)
    value = True if dtype == torch.bool else 3
    with ran("aten::index_fill_.int_Scalar"):
        got = x.to(mojo_device).index_fill(dim, di, value)
    _check(got, x.index_fill(dim, idx, value))
    # In place through a transposed self, and the 0-d value tensor overload.
    dt = x.to(mojo_device).transpose(0, 1)
    dt.index_fill_(dim, di, value)
    _check(dt, x.clone().transpose(0, 1).index_fill_(dim, idx, value))
    v = torch.tensor(value, dtype=dtype)
    with ran("aten::index_fill_.int_Tensor"):
        got = x.to(mojo_device).index_fill(dim, di, v.to(mojo_device))
    _check(got, x.index_fill(dim, idx, v))


def test_index_fill_scalar_conversions(mojo_device):
    x = _make((3, 4), torch.int64)
    idx = torch.tensor([1])
    di = idx.to(mojo_device)
    _check(x.to(mojo_device).index_fill(1, di, 2.7), x.index_fill(1, idx, 2.7))
    s = torch.tensor(5.0)
    i0 = torch.tensor(0)
    _check(s.to(mojo_device).index_fill(0, i0.to(mojo_device), -1.0), s.index_fill(0, i0, -1.0))
    with pytest.raises(RuntimeError, match="Expected dtype int64 for index"):
        x.to(mojo_device).index_fill(0, di.int(), 1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_index_copy(mojo_device, dtype, dim):
    x = _make((4, 5, 6), dtype)
    idx = torch.tensor([3, 0, 1])
    shape = list(x.shape)
    shape[dim] = 3
    src = _make(tuple(shape), dtype, seed=1)
    d, di, ds = x.to(mojo_device), idx.to(mojo_device), src.to(mojo_device)
    expected = x.index_copy(dim, idx, src)
    with ran("aten::index_copy"):
        got = d.index_copy(dim, di, ds)
    _check(got, expected)
    inplace = x.to(mojo_device)
    with ran("aten::index_copy_"):
        inplace.index_copy_(dim, di, ds)
    _check(inplace, expected)
    out = torch.empty(0, dtype=dtype, device=mojo_device)
    strided_src = ds.transpose(0, 2).contiguous().transpose(0, 2)
    with ran("aten::index_copy.out"):
        torch.index_copy(d, dim, di, strided_src, out=out)
    _check(out, expected)


def test_index_copy_scalar_and_errors(mojo_device):
    s = torch.tensor(1.0)
    i0 = torch.tensor([0])
    v = torch.tensor(9.0)
    got = s.to(mojo_device).index_copy(0, i0.to(mojo_device), v.to(mojo_device))
    _check(got, s.index_copy(0, i0, v))
    d = torch.zeros(3, 4, device=mojo_device)
    two = torch.tensor([0, 1], device=mojo_device)
    with pytest.raises(RuntimeError, match="Number of indices"):
        d.index_copy(0, two, torch.ones(3, 4, device=mojo_device))
    with pytest.raises(RuntimeError, match="same slice shapes"):
        d.index_copy(0, two[:1], torch.ones(1, 5, device=mojo_device))
    with pytest.raises(RuntimeError, match="index out of range"):
        d.index_copy(0, two[:1] + 3, torch.ones(1, 4, device=mojo_device))


# ---------------------------------------------------------------------------
# masked_scatter_
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mask_shape", [(4, 5, 6), (6,), (5, 1), ()])
def test_masked_scatter(mojo_device, dtype, mask_shape):
    x = _make((4, 5, 6), dtype)
    mask = torch.randn(mask_shape, generator=torch.Generator().manual_seed(2)) > 0
    src = _make((200,), dtype, seed=1)
    with ran("aten::masked_scatter_"):
        got = x.to(mojo_device).masked_scatter(mask.to(mojo_device), src.to(mojo_device))
    _check(got, x.masked_scatter(mask, src))


def test_masked_scatter_strided_broadcast_and_errors(mojo_device):
    x = _make((4, 5, 6), torch.float32)
    mask = torch.randn(4, 5, 6, generator=torch.Generator().manual_seed(3)) > 0
    src = _make((6, 5, 4), torch.float32, seed=1).transpose(0, 2)
    dm, ds = mask.to(mojo_device), src.to(mojo_device)
    dt = x.to(mojo_device).transpose(0, 2)
    dt.masked_scatter_(dm.transpose(0, 2), ds)
    _check(dt, x.clone().transpose(0, 2).masked_scatter_(mask.transpose(0, 2), src))
    # The out-of-place form broadcasts self against the mask.
    small = _make((6,), torch.float32)
    _check(small.to(mojo_device).masked_scatter(dm, ds), small.masked_scatter(mask, src))
    d = x.to(mojo_device)
    with pytest.raises(RuntimeError, match="only supports boolean masks"):
        d.masked_scatter_(dm.to(torch.uint8), ds)
    with pytest.raises(RuntimeError, match="same dtypes"):
        d.masked_scatter_(dm, ds.to(torch.float16))


# ---------------------------------------------------------------------------
# repeat_interleave
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("repeats", [[1, 0, 3, 2], [0, 0], [], [5]])
def test_repeat_interleave_tensor(mojo_device, dtype, repeats):
    r = torch.tensor(repeats, dtype=dtype)
    with ran("aten::repeat_interleave.Tensor"):
        got = torch.repeat_interleave(r.to(mojo_device))
    _check(got, torch.repeat_interleave(r))
    got = torch.repeat_interleave(r.to(mojo_device), output_size=sum(repeats))
    _check(got, torch.repeat_interleave(r, output_size=sum(repeats)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_repeat_interleave_self(mojo_device, dtype):
    x = _make((4, 3), dtype)
    r = torch.tensor([1, 2, 0, 3])
    d = x.to(mojo_device)
    got = torch.repeat_interleave(d, r.to(mojo_device), dim=0)
    _check(got, torch.repeat_interleave(x, r, dim=0))
    _check(torch.repeat_interleave(d, 2), torch.repeat_interleave(x, 2))
    _check(torch.repeat_interleave(d, 2, dim=1), torch.repeat_interleave(x, 2, dim=1))


def test_repeat_interleave_errors(mojo_device):
    with pytest.raises(RuntimeError, match="repeats can not be negative"):
        torch.repeat_interleave(torch.tensor([1, -1], device=mojo_device))
    with pytest.raises(RuntimeError, match="1D vector"):
        torch.repeat_interleave(torch.ones(2, 2, dtype=torch.int64, device=mojo_device))
