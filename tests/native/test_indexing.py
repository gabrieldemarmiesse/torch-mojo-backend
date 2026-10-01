"""Native backend: indexing group (torch_mojo_backend/mojo/tmb/ops/indexing.mojo).

flip / roll / unfold / channel_shuffle / take / put_ / index_fill_ /
index_copy / masked_scatter_ / repeat_interleave.Tensor, plus the ATen
composites that reach them (fliplr, flipud, rot90, fft_fftshift,
fft_ifftshift, unfold_copy, put, index_fill, masked_scatter, the
repeat_interleave self overloads). Public torch API only, compared against
CPU torch; `ran` confirms the native kernel is what ran.
"""

import numpy as np
import pytest
import torch

from tests.native.conftest import is_metal, ran

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
    [
        ((10,), 0, 4, 3),
        ((10,), 0, 3, 3),
        ((3, 11, 2), 1, 4, 1),
        ((3, 11, 2), -2, 5, 2),
        ((), 0, 1, 1),
    ],
)
def test_unfold_backward(mojo_device, shape, dim, size, step):
    x = torch.randn(shape)
    grad = torch.randn(x.unfold(dim, size, step).shape)
    expected = torch.ops.aten.unfold_backward(grad, list(shape), dim, size, step)
    with ran("aten::unfold_backward"):
        got = torch.ops.aten.unfold_backward(
            grad.to(mojo_device), list(shape), dim, size, step
        )
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


def test_out_overloads_reject_internal_overlap(mojo_device):
    """Every out= of this batch refuses an out whose elements alias each
    other (copy_'s / the structured kernels' assert_no_internal_overlap)."""

    def expanded(*shape: int) -> torch.Tensor:
        return torch.empty(1, device=mojo_device).expand(*shape)

    d = torch.randn(3, 4, device=mojo_device)
    mask = torch.randn(3, 4, device=mojo_device) > 0
    i = torch.tensor([0, 2], device=mojo_device)
    src = torch.ones(2, 4, device=mojo_device)
    value = torch.tensor(1.0, device=mojo_device)
    cases = [
        lambda: torch.linspace(0, 1, 3, out=expanded(3)),
        lambda: torch.logspace(0, 1, 3, out=expanded(3)),
        lambda: torch.eye(3, out=expanded(3, 3)),
        lambda: torch.eye(3, 4, out=expanded(3, 4)),
        lambda: torch.take(d, i, out=expanded(2)),
        lambda: torch.index_copy(d, 0, i, src, out=expanded(3, 4)),
        lambda: torch.ops.aten.masked_fill.Scalar_out(d, mask, 1.0, out=expanded(3, 4)),
        lambda: torch.ops.aten.masked_fill.Tensor_out(
            d, mask, value, out=expanded(3, 4)
        ),
    ]
    for case in cases:
        with pytest.raises(
            RuntimeError, match="more than one element of the written-to"
        ):
            case()


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
    # out= an empty view of self's storage: the resize grows that storage.
    base = x.flatten().to(mojo_device)
    out = base[:0]
    torch.take(base, di, out=out)
    _check(out, torch.take(x.flatten(), idx))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("accumulate", [False, True])
def test_put(mojo_device, dtype, accumulate):
    if (
        accumulate
        and is_metal(mojo_device)
        and dtype not in (torch.float32, torch.bool)
    ):
        # Apple GPUs have no 16- or 64-bit atomic add: declined.
        with pytest.raises(NotImplementedError, match="Apple GPU"):
            torch.zeros(4, dtype=dtype, device=mojo_device).put_(
                torch.tensor([0], device=mojo_device),
                torch.ones(1, dtype=dtype, device=mojo_device),
                accumulate=True,
            )
        return
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
    _check(
        s.to(mojo_device).index_fill(0, i0.to(mojo_device), -1.0),
        s.index_fill(0, i0, -1.0),
    )
    with pytest.raises(RuntimeError, match="Expected dtype int64 for index"):
        x.to(mojo_device).index_fill(0, di.int(), 1)
    u = torch.zeros(3, 4, dtype=torch.uint8, device=mojo_device)
    with pytest.raises(RuntimeError, match="without overflow"):
        u.index_fill(1, di, 256.0)
    # The value is range-checked even when nothing is filled.
    for target, index in [(u, di[:0]), (u[:0], di)]:
        with pytest.raises(RuntimeError, match="without overflow"):
            target.clone().index_fill_(1, index, 256)
    # An empty filled dimension rejects every index; an empty other one is a
    # no-op.
    e = torch.empty(0, 3, device=mojo_device)
    with pytest.raises(RuntimeError, match="out of bounds for dimension 0 with size 0"):
        e.index_fill(0, torch.tensor([0], device=mojo_device), 2)
    assert e.index_fill(1, torch.tensor([0], device=mojo_device), 2).shape == (0, 3)


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
    # out= partially overlapping self is refused (copy_'s partial-overlap
    # check); out= being self itself is fine.
    base = torch.zeros(5, 4, device=mojo_device)
    with pytest.raises(RuntimeError, match="unsupported operation"):
        torch.index_copy(
            base[:-1], 0, two[:1], torch.ones(1, 4, device=mojo_device), out=base[1:]
        )
    torch.index_copy(base, 0, two[:1], torch.ones(1, 4, device=mojo_device), out=base)
    assert base.cpu()[0].tolist() == [1.0] * 4
    # out= on another device is refused before any kernel runs.
    with pytest.raises(RuntimeError, match="Expected out tensor to have device"):
        torch.index_copy(
            d, 0, two[:1], torch.ones(1, 4, device=mojo_device), out=torch.empty(3, 4)
        )
    # A scalar self bounds-checks every index (index 1 of a size-1 dim).
    scalar = torch.tensor(1.0, device=mojo_device)
    with pytest.raises(RuntimeError, match="index out of range"):
        scalar.index_copy(0, two, torch.tensor([5.0, 6.0], device=mojo_device))
    # An empty destination dimension rejects every index.
    empty = torch.zeros(0, 3, device=mojo_device)
    with pytest.raises(RuntimeError, match="out of bounds for dimension 0 with size 0"):
        empty.index_copy(0, two[:1], torch.ones(1, 3, device=mojo_device))


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
        got = x.to(mojo_device).masked_scatter(
            mask.to(mojo_device), src.to(mojo_device)
        )
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
    _check(
        small.to(mojo_device).masked_scatter(dm, ds), small.masked_scatter(mask, src)
    )
    d = x.to(mojo_device)
    with pytest.raises(RuntimeError, match="only supports boolean masks"):
        d.masked_scatter_(dm.to(torch.uint8), ds)
    with pytest.raises(RuntimeError, match="same dtypes"):
        d.masked_scatter_(dm, ds.to(torch.float16))
    # Fewer source elements than selected positions raises, empty source
    # included; an all-false mask with an empty source is a no-op.
    with pytest.raises(RuntimeError, match="Number of elements of source"):
        d.masked_scatter_(dm, ds.flatten()[: int(mask.sum()) - 1])
    with pytest.raises(RuntimeError, match="Number of elements of source"):
        d.masked_scatter_(dm, torch.empty(0, device=mojo_device))
    none = torch.zeros_like(dm)
    _check(d.clone().masked_scatter_(none, torch.empty(0, device=mojo_device)), x)


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
    r = torch.tensor([1, 2], device=mojo_device)
    with pytest.raises(RuntimeError, match="output_size"):
        torch.repeat_interleave(r, output_size=2)
    with pytest.raises(RuntimeError, match="repeats can not be negative"):
        torch.repeat_interleave(
            torch.tensor([3, -1], device=mojo_device), output_size=2
        )


# ---------------------------------------------------------------------------
# scatter(reduce=) / scatter_reduce / index_reduce
# ---------------------------------------------------------------------------

REDUCE_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.bool,
]


def _small(shape: tuple[int, ...], dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    """Small integers (exact in every dtype, products included): duplicate
    indices then reduce to the same value in any order."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-2, 3, shape, generator=g)
    if dtype == torch.bool:
        return x > 0
    return x.to(dtype)


@pytest.mark.parametrize("dtype", REDUCE_DTYPES)
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_scatter_reduce(mojo_device, dtype, reduce, include_self, dim):
    if dtype == torch.bool and reduce == "mean":
        pytest.skip("mean of bool is declined (CUDA has no bool scatter_reduce)")
    x = _small((5, 7, 3), dtype, 1)
    src = _small((5, 7, 3), dtype, 2)
    g = torch.Generator().manual_seed(3)
    # Many collisions: every target is hit several times.
    index = torch.randint(0, x.shape[dim], (4, 6, 3), generator=g)
    expected = x.scatter_reduce(dim, index, src, reduce, include_self=include_self)
    with ran("aten::scatter_reduce.two"):
        got = x.to(mojo_device).scatter_reduce(
            dim,
            index.to(mojo_device),
            src.to(mojo_device),
            reduce,
            include_self=include_self,
        )
    _check(got, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduce", ["amax", "amin", "prod", "sum"])
def test_scatter_reduce_nan_inf_and_float_values(mojo_device, dtype, reduce):
    x = torch.tensor([1.5, -0.0, float("inf"), 2.0, -3.0, 0.0], dtype=dtype)
    src = torch.tensor(
        [float("nan"), 4.0, -float("inf"), 0.5, 2.0, -2.0, 1.0, 3.0], dtype=dtype
    )
    index = torch.tensor([0, 0, 2, 3, 3, 4, 5, 5])
    for include_self in (True, False):
        expected = x.scatter_reduce(0, index, src, reduce, include_self=include_self)
        got = x.to(mojo_device).scatter_reduce(
            0,
            index.to(mojo_device),
            src.to(mojo_device),
            reduce,
            include_self=include_self,
        )
        torch.testing.assert_close(got.cpu(), expected, equal_nan=True)


def test_scatter_reduce_out_in_place_and_edges(mojo_device):
    x = _small((4, 5), torch.float32)
    src = _small((4, 5), torch.float32, 1)
    index = torch.tensor([[0, 1, 2, 3, 0], [3, 2, 1, 0, 0]])
    expected = x.scatter_reduce(0, index, src, "amax", include_self=False)
    d = x.to(mojo_device)
    # A non-contiguous out of the right shape is written where it lives.
    base = torch.zeros(5, 4, device=mojo_device)
    out = base.t()
    with ran("aten::scatter_reduce.two_out"):
        torch.scatter_reduce(
            d,
            0,
            index.to(mojo_device),
            src.to(mojo_device),
            "amax",
            include_self=False,
            out=out,
        )
    _check(out, expected)
    # A wrongly sized out is resized.
    out = torch.empty(0, device=mojo_device)
    torch.scatter_reduce(
        d,
        0,
        index.to(mojo_device),
        src.to(mojo_device),
        "amax",
        include_self=False,
        out=out,
    )
    _check(out, expected)
    # In place.
    y = x.to(mojo_device)
    with ran("aten::scatter_reduce_.two"):
        y.scatter_reduce_(
            0, index.to(mojo_device), src.to(mojo_device), "amax", include_self=False
        )
    _check(y, expected)
    # int32 index, empty index, 0-d tensors.
    i32 = index.to(torch.int32)
    _check(
        d.scatter_reduce(0, i32.to(mojo_device), src.to(mojo_device), "sum"),
        x.scatter_reduce(0, i32, src, "sum"),
    )
    empty = torch.empty(0, 5, dtype=torch.int64)
    _check(
        d.scatter_reduce(0, empty.to(mojo_device), src.to(mojo_device), "prod"),
        x.scatter_reduce(0, empty, src, "prod"),
    )
    s = torch.tensor(3.0)
    zero = torch.tensor(0)
    _check(
        s.to(mojo_device).scatter_reduce(
            0, zero.to(mojo_device), torch.tensor(5.0).to(mojo_device), "amax"
        ),
        s.scatter_reduce(0, zero, torch.tensor(5.0), "amax"),
    )


def test_scatter_reduce_errors(mojo_device):
    d = torch.zeros(3, 4, device=mojo_device)
    src = torch.ones(3, 4, device=mojo_device)
    index = torch.zeros(3, 4, dtype=torch.int64, device=mojo_device)
    with pytest.raises(RuntimeError, match="reduce argument must be either sum"):
        d.scatter_reduce(0, index, src, "max_")
    with pytest.raises(
        RuntimeError, match="reduce argument must be either add or multiply"
    ):
        d.scatter(0, index, src, reduce="sum")
    bad = index.clone()
    bad[1, 2] = 3
    with pytest.raises(RuntimeError, match="index out of range"):
        d.scatter_reduce(0, bad, src, "prod")
    with pytest.raises(
        RuntimeError, match="Expected self.dtype to be equal to src.dtype"
    ):
        d.scatter_reduce(0, index, src.half(), "sum")
    with pytest.raises(RuntimeError, match="int32/int64"):
        d.scatter_reduce(0, index.float(), src, "sum")


@pytest.mark.parametrize("dtype", REDUCE_DTYPES)
@pytest.mark.parametrize("reduce", ["add", "multiply"])
def test_scatter_legacy_reduce(mojo_device, dtype, reduce):
    x = _small((5, 6), dtype)
    src = _small((5, 6), dtype, 1)
    index = torch.randint(0, 5, (4, 6), generator=torch.Generator().manual_seed(1))
    d = x.to(mojo_device)
    di = index.to(mojo_device)
    with ran("aten::scatter.reduce"):
        got = d.scatter(0, di, src.to(mojo_device), reduce=reduce)
    _check(got, x.scatter(0, index, src, reduce=reduce))
    value = True if dtype == torch.bool else 2
    with ran("aten::scatter.value_reduce"):
        got = d.scatter(1, di[:, :3], value, reduce=reduce)
    _check(got, x.scatter(1, index[:, :3], value, reduce=reduce))
    y = x.to(mojo_device)
    y.scatter_(0, di, src.to(mojo_device), reduce=reduce)
    _check(y, x.scatter(0, index, src, reduce=reduce))
    y = x.to(mojo_device)
    y.scatter_(0, di, value, reduce=reduce)
    _check(y, x.scatter(0, index, value, reduce=reduce))
    out = torch.empty(0, dtype=dtype, device=mojo_device)
    torch.scatter(d, 0, di, src.to(mojo_device), reduce=reduce, out=out)
    _check(out, x.scatter(0, index, src, reduce=reduce))
    out = torch.empty(0, dtype=dtype, device=mojo_device)
    torch.scatter(d, 0, di, value, reduce=reduce, out=out)
    _check(out, x.scatter(0, index, value, reduce=reduce))


@pytest.mark.parametrize("dtype", DTYPES)
def test_scatter_value_in_place_and_out(mojo_device, dtype):
    x = _make((4, 5), dtype)
    index = torch.tensor([[0, 1, 2, 3, 0], [3, 2, 1, 0, 1]])
    value = False if dtype == torch.bool else -3
    y = x.to(mojo_device)
    with ran("aten::scatter_.value"):
        y.scatter_(0, index.to(mojo_device), value)
    _check(y, x.scatter(0, index, value))
    out = torch.empty(2, 2, dtype=dtype, device=mojo_device)
    with ran("aten::scatter.value_out"):
        torch.scatter(x.to(mojo_device), 0, index.to(mojo_device), value, out=out)
    _check(out, x.scatter(0, index, value))
    with pytest.raises(RuntimeError, match="value cannot be converted"):
        torch.zeros(3, dtype=torch.int8, device=mojo_device).scatter_(
            0, torch.tensor([0], device=mojo_device), 300
        )


def test_one_hot(mojo_device):
    labels = torch.tensor([[0, 3], [2, 1], [3, 3]])
    with ran("aten::scatter_.value", "aten::scatter.value_out", "aten::scatter.value"):
        got = torch.nn.functional.one_hot(labels.to(mojo_device), 5)
    _check(got, torch.nn.functional.one_hot(labels, 5))


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64]
)
@pytest.mark.parametrize("reduce", ["prod", "mean", "amax", "amin"])
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("n_index", [6, 40])
def test_index_reduce(mojo_device, dtype, reduce, include_self, n_index):
    """6 indices take the ordered small-index route (CUDA's
    indexFuncSmallIndex: each slot reduced in index order, so even rounded
    floats match CPU); 40 take the atomic one (exact values only)."""
    g = torch.Generator().manual_seed(4)
    x = _small((5, 6, 3), dtype, 5)
    index = torch.randint(0, 6, (n_index,), generator=g)
    if n_index <= 16 and dtype != torch.int64:
        source = (torch.randn(5, n_index, 3, generator=g) * 3).to(dtype)
    else:
        source = _small((5, n_index, 3), dtype, 6)
    expected = x.index_reduce(1, index, source, reduce, include_self=include_self)
    with ran("aten::index_reduce"):
        got = x.to(mojo_device).index_reduce(
            1,
            index.to(mojo_device),
            source.to(mojo_device),
            reduce,
            include_self=include_self,
        )
    _check(got, expected)


def test_index_reduce_out_in_place_and_errors(mojo_device):
    x = torch.randn(4, 3)
    source = torch.randn(5, 3)
    index = torch.tensor([0, 3, 0, 1, 3], dtype=torch.int32)
    expected = x.index_reduce(0, index, source, "amax")
    y = x.to(mojo_device)
    with ran("aten::index_reduce_"):
        y.index_reduce_(0, index.to(mojo_device), source.to(mojo_device), "amax")
    _check(y, expected)
    out = torch.empty(0, device=mojo_device)
    with ran("aten::index_reduce.out"):
        torch.index_reduce(
            x.to(mojo_device),
            0,
            index.to(mojo_device),
            source.to(mojo_device),
            "amax",
            out=out,
        )
    _check(out, expected)
    d = x.to(mojo_device)
    with pytest.raises(RuntimeError, match="Expected reduce to be one of"):
        d.index_reduce(0, index.to(mojo_device), source.to(mojo_device), "sum")
    with pytest.raises(IndexError, match="Index is supposed to be a vector"):
        d.index_reduce(
            0, index.reshape(1, 5).to(mojo_device), source.to(mojo_device), "prod"
        )
    with pytest.raises(RuntimeError, match="should be equal to source.size"):
        d.index_reduce(0, index[:3].to(mojo_device), source.to(mojo_device), "prod")
    with pytest.raises(RuntimeError, match="index out of range"):
        d.index_reduce(
            0,
            torch.tensor([0, 1, 9, 1, 2], device=mojo_device),
            source.to(mojo_device),
            "prod",
        )


# ---------------------------------------------------------------------------
# nonzero.out / nonzero_static / index.Tensor_out / narrow_copy.out /
# fill_.Tensor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_nonzero_out_and_static(mojo_device, dtype):
    x = _make((3, 4), dtype)
    x[1] = 0
    out = torch.empty(7, 1, dtype=torch.int64, device=mojo_device)
    with ran("aten::nonzero.out"):
        torch.nonzero(x.to(mojo_device), out=out)
    _check(out, torch.nonzero(x))
    for size in (0, 3, 20):
        with ran("aten::nonzero_static"):
            got = torch.nonzero_static(x.to(mojo_device), size=size, fill_value=-4)
        _check(got, torch.nonzero_static(x, size=size, fill_value=-4))
    out = torch.empty(0, dtype=torch.int64, device=mojo_device)
    torch.nonzero_static(x.to(mojo_device), size=5, out=out)
    _check(out, torch.nonzero_static(x, size=5))
    s = torch.tensor(2.0)
    _check(
        torch.nonzero_static(s.to(mojo_device), size=2), torch.nonzero_static(s, size=2)
    )
    with pytest.raises(RuntimeError, match="non-negative"):
        torch.nonzero_static(x.to(mojo_device), size=-1)
    with pytest.raises(RuntimeError, match="scalar type Long"):
        torch.nonzero(x.to(mojo_device), out=torch.empty(0, device=mojo_device))


def test_index_tensor_out(mojo_device):
    x = _make((5, 4), torch.float32)
    idx = torch.tensor([3, 0, 3])
    out = torch.empty(0, device=mojo_device)
    with ran("aten::index.Tensor_out"):
        torch.ops.aten.index.Tensor_out(
            x.to(mojo_device), [idx.to(mojo_device)], out=out
        )
    _check(out, x[idx])
    out = torch.empty(0, device=mojo_device)
    mask = torch.tensor([True, False, True, False, True])
    torch.ops.aten.index.Tensor_out(x.to(mojo_device), [mask.to(mojo_device)], out=out)
    _check(out, x[mask])
    with pytest.raises(RuntimeError, match="dtype"):
        torch.ops.aten.index.Tensor_out(
            x.to(mojo_device),
            [idx.to(mojo_device)],
            out=torch.empty(0, dtype=torch.int64, device=mojo_device),
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_narrow_copy_out(mojo_device, dtype):
    x = _make((4, 6), dtype)
    out = torch.empty(0, dtype=dtype, device=mojo_device)
    with ran("aten::narrow_copy.out"):
        torch.ops.aten.narrow_copy.out(x.to(mojo_device), 1, 2, 3, out=out)
    _check(out, x.narrow(1, 2, 3))
    out = torch.empty(0, dtype=dtype, device=mojo_device)
    torch.ops.aten.narrow_copy.out(x.to(mojo_device), 0, -2, 2, out=out)
    _check(out, x.narrow(0, -2, 2))


@pytest.mark.parametrize("dtype", DTYPES)
def test_fill_tensor(mojo_device, dtype):
    x = _make((3, 4), dtype)
    for value in (torch.tensor(2.5), torch.tensor(-3), torch.tensor(True)):
        for where in ("cpu", mojo_device):
            y = x.to(mojo_device)
            with ran("aten::fill_.Tensor"):
                y.fill_(value.to(where))
            _check(y, x.clone().fill_(value))
    # Strided self, and a value aliasing self.
    y = x.to(mojo_device).t()
    y.fill_(torch.tensor(1))
    _check(y, x.t().clone().fill_(1))
    y = x.to(mojo_device)
    y.fill_(y[1, 2])
    _check(y, x.clone().fill_(x[1, 2].item()))
    with pytest.raises(RuntimeError, match="0-dimension value tensor"):
        x.to(mojo_device).fill_(torch.ones(2))


# ---------------------------------------------------------------------------
# unique family (host round trip)
# ---------------------------------------------------------------------------


def _as_tuple(r: torch.Tensor | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return r if isinstance(r, tuple) else (r,)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"return_inverse": True},
        {"return_counts": True},
        {"return_inverse": True, "return_counts": True},
        {"sorted": False, "return_inverse": True},
        {"dim": 0, "return_inverse": True, "return_counts": True},
        {"dim": 1, "return_counts": True},
    ],
)
def test_unique(mojo_device, dtype, kwargs):
    x = (_make((4, 6), torch.float32) / 6).round().to(dtype)
    x[2] = x[0]
    with ran("aten::_unique2", "aten::unique_dim"):
        got = _as_tuple(torch.unique(x.to(mojo_device), **kwargs))
    for g, e in zip(got, _as_tuple(torch.unique(x, **kwargs)), strict=True):
        _check(g, e)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"return_inverse": True, "return_counts": True},
        {"dim": 0, "return_inverse": True},
        {"dim": 1, "return_counts": True},
    ],
)
def test_unique_consecutive(mojo_device, dtype, kwargs):
    x = (_make((5, 4), torch.float32) / 8).round().to(dtype)
    x[1] = x[0]
    with ran("aten::unique_consecutive", "aten::unique_dim_consecutive"):
        got = _as_tuple(torch.unique_consecutive(x.to(mojo_device), **kwargs))
    for g, e in zip(got, _as_tuple(torch.unique_consecutive(x, **kwargs)), strict=True):
        _check(g, e)


def test_unique_edges(mojo_device):
    s = torch.tensor(3.0)
    for g, e in zip(
        torch.unique(s.to(mojo_device), return_inverse=True),
        torch.unique(s, return_inverse=True),
        strict=True,
    ):
        _check(g, e)
    _check(torch.unique(torch.empty(0).to(mojo_device)), torch.unique(torch.empty(0)))
    nan = float("nan")
    x = torch.tensor([2.0, nan, 1.0, 2.0, nan])
    got = torch.unique(x.to(mojo_device), return_counts=True)
    torch.testing.assert_close(got[0].cpu(), torch.unique(x), equal_nan=True)
    # torch.unique_consecutive of a strided view.
    y = torch.tensor([[1, 1], [2, 2], [2, 3]]).t()
    _check(torch.unique_consecutive(y.to(mojo_device)), torch.unique_consecutive(y))
    with pytest.raises(IndexError, match="Dimension out of range"):
        torch.unique(torch.ones(2, 3).to(mojo_device), dim=4)


# ---------------------------------------------------------------------------
# review follow-ups: overlap, exact integer scalars, determinism, unique raw
# outputs, fill_ overflow
# ---------------------------------------------------------------------------


def test_in_place_scatter_and_index_reduce_reject_overlap(mojo_device):
    a = torch.arange(8.0, device=mojo_device)
    idx = torch.tensor([0, 1], device=mojo_device)
    for call in (
        lambda: a.scatter_reduce_(0, idx, a[2:4], "sum"),
        lambda: a.scatter_(0, idx, a[2:4], reduce="add"),
        lambda: a.scatter_(0, idx, a[2:4]),
        lambda: a.index_reduce_(0, idx, a[2:4], "amax"),
    ):
        with pytest.raises(RuntimeError, match="refer to a single memory location"):
            call()
    expanded = torch.zeros(1, device=mojo_device).expand(4)
    with pytest.raises(RuntimeError, match="more than one element"):
        expanded.scatter_(0, torch.tensor([0], device=mojo_device), 1.0)
    with pytest.raises(RuntimeError, match="more than one element"):
        expanded.index_reduce_(
            0,
            torch.tensor([0], device=mojo_device),
            torch.ones(1, device=mojo_device),
            "prod",
        )


@pytest.mark.parametrize("reduce", [None, "add", "multiply"])
def test_scatter_int64_scalar_is_exact(mojo_device, reduce):
    big = 2**53 + 1
    x = torch.ones(3, dtype=torch.int64)
    idx = torch.tensor([2])
    kwargs = {} if reduce is None else {"reduce": reduce}
    expected = x.scatter(0, idx, big, **kwargs)
    _check(x.to(mojo_device).scatter(0, idx.to(mojo_device), big, **kwargs), expected)
    y = x.to(mojo_device)
    y.scatter_(0, idx.to(mojo_device), big, **kwargs)
    _check(y, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_scatter_reduce_sum_is_ordered_and_deterministic(mojo_device, dtype, reduce):
    """Under deterministic algorithms a floating sum takes the sorted route
    (`_scatter_via_index_put`): the same answer every run, and allowed under
    torch.use_deterministic_algorithms."""
    g = torch.Generator().manual_seed(7)
    x = torch.randn(20, 9, generator=g).to(dtype)
    src = (torch.randn(300, 9, generator=g) * 100).to(dtype)
    index = torch.randint(0, 20, (300, 9), generator=g)
    expected = x.scatter_reduce(0, index, src, reduce)
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        first = x.to(mojo_device).scatter_reduce(
            0, index.to(mojo_device), src.to(mojo_device), reduce
        )
        second = x.to(mojo_device).scatter_reduce(
            0, index.to(mojo_device), src.to(mojo_device), reduce
        )
        legacy = x.to(mojo_device).scatter(
            0, index.to(mojo_device), src.to(mojo_device), reduce="add"
        )
    finally:
        torch.use_deterministic_algorithms(before)
    # Run to run identical; CUDA's stride-1 index_put sums each run with 32
    # lanes and a tree, so float32 can differ from CPU's sequential sum in
    # the last bits.
    assert torch.equal(first.cpu(), second.cpu())
    tol = {"rtol": 1e-5, "atol": 1e-3} if dtype == torch.float32 else {}
    torch.testing.assert_close(first.cpu(), expected, **tol)
    torch.testing.assert_close(
        legacy.cpu(), x.scatter(0, index, src, reduce="add"), **tol
    )


def test_scatter_reduce_nondeterministic_alerts(mojo_device):
    """CUDA alerts for prod, and for the legacy kernel's multiply and integer
    add."""
    d = torch.zeros(3, device=mojo_device)
    i = torch.tensor([0], device=mojo_device)
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with pytest.raises(RuntimeError, match="scatter_reduce_cuda_prod_"):
            d.scatter_reduce(0, i, torch.ones(1, device=mojo_device), "prod")
        with pytest.raises(RuntimeError, match="scatter_reduce_cuda_kernel"):
            d.long().scatter(
                0, i, torch.ones(1, dtype=torch.long, device=mojo_device), reduce="add"
            )
        with pytest.raises(RuntimeError, match="scatter_reduce_cuda_kernel"):
            d.scatter(0, i, torch.ones(1, device=mojo_device), reduce="multiply")
        # amax / amin are order-independent: no alert.
        d.scatter_reduce(0, i, torch.ones(1, device=mojo_device), "amax")
    finally:
        torch.use_deterministic_algorithms(before)


def _unique_reference(
    op: str, x: torch.Tensor, *args: object
) -> tuple[torch.Tensor, ...]:
    """CPU torch's outputs, with CUDA's conventions where they differ:
    outputs not asked for are empty (CPU fills them for the dim ops), and
    `sorted=False` sorts (the flat CPU op hashes)."""
    a = torch.ops.aten
    if op == "_unique2":
        return a._unique2(x, True, *args[1:])
    if op == "_unique":
        return a._unique(x, True, *args[1:])
    out = tuple(getattr(a, op)(x, *args))
    inverse, counts = (args[-2], args[-1])
    empty = torch.empty(0, dtype=torch.int64)
    return (out[0], out[1] if inverse else empty, out[2] if counts else empty)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.int64, torch.int8, torch.bool]
)
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("counts", [False, True])
def test_unique_raw_aten_outputs(mojo_device, dtype, inverse, counts):
    g = torch.Generator().manual_seed(3)
    x = torch.randint(0, 3, (9, 4), generator=g).to(dtype)
    x[4] = x[1]
    a = torch.ops.aten
    d = x.to(mojo_device)
    cases = [
        (
            "unique_dim",
            (0, True, inverse, counts),
            lambda: a.unique_dim(d, 0, True, inverse, counts),
        ),
        (
            "unique_dim",
            (1, True, inverse, counts),
            lambda: a.unique_dim(d, 1, True, inverse, counts),
        ),
        (
            "unique_dim_consecutive",
            (0, inverse, counts),
            lambda: a.unique_dim_consecutive(d, 0, inverse, counts),
        ),
        (
            "unique_dim_consecutive",
            (-1, inverse, counts),
            lambda: a.unique_dim_consecutive(d, -1, inverse, counts),
        ),
    ]
    for op, args, run in cases:
        with ran(f"aten::{op}"):
            got = run()
        for g_, e in zip(got, _unique_reference(op, x, *args), strict=True):
            _check(g_, e)
    with ran("aten::_unique2"):
        got = a._unique2(d, False, inverse, counts)
    expected = a._unique2(x, True, inverse, counts)
    _check(got[0], expected[0])
    if inverse:
        _check(got[1], expected[1])
    else:
        assert got[1].numel() == 0
    if counts or dtype == torch.bool:
        # CUDA's bool route always counts.
        _check(got[2], torch.ops.aten._unique2(x, True, False, True)[2])
    else:
        assert got[2].numel() == 0
    got = a.unique_consecutive(d, inverse, counts, None)
    expected = a.unique_consecutive(x, inverse, counts, None)
    for g_, e in zip(got, expected, strict=True):
        _check(g_, e)


def test_unique_signed_zero_follows_cuda(mojo_device):
    """A run keeps its first element, or its last when counts are asked for
    (cub's run_length_encode)."""
    x = torch.tensor([-0.0, 0.0, 1.0])
    a = torch.ops.aten
    d = x.to(mojo_device)
    assert torch.signbit(a._unique2(d, True, False, False)[0].cpu()[0])
    assert not torch.signbit(a._unique2(d, True, False, True)[0].cpu()[0])
    assert torch.signbit(a.unique_consecutive(d, False, False, None)[0].cpu()[0])
    assert not torch.signbit(a.unique_consecutive(d, False, True, None)[0].cpu()[0])


def test_unique_dim_errors(mojo_device):
    a = torch.ops.aten
    with pytest.raises(IndexError, match="tensor has no dimensions"):
        a.unique_dim(torch.tensor(3.0, device=mojo_device), 0)
    with pytest.raises(RuntimeError, match="0 sized dimensions"):
        a.unique_dim(torch.ones(2, 0, device=mojo_device), 0)
    out = a.unique_dim(torch.ones(0, 3, device=mojo_device), 0)
    assert out[0].shape == (0, 3)


def test_fill_tensor_overflow(mojo_device):
    x = torch.zeros(3, dtype=torch.int8, device=mojo_device)
    with pytest.raises(RuntimeError, match="without overflow"):
        x.fill_(torch.tensor(300))
    # A same-device value is a copy_: it converts without a check.
    x.fill_(torch.tensor(300, device=mojo_device))
    _check(x, torch.full((3,), 300).to(torch.int8))


@pytest.mark.parametrize(
    ("dtype", "n"),
    [(torch.bfloat16, 1024), (torch.float16, 4096), (torch.float16, 3000)],
)
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("deterministic", [False, True])
def test_mean_counts_do_not_saturate_against_the_sum(
    mojo_device, dtype, n, include_self, deterministic
):
    """CUDA-faithful references (checked against stock CUDA 2.11). By
    default CUDA adds the sum and the count with dtype atomics: both
    saturate together (256 for bfloat16, 2048 for float16), so n ones
    average to 1. Under deterministic algorithms scatter_reduce takes the
    index_put route for the sum AND the count: both accumulate in float and
    are stored once, also 1."""
    x = torch.zeros(1, dtype=dtype)
    idx = torch.zeros(n, dtype=torch.long).to(mojo_device)
    ones = torch.ones(n, dtype=dtype).to(mojo_device)
    d = x.to(mojo_device)
    saturated = torch.zeros(1, dtype=dtype)
    for _ in range(n):
        saturated += 1  # one rounding per add, like a dtype atomic
    expected_sum = torch.full((1,), float(n)).to(dtype) if deterministic else saturated
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(deterministic)
    try:
        mean = d.scatter_reduce(0, idx, ones, "mean", include_self=include_self)
        total = d.scatter_reduce(0, idx, ones, "sum")
    finally:
        torch.use_deterministic_algorithms(before)
    _check(mean, torch.ones(1, dtype=dtype))
    _check(total, expected_sum)
    # index_reduce is always the atomic route (it alerts in deterministic
    # mode, like CUDA).
    got = d.index_reduce(0, idx, ones, "mean", include_self=include_self)
    _check(got, torch.ones(1, dtype=dtype))


def test_scatter_add_and_index_add_in_place_overlap_and_determinism(mojo_device):
    a = torch.arange(6.0, device=mojo_device)
    with pytest.raises(RuntimeError, match="single memory location"):
        a.scatter_add_(0, torch.tensor([0, 1], device=mojo_device), a[:2])
    with pytest.raises(RuntimeError, match="single memory location"):
        a.index_add_(0, torch.tensor([0, 1], device=mojo_device), a[2:4])
    g = torch.Generator().manual_seed(1)
    x = torch.randn(30, 5, generator=g)
    src = torch.randn(400, 5, generator=g) * 100
    idx = torch.randint(0, 30, (400, 5), generator=g)
    rows = torch.randint(0, 30, (400,), generator=g)
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        sa = x.to(mojo_device).scatter_add(0, idx.to(mojo_device), src.to(mojo_device))
        ia = x.to(mojo_device).index_add(0, rows.to(mojo_device), src.to(mojo_device))
    finally:
        torch.use_deterministic_algorithms(before)
    # CUDA's index_put orders (a 32-lane tree for width-1 slices): close to
    # CPU's sequential sums, identical run to run.
    torch.testing.assert_close(
        sa.cpu(), x.scatter_add(0, idx, src), rtol=1e-5, atol=1e-3
    )
    torch.testing.assert_close(
        ia.cpu(), x.index_add(0, rows, src), rtol=1e-5, atol=1e-3
    )


def test_unique_dim_many_columns(mojo_device):
    """One merge sort of the rows, whatever their width."""
    g = torch.Generator().manual_seed(5)
    x = torch.randint(0, 2, (6, 20000), generator=g)
    x[3] = x[0]
    for dim in (0, 1):
        got = torch.unique(
            x.to(mojo_device), dim=dim, return_inverse=True, return_counts=True
        )
        for g_, e in zip(
            got,
            torch.unique(x, dim=dim, return_inverse=True, return_counts=True),
            strict=True,
        ):
            _check(g_, e)


def _unique_rows_reference(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """unique(dim=0) with a strict total order: rows sorted lexicographically
    with NaN after every number (ties by position), and NaN never equal to
    anything, so a row holding one is its own group. CPU's comparator-based
    sort is no strict weak order with NaN and can split equal rows."""
    rows = x.tolist()

    def key(i: int) -> tuple[object, ...]:
        return (tuple((v != v, 0.0 if v != v else v) for v in rows[i]), i)

    order = sorted(range(len(rows)), key=key)
    values: list[list[float]] = []
    inverse = [0] * len(rows)
    counts: list[int] = []
    for pos, i in enumerate(order):
        prev = rows[order[pos - 1]] if pos else None
        same = prev is not None and all(
            a == b for a, b in zip(prev, rows[i], strict=True)
        )
        if not same:
            values.append(rows[i])
            counts.append(0)
        inverse[i] = len(values) - 1
        counts[-1] += 1
    return (
        torch.tensor(values, dtype=x.dtype).reshape(-1, x.shape[1]),
        torch.tensor(inverse),
        torch.tensor(counts),
    )


@pytest.mark.parametrize("seed", range(6))
def test_unique_dim_nan_signed_zero_duplicates_stress(mojo_device, seed):
    """Rows with NaN, -0.0 / 0.0 (equal) and many duplicates: every row
    lands in exactly one merge slot, on every run."""
    g = torch.Generator().manual_seed(seed)
    choices = torch.tensor([float("nan"), -0.0, 0.0, 1.0, -2.5])
    x = choices[torch.randint(0, 5, (97, 2), generator=g)]
    x[5] = x[11]
    expected = _unique_rows_reference(x)
    for _ in range(3):
        got = torch.unique(
            x.to(mojo_device), dim=0, return_inverse=True, return_counts=True
        )
        torch.testing.assert_close(
            got[0].cpu(), expected[0], equal_nan=True, rtol=0, atol=0
        )
        _check(got[1], expected[1])
        _check(got[2], expected[2])
    small = torch.tensor([[1.0], [float("nan")], [0.0], [2.0]])
    got = torch.unique(
        small.to(mojo_device), dim=0, return_inverse=True, return_counts=True
    )
    expected = _unique_rows_reference(small)
    assert expected[2].tolist() == [1, 1, 1, 1]  # the NaN row is kept
    torch.testing.assert_close(
        got[0].cpu(), expected[0], equal_nan=True, rtol=0, atol=0
    )
    _check(got[1], expected[1])
    _check(got[2], expected[2])


def test_put_alerts_in_deterministic_mode(mojo_device):
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        for accumulate in (False, True):
            with pytest.raises(RuntimeError, match="put_"):
                torch.zeros(4, device=mojo_device).put_(
                    torch.tensor([0], device=mojo_device),
                    torch.ones(1, device=mojo_device),
                    accumulate=accumulate,
                )
    finally:
        torch.use_deterministic_algorithms(before)


@pytest.mark.parametrize("width", [1, 8, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_deterministic_index_add_rounds_like_cuda(mojo_device, width, dtype):
    """CUDA's deterministic index_put: a slice wider than a warp rounds into
    the dtype after every addition (1024 bfloat16 ones saturate at 256);
    narrower slices sum the run in float and add it once (1024)."""
    z = torch.zeros(2, width, dtype=dtype)
    idx = torch.zeros(1024, dtype=torch.long)
    ones = torch.ones(1024, width, dtype=dtype)
    if width > 32:
        expected_value = torch.zeros(1, dtype=dtype)
        for _ in range(1024):
            expected_value += 1
        expected_value = expected_value.item()
    else:
        expected_value = 1024.0
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        got = z.to(mojo_device).index_add(0, idx.to(mojo_device), ones.to(mojo_device))
        # Expanded on the device: `.to()` would materialize the stride-0 view.
        expanded = idx.to(mojo_device).view(-1, 1).expand(1024, width)
        sa = z.to(mojo_device).scatter_add(0, expanded, ones.to(mojo_device))
    finally:
        torch.use_deterministic_algorithms(before)
    expected = torch.zeros(2, width, dtype=dtype)
    expected[0] = expected_value
    _check(got, expected)
    _check(sa, expected)


def test_in_place_ops_accept_disjoint_strided_views(mojo_device):
    """ATen's overlap check calls an interleaved view TooHard and lets it
    through: x[::2] written from x[1::2] is valid."""
    calls = [
        lambda b, i: b[::2].scatter_add_(0, i, b[1::2]),
        lambda b, i: b[::2].index_add_(0, i, b[1::2]),
        lambda b, i: b[::2].scatter_(0, i, b[1::2]),
        lambda b, i: b[::2].scatter_reduce_(0, i, b[1::2], "amax"),
        lambda b, i: b[::2].index_reduce_(0, i, b[1::2], "prod"),
    ]
    for call in calls:
        expected = torch.arange(10.0)
        call(expected, torch.arange(5))
        got = torch.arange(10.0, device=mojo_device)
        call(got, torch.arange(5, device=mojo_device))
        _check(got, expected)


def test_fill_tensor_value_devices_and_overlap(mojo_device):
    with pytest.raises(RuntimeError, match="more than one element"):
        torch.zeros(1, device=mojo_device).expand(3).fill_(
            torch.tensor(2.0, device=mojo_device)
        )
    x = torch.zeros(1, device=mojo_device).expand(3)
    x.fill_(torch.tensor(2.0))  # a CPU value is fill_(Scalar): allowed
    assert x.cpu().tolist() == [2.0, 2.0, 2.0]


@pytest.mark.skipif(
    len(
        [
            a
            for a in __import__("torch_mojo_backend").get_accelerators()
            if a.label != "cpu"
        ]
    )
    < 2,
    reason="needs two mojo GPUs",
)
def test_fill_tensor_value_on_another_gpu_is_range_checked():
    x = torch.zeros(3, dtype=torch.int8, device="mojo:0")
    with pytest.raises(RuntimeError, match="without overflow"):
        x.fill_(torch.tensor(300, device="mojo:1"))


def _index_put_sum_reference(values: list[float], start: float, width: int) -> float:
    """CUDA's sorted index_put accumulation of one float32 run into `start`:
    width 1 sums 32 lanes and a shuffle-down tree (indexing_backward_kernel_
    stride_1), widths up to 32 sum sequentially from 0 (_small_stride); both
    then add the sum to `start` once."""
    f = np.float32
    acc = f(0)
    j = 0
    if width == 1:
        passes = len(values) // 32
        if passes:
            lanes = [f(0)] * 32
            for p in range(passes):
                for lane in range(32):
                    lanes[lane] = f(lanes[lane] + f(values[p * 32 + lane]))
            offset = 16
            while offset:
                lanes = [
                    f(
                        lanes[lane]
                        + (lanes[lane + offset] if lane + offset < 32 else lanes[lane])
                    )
                    for lane in range(32)
                ]
                offset //= 2
            acc = lanes[0]
        j = passes * 32
    for v in values[j:]:
        acc = f(acc + f(v))
    return float(f(f(start) + acc))


@pytest.mark.parametrize("width", [1, 8])
def test_deterministic_index_add_float32_order_is_cuda_s(mojo_device, width):
    """Random float32 values pin the summation order, not just the rounding."""
    if is_metal(mojo_device):
        pytest.skip("the lane tree is CUDA's warp (Metal has no warp-size rule here)")
    g = torch.Generator().manual_seed(2)
    n = 1000
    base = torch.randn(2, width, generator=g)
    src = torch.randn(n, width, generator=g) * 7
    idx = torch.zeros(n, dtype=torch.long)
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        got = (
            base.to(mojo_device)
            .index_add(0, idx.to(mojo_device), src.to(mojo_device))
            .cpu()
        )
    finally:
        torch.use_deterministic_algorithms(before)
    for c in range(width):
        expected = _index_put_sum_reference(
            src[:, c].tolist(), base[0, c].item(), width
        )
        assert got[0, c].item() == expected
    assert torch.equal(got[1], base[1])
