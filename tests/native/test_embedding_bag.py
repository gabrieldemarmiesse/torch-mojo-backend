"""Native backend: embedding_bag group (torch_mojo_backend/mojo/tmb/ops/embedding_bag.mojo).

_embedding_bag / _embedding_bag_forward_only through F.embedding_bag (sum,
mean, max; offsets, include_last_offset, padding_idx, per_sample_weights),
their backwards through autograd, and embedding_renorm_ (F.embedding's
max_norm). Public torch API only, compared against CPU torch.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.native.conftest import ran, skip_if_metal

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.float64]


def _close(got: torch.Tensor, expected: torch.Tensor | None, dtype: torch.dtype):
    assert expected is not None
    tol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    torch.testing.assert_close(got.cpu(), expected, rtol=tol, atol=tol)


def _grad(t: torch.Tensor) -> torch.Tensor:
    assert t.grad is not None
    return t.grad.cpu()


def _case(dtype: torch.dtype, seed: int = 0) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    weight = torch.randn(11, 6, generator=g).to(dtype)
    # Bag 1 is empty; index 2 repeats inside and across bags.
    indices = torch.tensor([1, 2, 4, 2, 7, 3, 2, 10, 0, 5])
    offsets = torch.tensor([0, 4, 4, 7])
    psw = torch.randn(indices.numel(), generator=g).to(dtype)
    return weight, indices, offsets, psw


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
@pytest.mark.parametrize("include_last_offset", [False, True])
@pytest.mark.parametrize("padding_idx", [None, 2])
def test_embedding_bag_forward_and_backward(
    mojo_device, dtype, mode, include_last_offset, padding_idx
):
    if dtype == torch.float64:
        skip_if_metal(mojo_device, "Apple GPUs have no float64")
    weight, indices, offsets, _ = _case(dtype)
    if include_last_offset:
        offsets = torch.cat([offsets, torch.tensor([indices.numel()])])
    w_ref = weight.clone().requires_grad_(True)
    expected = F.embedding_bag(
        indices,
        w_ref,
        offsets,
        mode=mode,
        include_last_offset=include_last_offset,
        padding_idx=padding_idx,
    )
    w_dev = weight.to(mojo_device).requires_grad_(True)
    with ran("aten::_embedding_bag"):
        got = F.embedding_bag(
            indices.to(mojo_device),
            w_dev,
            offsets.to(mojo_device),
            mode=mode,
            include_last_offset=include_last_offset,
            padding_idx=padding_idx,
        )
    _close(got, expected.detach(), dtype)
    grad = torch.randn(expected.shape, generator=torch.Generator().manual_seed(1)).to(
        dtype
    )
    expected.backward(grad)
    with ran("aten::_embedding_bag_dense_backward"):
        got.backward(grad.to(mojo_device))
    _close(_grad(w_dev), w_ref.grad, dtype)
    # A weight that needs no grad takes the forward-only overload.
    with ran("aten::_embedding_bag_forward_only"):
        again = F.embedding_bag(
            indices.to(mojo_device),
            w_dev.detach(),
            offsets.to(mojo_device),
            mode=mode,
            include_last_offset=include_last_offset,
            padding_idx=padding_idx,
        )
    _close(again, expected.detach(), dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding_idx", [None, 4])
def test_embedding_bag_per_sample_weights(mojo_device, dtype, padding_idx):
    weight, indices, offsets, psw = _case(dtype, 3)
    w_ref = weight.clone().requires_grad_(True)
    p_ref = psw.clone().requires_grad_(True)
    expected = F.embedding_bag(
        indices,
        w_ref,
        offsets,
        mode="sum",
        per_sample_weights=p_ref,
        padding_idx=padding_idx,
    )
    w_dev = weight.to(mojo_device).requires_grad_(True)
    p_dev = psw.to(mojo_device).requires_grad_(True)
    got = F.embedding_bag(
        indices.to(mojo_device),
        w_dev,
        offsets.to(mojo_device),
        mode="sum",
        per_sample_weights=p_dev,
        padding_idx=padding_idx,
    )
    _close(got, expected.detach(), dtype)
    grad = torch.randn(expected.shape, generator=torch.Generator().manual_seed(2)).to(
        dtype
    )
    expected.backward(grad)
    with ran("aten::_embedding_bag_per_sample_weights_backward"):
        got.backward(grad.to(mojo_device))
    _close(_grad(w_dev), w_ref.grad, dtype)
    _close(_grad(p_dev), p_ref.grad, dtype)


@pytest.mark.parametrize("mode", ["sum", "mean"])
def test_embedding_bag_scale_grad_by_freq_and_index_dtypes(mojo_device, mode):
    """CUDA's scale_grad_by_freq divides every row's gradient by how often
    its index occurs. CPU torch 2.11 scales the wrong rows (it indexes the
    counts by position), so the reference is the unscaled CPU gradient
    divided by the counts."""
    weight, indices, offsets, _ = _case(torch.float32, 4)
    w_ref = weight.clone().requires_grad_(True)
    F.embedding_bag(indices, w_ref, offsets, mode=mode).sum().backward()
    counts = torch.bincount(indices, minlength=weight.shape[0]).clamp(min=1)
    expected_grad = _grad(w_ref) / counts.unsqueeze(1)
    for idx_dtype, off_dtype in [
        (torch.int32, torch.int32),
        (torch.int32, torch.int64),
        (torch.int64, torch.int32),
    ]:
        i = indices.to(idx_dtype)
        o = offsets.to(off_dtype)
        expected = F.embedding_bag(i, weight, o, mode=mode)
        w_dev = weight.to(mojo_device).requires_grad_(True)
        got = F.embedding_bag(
            i.to(mojo_device),
            w_dev,
            o.to(mojo_device),
            mode=mode,
            scale_grad_by_freq=True,
        )
        torch.testing.assert_close(got.detach().cpu(), expected)
        got.sum().backward()
        torch.testing.assert_close(_grad(w_dev), expected_grad)


def test_embedding_bag_2d_input_and_edges(mojo_device):
    weight = torch.randn(7, 3)
    # A 2-D input is flattened by F.embedding_bag: one bag per row.
    x = torch.tensor([[0, 2, 2], [6, 1, 5]])
    torch.testing.assert_close(
        F.embedding_bag(x.to(mojo_device), weight.to(mojo_device), mode="max").cpu(),
        F.embedding_bag(x, weight, mode="max"),
    )
    # No indices at all: every bag is empty.
    empty = torch.empty(0, dtype=torch.int64)
    off = torch.tensor([0, 0])
    for mode in ("sum", "mean", "max"):
        torch.testing.assert_close(
            F.embedding_bag(
                empty.to(mojo_device),
                weight.to(mojo_device),
                off.to(mojo_device),
                mode=mode,
            ).cpu(),
            F.embedding_bag(empty, weight, off, mode=mode),
        )


def test_embedding_bag_errors(mojo_device):
    weight = torch.randn(5, 3, device=mojo_device)
    offsets = torch.tensor([0, 2], device=mojo_device)
    with pytest.raises(IndexError, match="index out of range"):
        F.embedding_bag(torch.tensor([0, 5, 1], device=mojo_device), weight, offsets)
    with pytest.raises(IndexError, match="index out of range"):
        F.embedding_bag(torch.tensor([0, -1, 1], device=mojo_device), weight, offsets)
    with pytest.raises(RuntimeError, match="Long, Int"):
        torch.ops.aten._embedding_bag(
            weight, torch.tensor([0.0, 1.0], device=mojo_device), offsets
        )
    with pytest.raises(RuntimeError, match="weight has to be a 2D Tensor"):
        torch.ops.aten._embedding_bag(
            torch.randn(5, device=mojo_device),
            torch.tensor([0, 1], device=mojo_device),
            offsets,
        )
    with pytest.raises(RuntimeError, match="only supported for mode='sum'"):
        torch.ops.aten._embedding_bag_per_sample_weights_backward(
            torch.randn(2, 3, device=mojo_device),
            weight,
            torch.tensor([0, 1], device=mojo_device),
            offsets,
            torch.tensor([0, 1], device=mojo_device),
            1,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("norm_type", [1.0, 2.0, 3.0])
def test_embedding_renorm(mojo_device, dtype, norm_type):
    weight = (torch.randn(9, 5, generator=torch.Generator().manual_seed(5)) * 2).abs()
    weight = weight.to(dtype)
    # Duplicates and a negative index (wrapped) touch each row once.
    idx = torch.tensor([1, 1, 4, -1, 8, 0])
    expected = weight.clone()
    torch.embedding_renorm_(expected, idx, 1.5, norm_type)
    got = weight.to(mojo_device)
    with ran("aten::embedding_renorm_"):
        torch.embedding_renorm_(got, idx.to(mojo_device), 1.5, norm_type)
    _close(got, expected, dtype)
    # Through F.embedding's max_norm.
    w2 = weight.to(mojo_device)
    out = F.embedding(idx.to(mojo_device) % 9, w2, max_norm=1.5, norm_type=norm_type)
    w_ref = weight.clone()
    ref = F.embedding(idx % 9, w_ref, max_norm=1.5, norm_type=norm_type)
    _close(out, ref, dtype)
    _close(w2, w_ref, dtype)


def test_embedding_renorm_errors(mojo_device):
    w = torch.randn(4, 3, device=mojo_device)
    with pytest.raises(IndexError, match="out of bounds"):
        torch.embedding_renorm_(w, torch.tensor([4], device=mojo_device), 1.0, 2.0)
    with pytest.raises(RuntimeError, match="2-dimensional"):
        torch.embedding_renorm_(
            torch.randn(4, device=mojo_device),
            torch.tensor([0], device=mojo_device),
            1.0,
            2.0,
        )
    # No indices: nothing to do.
    before = w.cpu()
    torch.embedding_renorm_(
        w, torch.empty(0, dtype=torch.int64, device=mojo_device), 1.0, 2.0
    )
    torch.testing.assert_close(w.cpu(), before)


def test_embedding_bag_backward_rebuilds_offset2bag(mojo_device):
    """`_embedding_bag_backward` given no offset2bag (what a forward-only
    call returns) rebuilds it from the offsets, as ATen does."""
    weight, indices, offsets, _ = _case(torch.float32, 6)
    grad = torch.randn(offsets.numel(), weight.shape[1])
    empty = torch.empty(0, dtype=torch.int64)
    bag_size = torch.tensor([4, 0, 3, 3])
    args = (
        indices,
        offsets,
        empty,
        bag_size,
        empty,
        weight.shape[0],
        False,
        0,
        False,
        None,
    )
    expected = torch.ops.aten._embedding_bag_backward(grad, *args)
    dev_args = tuple(
        a.to(mojo_device) if isinstance(a, torch.Tensor) else a for a in args
    )
    with ran("aten::_embedding_bag_backward"):
        got = torch.ops.aten._embedding_bag_backward(grad.to(mojo_device), *dev_args)
    torch.testing.assert_close(got.cpu(), expected)
    with pytest.raises(NotImplementedError, match="sparse"):
        torch.ops.aten._embedding_bag_backward(
            grad.to(mojo_device), *dev_args[:8], True, None
        )
