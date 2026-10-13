"""Tests for the complex dtypes on the mojo device (complex64 / complex128):
dtype plumbing (allocation, transfers, casts, fills, views), the lazy conj
bit, and the complex ops of torch_mojo_backend/mojo/tmb/ops/complex.mojo.

Public torch API only, compared against CPU torch. complex128 needs float64,
which Apple GPUs do not have.
"""

import warnings

import pytest
import torch

from tests.native.conftest import is_metal, skip_if_metal
from torch_mojo_backend import native

CDTYPES = [torch.complex64, torch.complex128]


def _skip_c128(device: str, dtype: torch.dtype):
    if dtype == torch.complex128:
        skip_if_metal(device, "complex128 needs float64, which Apple GPUs lack")


def _close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = False,
):
    torch.testing.assert_close(
        actual.cpu(), expected, rtol=rtol, atol=atol, equal_nan=equal_nan
    )


def _put(
    t: torch.Tensor, index: int | slice | tuple[int, int] | torch.Tensor, value: complex
):
    """t[index] = value (a complex Python scalar, wrapped for the stubs)."""
    t[index] = torch.tensor(value, dtype=t.dtype, device=t.device)


def _rand(*shape: int, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    return torch.randn(shape, dtype=dtype)


# --- plumbing ------------------------------------------------------------------


@pytest.mark.parametrize("dtype", CDTYPES)
def test_roundtrip_and_layouts(mojo_gpu: str, dtype: torch.dtype):
    _skip_c128(mojo_gpu, dtype)
    x = _rand(4, 5, 3, dtype=dtype)
    d = x.to(mojo_gpu)
    assert d.dtype == dtype and d.device.type == "mojo"
    assert torch.equal(d.cpu(), x)
    assert torch.equal(
        d.permute(2, 0, 1).contiguous().cpu(), x.permute(2, 0, 1).contiguous()
    )
    assert torch.equal(d[1:, ::2].clone().cpu(), x[1:, ::2].clone())
    assert torch.equal(d.reshape(20, 3).cpu(), x.reshape(20, 3))
    assert torch.equal(
        d.transpose(0, 1).reshape(-1).cpu(), x.transpose(0, 1).reshape(-1)
    )
    # strided host source and destination
    assert torch.equal(x[:, ::2].to(mojo_gpu).cpu(), x[:, ::2])
    out = torch.empty(5, 4, 3, dtype=dtype).transpose(0, 1)
    out.copy_(d)
    assert torch.equal(out, x)


@pytest.mark.parametrize("dtype", CDTYPES)
def test_empty_and_storage(mojo_gpu: str, dtype: torch.dtype):
    _skip_c128(mojo_gpu, dtype)
    t = torch.empty(3, 2, dtype=dtype, device=mojo_gpu)
    assert t.untyped_storage().nbytes() == 6 * t.element_size()
    assert torch.empty(0, dtype=dtype, device=mojo_gpu).numel() == 0


def test_supported_dtypes_lists_complex(mojo_gpu: str):
    if not hasattr(torch.accelerator, "get_device_capability"):
        pytest.skip("torch < 2.10")
    index = torch.device(mojo_gpu).index
    got = torch.accelerator.get_device_capability(index)["supported_dtypes"]
    assert torch.complex64 in got
    assert (torch.complex128 in got) == (not is_metal(mojo_gpu))
    assert torch.complex32 not in got


def test_complex32_declines(mojo_gpu: str):
    with pytest.raises(NotImplementedError):
        torch.empty(2, dtype=torch.complex32, device=mojo_gpu)


@pytest.mark.parametrize("dtype", CDTYPES)
def test_fills(mojo_gpu: str, dtype: torch.dtype):
    _skip_c128(mojo_gpu, dtype)
    assert torch.equal(
        torch.zeros(3, 4, dtype=dtype, device=mojo_gpu).cpu(),
        torch.zeros(3, 4, dtype=dtype),
    )
    assert torch.equal(
        torch.ones(3, dtype=dtype, device=mojo_gpu).cpu(), torch.ones(3, dtype=dtype)
    )
    assert torch.equal(
        torch.full((2, 3), 1.5 - 2j, dtype=dtype, device=mojo_gpu).cpu(),
        torch.full((2, 3), 1.5 - 2j, dtype=dtype),
    )
    # strided fill and zero
    d = _rand(4, 6, dtype=dtype).to(mojo_gpu)
    h = d.cpu()
    d[:, ::2].fill_(3 + 4j)
    h[:, ::2].fill_(3 + 4j)
    d[1:3, 1::2].zero_()
    h[1:3, 1::2].zero_()
    assert torch.equal(d.cpu(), h)
    # fill_ from a 0-d tensor, host and device
    d.fill_(torch.tensor(2 - 1j))
    assert torch.equal(d.cpu(), torch.full((4, 6), 2 - 1j, dtype=dtype))
    d.fill_(torch.tensor(-1j, dtype=dtype, device=mojo_gpu))
    assert torch.equal(d.cpu(), torch.full((4, 6), -1j, dtype=dtype))


@pytest.mark.parametrize("dtype", CDTYPES)
def test_item(mojo_gpu: str, dtype: torch.dtype):
    _skip_c128(mojo_gpu, dtype)
    x = _rand(3, dtype=dtype)
    d = x.to(mojo_gpu)
    assert d[1].item() == x[1].item()
    assert d.conj()[2].item() == x.conj()[2].item()


def test_casts(mojo_gpu: str):
    x = _rand(5, 7)
    d = x.to(mojo_gpu)
    r = torch.randn(5, 7)
    i = torch.randint(-5, 5, (5, 7))
    b = torch.rand(5, 7) > 0.5
    assert torch.equal(r.to(mojo_gpu).to(torch.complex64).cpu(), r.to(torch.complex64))
    assert torch.equal(i.to(mojo_gpu).to(torch.complex64).cpu(), i.to(torch.complex64))
    assert torch.equal(b.to(mojo_gpu).to(torch.complex64).cpu(), b.to(torch.complex64))
    h = r.half()
    assert torch.equal(h.to(mojo_gpu).to(torch.complex64).cpu(), h.to(torch.complex64))
    z = x.clone()
    _put(z, (0, 0), 0)
    _put(z, (1, 1), 3j)
    assert torch.equal(z.to(mojo_gpu).bool().cpu(), z.bool())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert torch.equal(d.float().cpu(), x.float())
        assert torch.equal(d.long().cpu(), x.long())
        assert torch.equal(d.half().cpu(), x.half())
    # copy_ across dtypes, strided destination
    dst = torch.zeros(7, 5, dtype=torch.complex64, device=mojo_gpu).t()
    dst.copy_(r.to(mojo_gpu))
    assert torch.equal(dst.cpu(), r.to(torch.complex64))
    # a host source with another dtype
    dst.copy_(i)
    assert torch.equal(dst.cpu(), i.to(torch.complex64))


def test_complex_to_real_warns(mojo_gpu: str):
    d = _rand(3).to(mojo_gpu)
    # TORCH_WARN_ONCE: any one statement of torch's copy_ may already have
    # fired in this process, so only the value is checked here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert torch.equal(d.real.clone().cpu(), d.float().cpu())


def test_complex128_casts(mojo_gpu: str):
    skip_if_metal(mojo_gpu, "complex128 needs float64")
    x = _rand(6, 4)
    d = x.to(mojo_gpu)
    assert torch.equal(d.to(torch.complex128).cpu(), x.to(torch.complex128))
    y = torch.randn(6, 4, dtype=torch.complex128)
    assert torch.equal(y.to(mojo_gpu).to(torch.complex64).cpu(), y.to(torch.complex64))
    r = torch.randn(3, dtype=torch.float64)
    assert torch.equal(r.to(mojo_gpu).to(torch.complex64).cpu(), r.to(torch.complex64))


# --- views and the conj bit ----------------------------------------------------


@pytest.mark.parametrize("dtype", CDTYPES)
def test_view_as_real_and_complex(mojo_gpu: str, dtype: torch.dtype):
    _skip_c128(mojo_gpu, dtype)
    x = _rand(3, 4, dtype=dtype)
    d = x.to(mojo_gpu)
    assert torch.equal(torch.view_as_real(d).cpu(), torch.view_as_real(x))
    assert torch.equal(d.real.cpu(), x.real)
    assert torch.equal(d.imag.cpu(), x.imag)
    r = torch.randn(3, 4, 2, dtype=x.real.dtype)
    assert torch.equal(
        torch.view_as_complex(r.to(mojo_gpu)).cpu(), torch.view_as_complex(r)
    )
    # writes through the views
    d.real.mul_(2)
    d.imag.zero_()
    assert torch.equal(d.cpu(), torch.complex(2 * x.real, torch.zeros_like(x.real)))


def test_conj_bit(mojo_gpu: str):
    x = _rand(4, 5)
    y = _rand(4, 5)
    d = x.to(mojo_gpu)
    c = d.conj()
    assert c.is_conj()
    assert torch.equal(c.resolve_conj().cpu(), x.conj().resolve_conj())
    assert torch.equal(c.clone().cpu(), x.conj().clone())
    assert torch.equal(torch.conj_physical(d).cpu(), torch.conj_physical(x))
    assert torch.equal(c.cpu(), x.conj())
    assert torch.equal(c.imag.cpu(), x.conj().imag)
    # copy_ into a conj destination, and from a conj host source
    dst = torch.zeros(4, 5, dtype=torch.complex64, device=mojo_gpu)
    dst.conj().copy_(y.to(mojo_gpu))
    assert torch.equal(dst.cpu(), y.conj())
    dst.copy_(y.conj())
    assert torch.equal(dst.cpu(), y.conj())
    e = d.clone()
    e.conj_physical_()
    assert torch.equal(e.cpu(), x.conj())


def test_eq_ne(mojo_gpu: str):
    x = _rand(10)
    y = x.clone()
    _put(y, slice(None, None, 3), 1j)
    _put(y, 1, complex(float("nan"), 0))
    d, e = x.to(mojo_gpu), y.to(mojo_gpu)
    assert torch.equal((d == e).cpu(), x == y)
    assert torch.equal((d != e).cpu(), x != y)
    assert torch.equal((e == e).cpu(), y == y)
    assert torch.equal((d == x[2].item()).cpu(), x == x[2].item())
    assert torch.equal((d != 1j).cpu(), x != 1j)


# --- reductions, data movement, random --------------------------------------------


def test_data_movement(mojo_gpu: str):
    x, y = _rand(3, 4), _rand(2, 4)
    d, e = x.to(mojo_gpu), y.to(mojo_gpu)
    _close(torch.cat([d, e]), torch.cat([x, y]))
    _close(torch.cat([d, e.to(torch.complex64)], dim=-2), torch.cat([x, y], dim=-2))
    _close(torch.stack([d, d], dim=-1), torch.stack([x, x], dim=-1))
    idx = torch.tensor([2, 0])
    _close(d[idx.to(mojo_gpu)], x[idx])
    _close(torch.index_select(d, -1, idx.to(mojo_gpu)), torch.index_select(x, -1, idx))
    _close(d.flip(0, 1), x.flip(0, 1))
    _close(d.roll(1, -1), x.roll(1, -1))
    _close(d.roll(5), x.roll(5))
    _close(d.repeat(2, 1, 3), x.repeat(2, 1, 3))
    g, h = d.clone(), x.clone()
    _put(g, (idx.to(mojo_gpu)), 7 - 1j)
    _put(h, (idx), 7 - 1j)
    _close(g, h)
    g[1] = e[0]
    h[1] = y[0]
    _close(g, h)


def test_random_fills(mojo_gpu: str):
    torch.manual_seed(3)
    z = torch.randn(1000, dtype=torch.complex64, device=mojo_gpu).cpu()
    assert abs(z.real.std().item() - 0.5**0.5) < 0.06
    assert abs(z.imag.std().item() - 0.5**0.5) < 0.06
    u = torch.rand(1000, dtype=torch.complex64, device=mojo_gpu).cpu()
    assert 0 <= u.real.min() and u.real.max() < 1 and 0 <= u.imag.min()


def test_unsupported_declines(mojo_gpu: str):
    """Ops not written for complex decline it instead of reading the pairs
    as reals."""
    d = _rand(3, 3).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        torch.relu(torch.view_as_complex(torch.randn(3, 2).to(mojo_gpu)))
    with pytest.raises((NotImplementedError, RuntimeError)):
        torch.linalg.cholesky(d)
    with pytest.raises(NotImplementedError):
        torch.tanh(d)


def test_truth_ops(mojo_gpu: str):
    """Ops that see a complex value only as `z != 0`."""
    x = _rand(4, 5)
    _put(x, (0, 0), 0)
    _put(x, (1, 2), 3j)
    y = _rand(4, 5)
    _put(y, (2, 2), 0)
    d, e = x.to(mojo_gpu), y.to(mojo_gpu)
    assert torch.equal(d.all().cpu(), x.all())
    assert torch.equal(d.any(1).cpu(), x.any(1))
    assert torch.equal(torch.count_nonzero(d, 0).cpu(), torch.count_nonzero(x, 0))
    assert torch.equal(torch.nonzero(d).cpu(), torch.nonzero(x))
    assert torch.equal(torch.logical_and(d, e).cpu(), torch.logical_and(x, y))
    assert torch.equal(torch.logical_xor(d, e).cpu(), torch.logical_xor(x, y))
    assert torch.equal(torch.logical_not(d).cpu(), torch.logical_not(x))
    assert torch.equal(d, d.clone()) and not torch.equal(d, e)


def test_selection_ops(mojo_gpu: str):
    x, y = _rand(4, 5), _rand(4, 5)
    d, e = x.to(mojo_gpu), y.to(mojo_gpu)
    mask = torch.rand(4, 5) > 0.5
    m = mask.to(mojo_gpu)
    _close(torch.where(m, d, e), torch.where(mask, x, y))
    _close(torch.where(m, d, 2.0), torch.where(mask, x, 2.0))
    _close(d.masked_fill(m, 1 - 1j), x.masked_fill(mask, 1 - 1j))
    g, h = d.clone(), x.clone()
    g.masked_fill_(m, 3.0)
    h.masked_fill_(mask, 3.0)
    _close(g, h)
    _close(torch.masked_select(d, m), torch.masked_select(x, mask))
    idx = torch.randint(0, 4, (3, 5))
    _close(torch.gather(d, 0, idx.to(mojo_gpu)), torch.gather(x, 0, idx))
    _close(torch.tril(d), torch.tril(x))
    _close(torch.triu(d, 1), torch.triu(x, 1))
    rows = torch.tensor([0, 2])
    src = _rand(2, 5)
    _close(
        d.index_add(0, rows.to(mojo_gpu), src.to(mojo_gpu), alpha=2),
        x.index_add(0, rows, src, alpha=2),
    )
    _close(
        d.index_copy(0, rows.to(mojo_gpu), src.to(mojo_gpu)), x.index_copy(0, rows, src)
    )


def test_misc_ops(mojo_gpu: str):
    x = _rand(3, 4)
    d = x.to(mojo_gpu)
    _close(torch.full_like(d, 2 + 1j), torch.full_like(x, 2 + 1j))
    r = d.clone()
    r.resize_(2, 2)
    assert r.shape == (2, 2) and torch.equal(r.cpu(), x.flatten()[:4].reshape(2, 2))


def test_rank_checks_use_the_complex_shape(mojo_gpu: str):
    v = _rand(5).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="at least 2 dimensions"):
        torch.tril(v)
    with pytest.raises(RuntimeError, match="at least 2 dimensions"):
        torch.triu(v)
    m = _rand(3, 4).to(mojo_gpu)
    i = torch.tensor([0, 1]).to(mojo_gpu)
    with pytest.raises(IndexError, match="too many indices"):
        m[i, i, i]


def test_index_put_dtype_must_match(mojo_gpu: str):
    d = _rand(4, 3).to(mojo_gpu)
    rows = torch.tensor([0, 2]).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="dtypes match"):
        d.index_put_((rows,), torch.ones(2, 3, device=mojo_gpu))
    d.index_put_((rows,), torch.ones(2, 3, dtype=torch.complex64, device=mojo_gpu))
    assert torch.equal(d[rows].cpu(), torch.ones(2, 3, dtype=torch.complex64))


def test_zero_dim_dims(mojo_gpu: str):
    """A 0-d tensor takes dim 0 / -1 (itself); the real view's pair dim must
    never be reached through it."""
    z = torch.tensor(1 + 2j)
    d = z.to(mojo_gpu)
    assert d.flip(0).item() == z.flip(0).item() == 1 + 2j
    assert d.flip(-1).item() == 1 + 2j
    with pytest.raises(IndexError):
        d.flip(1)
    with pytest.raises(IndexError, match="no dimensions"):
        d.roll(1, 0)
    assert d.roll(1).item() == 1 + 2j
    with pytest.raises(IndexError):
        d.index_select(1, torch.tensor([0]).to(mojo_gpu))
    with pytest.raises(IndexError, match="too many indices"):
        torch.ops.aten.index(
            _rand(4).to(mojo_gpu), [None, torch.tensor([0]).to(mojo_gpu)]
        )


def test_math_bit_write_back(mojo_gpu: str):
    """Writes through a conj / neg view reach the base (the Conjugate and
    Negative keys let _copy_from see the bits)."""
    x = _rand(3, 4)
    d = x.to(mojo_gpu)
    h = x.clone()
    d.conj().imag.mul_(2)
    h.conj().imag.mul_(2)
    _close(d, h)
    y = d.conj().imag
    y.copy_(torch.ones(3, 4).to(mojo_gpu))
    h.conj().imag.copy_(torch.ones(3, 4))
    _close(d, h)
    d.conj().copy_(torch.ones(3, 4).to(mojo_gpu))
    h.conj().copy_(torch.ones(3, 4))
    _close(d, h)
    d.conj().copy_(torch.full((3, 4), 1 + 2j, dtype=torch.complex64))
    h.conj().copy_(torch.full((3, 4), 1 + 2j, dtype=torch.complex64))
    _close(d, h)
    r = torch.randn(5)
    e = r.to(mojo_gpu)
    torch._neg_view(e).copy_(torch.arange(5.0).to(mojo_gpu))
    torch._neg_view(r).copy_(torch.arange(5.0))
    _close(e, r)


def test_index_func_dtypes_must_match(mojo_gpu: str):
    d = _rand(3, 4).to(mojo_gpu)
    rows = torch.tensor([0, 2]).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="same scalar type"):
        d.index_add(0, rows, torch.ones(2, 4).to(mojo_gpu))
    with pytest.raises(RuntimeError, match="same dtype"):
        d.index_copy(0, rows, torch.ones(2, 4).to(mojo_gpu))


def test_core_ops_run_natively(mojo_gpu: str):
    native.op_counting(True)
    native.op_counts_reset()
    d = _rand(3).to(mojo_gpu)
    _ = (d == d).all()
    torch.cat([d, d])
    assert native.op_count("aten::eq.Tensor") == 1
    # the complex cat runs the real one on the pairs: counted twice
    assert native.op_count("aten::cat") == 2
