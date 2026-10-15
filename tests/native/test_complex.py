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


# --- cat, flip, logical ops, where, outs and index_put --------------------------


def test_cat_converts_each_input_to_the_out_dtype(mojo_gpu: str):
    # 2**24 + 1 is not a float32: an int64 -> complex64 -> complex128 detour
    # would round it
    big = torch.tensor([16777217], dtype=torch.int64)
    z = torch.tensor([1 + 2j], dtype=torch.complex64)
    out = torch.empty(2, dtype=torch.complex128, device=mojo_gpu)
    torch.cat([big.to(mojo_gpu), z.to(mojo_gpu)], out=out)
    ref = torch.empty(2, dtype=torch.complex128)
    torch.cat([big, z], out=ref)
    assert torch.equal(out.cpu(), ref)
    assert out.cpu()[0].real.item() == 16777217


def test_copy_of_a_conj_and_neg_view_keeps_both_bits(mojo_gpu: str):
    x = _rand(5)
    v = torch._neg_view(x.to(mojo_gpu).conj())
    expected = -x.conj()
    assert torch.equal(v.cpu(), expected)
    assert torch.equal(v.clone().cpu(), expected)
    assert torch.equal(v.resolve_neg().resolve_conj().cpu(), expected)
    # mojo -> mojo and cpu -> mojo copies of a doubly-flagged source
    dst = torch.empty(5, dtype=torch.complex64, device=mojo_gpu)
    dst.copy_(v)
    assert torch.equal(dst.cpu(), expected)
    dst.copy_(torch._neg_view(x.conj()))
    assert torch.equal(dst.cpu(), expected)


@pytest.mark.parametrize("dtype", CDTYPES)
def test_clone_of_a_neg_view(mojo_gpu: str, dtype: torch.dtype):
    _skip_c128(mojo_gpu, dtype)
    x = _rand(3, 2, dtype=dtype)
    v = torch._neg_view(x.to(mojo_gpu))
    assert torch.equal(v.clone().cpu(), -x)
    assert torch.equal(v.cpu(), -x)
    assert torch.equal(v.resolve_neg().cpu(), -x)


def test_cat_of_zero_dim_and_flip_of_duplicate_dims_raise(mojo_gpu: str):
    z = torch.tensor(1 + 2j, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="zero-dimensional"):
        torch.cat([z])
    with pytest.raises(RuntimeError, match="zero-dimensional"):
        torch.cat([z, z])
    with pytest.raises(RuntimeError, match="multiple times"):
        z.flip([0, 0])
    with pytest.raises(RuntimeError, match="multiple times"):
        _rand(2, 3).to(mojo_gpu).flip([1, -1])


def test_logical_ops_on_complex(mojo_gpu: str):
    x = torch.tensor([0j, 1 + 0j, 1j, 0j])
    y = torch.tensor([1j, 0j, 2 + 0j, 0j])
    d, e = x.to(mojo_gpu), y.to(mojo_gpu)
    assert torch.equal(d.clone().logical_not_().cpu(), x.clone().logical_not_())
    assert torch.equal(d.clone().logical_and_(e).cpu(), x.clone().logical_and_(y))
    r = torch.tensor([0.0, 2.0, 0.0, 1.0])
    for fn in (torch.logical_and, torch.logical_or, torch.logical_xor):
        out = torch.empty(4, dtype=torch.complex64, device=mojo_gpu)
        ref = torch.empty(4, dtype=torch.complex64)
        fn(r.to(mojo_gpu), r.to(mojo_gpu), out=out)
        fn(r, r, out=ref)
        assert torch.equal(out.cpu(), ref)
    out = torch.empty(4, dtype=torch.complex64, device=mojo_gpu)
    torch.logical_not(r.to(mojo_gpu), out=out)
    assert torch.equal(out.cpu(), torch.logical_not(r).to(torch.complex64))
    torch.logical_not(d, out=out)
    assert torch.equal(out.cpu(), torch.logical_not(x).to(torch.complex64))
    with pytest.raises(RuntimeError, match="only supports bool tensor for result"):
        torch.any(d, out=torch.empty((), dtype=torch.complex64, device=mojo_gpu))


def test_where_device_and_rank(mojo_gpu: str):
    mask = torch.tensor([True, False, True])
    got = torch.where(mask.to(mojo_gpu), torch.tensor(1j), torch.tensor(2 + 0j))
    assert got.device.type == "mojo"
    assert torch.equal(got.cpu(), torch.where(mask, 1j, 2 + 0j))
    # rank 4 is the real kernel's limit: the complex path must not add a dim
    c = torch.rand(2, 2, 2, 2) > 0.5
    x = _rand(2, 2, 2, 2)
    assert torch.equal(
        torch.where(c.to(mojo_gpu), x.to(mojo_gpu), 0j).cpu(), torch.where(c, x, 0j)
    )
    x = _rand(2, 2, 2, 3)
    i = torch.randint(0, 3, (2, 2, 2, 2))
    assert torch.equal(x.to(mojo_gpu).gather(3, i.to(mojo_gpu)).cpu(), x.gather(3, i))
    m = torch.rand(2, 2, 2, 2) > 0.5
    x = _rand(2, 2, 2, 2)
    assert torch.equal(
        x.to(mojo_gpu).masked_fill(m.to(mojo_gpu), 1j).cpu(), x.masked_fill(m, 1j)
    )
    s = _rand(2, 2, 2, 1)
    idx = torch.tensor([1])
    assert torch.equal(
        x.to(mojo_gpu).index_add(3, idx.to(mojo_gpu), s.to(mojo_gpu)).cpu(),
        x.index_add(3, idx, s),
    )


def test_data_movement_outs_reject_overlap(mojo_gpu: str):
    x = _rand(4).to(mojo_gpu)
    idx = torch.tensor([0, 1], device=mojo_gpu)
    with pytest.raises(RuntimeError, match="unsupported operation"):
        torch.gather(x, 0, idx, out=x[:2])
    with pytest.raises(RuntimeError, match="unsupported operation"):
        torch.index_select(x, 0, idx, out=x[:2])
    with pytest.raises(RuntimeError, match="unsupported operation"):
        torch.cat([x[:2], x[2:]], out=x)
    src = _rand(2).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="unsupported operation"):
        x.index_add_(0, idx, x[1:3])
    with pytest.raises(RuntimeError, match="unsupported operation"):
        x.index_copy_(0, idx, x[1:3])
    x.index_add_(0, idx, src)  # disjoint operands still work


def test_outs_are_resized_before_the_internal_overlap_check(mojo_gpu: str):
    x = _rand(3, 4)
    idx = torch.tensor([[0, 1, 2, 0]])
    out = torch.empty(1, dtype=torch.complex64, device=mojo_gpu).expand(5)
    torch.gather(x.to(mojo_gpu), 0, idx.to(mojo_gpu), out=out)
    assert torch.equal(out.cpu(), x.gather(0, idx))
    # (a 4-element expanded out of the right shape is not resized: it raises)
    out = torch.empty(1, dtype=torch.complex64, device=mojo_gpu).expand(5)
    a = _rand(2)
    torch.cat([a.to(mojo_gpu), a.to(mojo_gpu)], out=out)
    assert torch.equal(out.cpu(), torch.cat([a, a]))
    with pytest.raises(RuntimeError, match="unsupported operation"):
        torch.cat([a.to(mojo_gpu)] * 2, out=out.view(-1)[:1].expand(4))


def test_outs_need_the_exact_dtype(mojo_gpu: str):
    x = _rand(3, 3).to(mojo_gpu)
    idx = torch.tensor([0, 2], device=mojo_gpu)
    out = torch.empty(0, dtype=torch.complex128, device=mojo_gpu)
    with pytest.raises(RuntimeError):
        torch.where(x.real > 0, x, x, out=out)
    with pytest.raises(RuntimeError):
        torch.where(x.real > 0, x.real, x.real, out=out)
    with pytest.raises(RuntimeError):
        torch.gather(x, 0, idx.view(1, 2).expand(3, 2), out=out)
    with pytest.raises(RuntimeError):
        torch.index_select(x, 0, idx, out=out)
    with pytest.raises(RuntimeError):
        torch.masked_select(x, x.real > 0, out=out)
    with pytest.raises(RuntimeError):
        torch.tril(x, out=out)


def test_input_device_errors(mojo_gpu: str):
    x = _rand(4).to(mojo_gpu)
    with pytest.raises(RuntimeError):
        x.gather(0, torch.tensor([0, 1]))
    with pytest.raises(RuntimeError):
        x.index_select(0, torch.tensor([0, 1]))
    with pytest.raises(RuntimeError):
        torch.cat([x, _rand(4)])
    with pytest.raises(RuntimeError):
        torch.eq(x, _rand(4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA torch")
def test_to_from_a_cuda_tensor_with_a_complex_dtype(mojo_gpu: str):
    x = torch.randn(5)
    got = x.cuda().to(mojo_gpu, dtype=torch.complex64)
    assert got.device.type == "mojo" and got.dtype == torch.complex64
    assert torch.equal(got.cpu(), x.to(torch.complex64))


def test_boolean_mask_assignment(mojo_gpu: str):
    x = _rand(3, 4)
    m = torch.rand(3, 4) > 0.5
    d = x.to(mojo_gpu)
    d[m.to(mojo_gpu)] = 3j  # ty: ignore[invalid-assignment]
    x[m] = 3j  # ty: ignore[invalid-assignment]
    assert torch.equal(d.cpu(), x)
    rows = torch.tensor([True, False, True])
    d[rows.to(mojo_gpu)] = torch.tensor(1 + 1j, device=mojo_gpu)
    x[rows] = 1 + 1j  # ty: ignore[invalid-assignment]
    assert torch.equal(d.cpu(), x)
    v = _rand(2, 4)
    d[rows.to(mojo_gpu)] = v.to(mojo_gpu)
    x[rows] = v
    assert torch.equal(d.cpu(), x)
    u8 = torch.tensor([0, 1, 1], dtype=torch.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d[u8.to(mojo_gpu)] = 2j  # ty: ignore[invalid-assignment]
        x[u8.bool()] = 2j  # ty: ignore[invalid-assignment]
    assert torch.equal(d.cpu(), x)


# --- overlap, unsigned scalars, index_add and general index_put ---------------


def test_in_place_logical_ops_never_resize_self(mojo_gpu: str):
    x = torch.tensor([1j], device=mojo_gpu)
    other = torch.tensor([1j, 0j], device=mojo_gpu)
    with pytest.raises(RuntimeError, match="doesn't match the broadcast shape"):
        x.logical_and_(other)
    with pytest.raises(RuntimeError, match="doesn't match the broadcast shape"):
        x.logical_or_(torch.tensor([1j, 0j, 1], device=mojo_gpu))
    with pytest.raises(RuntimeError, match="doesn't match the broadcast shape"):
        torch.logical_and(other, x, out=x)


@pytest.mark.parametrize("dtype", [torch.uint16, torch.uint32, torch.uint64])
def test_unsigned_cpu_scalars(mojo_gpu: str, dtype: torch.dtype):
    z = torch.tensor([256 + 0j, 70000 + 0j], device=mojo_gpu)
    s = torch.tensor(256, dtype=dtype)
    assert (z == s).cpu().tolist() == [True, False]
    got = torch.where(torch.tensor([True, False], device=mojo_gpu), z, s)
    assert got.cpu().tolist() == [256 + 0j, 256 + 0j]


@pytest.mark.parametrize("dtype", CDTYPES)
def test_index_add_multiplies_like_complex(mojo_gpu: str, dtype: torch.dtype):
    _skip_c128(mojo_gpu, dtype)
    src = torch.tensor([complex(float("inf"), 2)], dtype=dtype)
    idx = torch.tensor([0])
    for kw in ({}, {"alpha": 2}):
        got = torch.zeros(1, dtype=dtype, device=mojo_gpu).index_add(
            0, idx.to(mojo_gpu), src.to(mojo_gpu), **kw
        )
        # CUDA: self + alpha * source, a complex multiply even for alpha=1
        assert got.cpu()[0].real.item() == float("inf")
        assert got.cpu()[0].imag.isnan()
    x = _rand(2, dtype=dtype)
    s = _rand(2, dtype=dtype)
    i2 = torch.tensor([0, 1])
    _close(
        x.to(mojo_gpu).index_add(0, i2.to(mojo_gpu), s.to(mojo_gpu), alpha=1 - 2j),
        x.index_add(0, i2, s, alpha=1 - 2j),
    )


def test_outs_check_overlap_with_the_original_operands(mojo_gpu: str):
    z = torch.tensor([1j, 0j, 2 + 0j], device=mojo_gpu)
    with pytest.raises(RuntimeError, match="unsupported operation"):
        torch.logical_not(z, out=z.view(torch.float32)[1:4])
    with pytest.raises(RuntimeError, match="unsupported operation"):
        torch.masked_select(
            z, torch.tensor([True, False, True], device=mojo_gpu), out=z[:2]
        )
    w = torch.tensor([1j, 2j, 3j, 4j], device=mojo_gpu)
    with pytest.raises(RuntimeError, match="unsupported operation"):
        w[:3].copy_(w[1:].conj())
    with pytest.raises(RuntimeError, match="unsupported operation"):
        w[:3].copy_(torch._neg_view(w[1:]))


def test_masked_fill_out_needs_the_exact_dtype(mojo_gpu: str):
    x = torch.tensor([1j, 2j], device=mojo_gpu)
    m = torch.tensor([True, False], device=mojo_gpu)
    out = torch.empty(2, dtype=torch.complex128, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
        torch.ops.aten.masked_fill.Scalar_out(x, m, 3.0, out=out)


def test_cat_of_real_inputs_into_a_complex_out(mojo_gpu: str):
    out = torch.empty(1, dtype=torch.complex128, device=mojo_gpu)
    torch.cat([torch.tensor([16777217], device=mojo_gpu)], out=out)
    assert out.cpu().real.item() == 16777217
    out = torch.empty(2, dtype=torch.complex64, device=mojo_gpu)
    torch.cat(
        [torch.tensor([1.5], device=mojo_gpu), torch.tensor([2.0], device=mojo_gpu)],
        out=out,
    )
    assert out.cpu().tolist() == [1.5 + 0j, 2 + 0j]
    # (CUDA raises ValueError; this device can only raise RuntimeError)
    with pytest.raises((RuntimeError, ValueError), match="non-empty list"):
        torch.cat([], out=out)


def test_index_put_general(mojo_gpu: str):
    x = _rand(3, 4)
    d = x.to(mojo_gpu)
    i = torch.tensor([0, 2])
    # a CPU scalar value and CPU indices
    d.index_put_((i.to(mojo_gpu),), torch.tensor(1j))
    x.index_put_((i,), torch.tensor(1j))
    assert torch.equal(d.cpu(), x)
    d[torch.tensor([1])] = 2j  # ty: ignore[invalid-assignment]
    x[torch.tensor([1])] = 2j  # ty: ignore[invalid-assignment]
    assert torch.equal(d.cpu(), x)
    # a non-leading index, a 2-d index, a column mask
    c = torch.tensor([1, 3])
    d[:, c.to(mojo_gpu)] = 2j  # ty: ignore[invalid-assignment]
    x[:, c] = 2j  # ty: ignore[invalid-assignment]
    assert torch.equal(d.cpu(), x)
    i2 = torch.tensor([[0, 1], [2, 3]])
    v = torch.arange(12.0).reshape(3, 2, 2).to(torch.complex64)
    d[:, i2.to(mojo_gpu)] = v.to(mojo_gpu)
    x[:, i2] = v
    assert torch.equal(d.cpu(), x)
    cm = torch.tensor([True, False, True, False])
    d[:, cm.to(mojo_gpu)] = 5j  # ty: ignore[invalid-assignment]
    x[:, cm] = 5j  # ty: ignore[invalid-assignment]
    assert torch.equal(d.cpu(), x)
    # rank 5, a middle index; a transposed (non-contiguous) self
    y = _rand(2, 3, 2, 2, 2)
    e = y.to(mojo_gpu)
    e[:, :, torch.tensor([1], device=mojo_gpu)] = 9j  # ty: ignore[invalid-assignment]
    y[:, :, torch.tensor([1])] = 9j  # ty: ignore[invalid-assignment]
    assert torch.equal(e.cpu(), y)
    t = _rand(4, 3)
    f = t.to(mojo_gpu).t()
    f[torch.tensor([0, 2], device=mojo_gpu)] = 7j  # ty: ignore[invalid-assignment]
    tt = t.t()
    tt[torch.tensor([0, 2])] = 7j  # ty: ignore[invalid-assignment]
    assert torch.equal(f.cpu(), tt)
    # accumulate with duplicates
    a = torch.zeros(3, dtype=torch.complex64)
    g = a.to(mojo_gpu)
    k = torch.tensor([1, 1, 2])
    g.index_put_(
        (k.to(mojo_gpu),), torch.tensor(1 + 1j, device=mojo_gpu), accumulate=True
    )
    a.index_put_((k,), torch.tensor(1 + 1j), accumulate=True)
    assert torch.equal(g.cpu(), a)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        d[:, c.to(mojo_gpu)] = torch.ones(3, dtype=torch.complex64, device=mojo_gpu)


def test_index_put_on_an_expanded_self(mojo_gpu: str):
    # deprecated, but CUDA writes the shared row: every row reads 7j
    x = torch.zeros(1, 4, dtype=torch.complex64, device=mojo_gpu).expand(3, 4)
    x.index_put_(
        (torch.tensor([1], device=mojo_gpu),), torch.tensor(7j, device=mojo_gpu)
    )
    assert torch.equal(x.cpu(), torch.full((3, 4), 7j))


# --- copies with math bits, identity, overflow and CPU operands ---------------


def test_copy_converts_before_applying_math_bits(mojo_gpu: str):
    z = torch.zeros(1, dtype=torch.complex64, device=mojo_gpu)
    z.conj().imag.copy_(torch.tensor([255], dtype=torch.uint8, device=mojo_gpu))
    assert z.cpu().tolist() == [-255j]
    n = torch.zeros(3, dtype=torch.complex64, device=mojo_gpu)
    torch._neg_view(n).copy_(
        torch.tensor([255, 1, 0], dtype=torch.uint8, device=mojo_gpu)
    )
    assert n.cpu().real.tolist() == [-255.0, -1.0, 0.0]
    # a different bit on each side is declined, not emulated
    f = torch.zeros(2, dtype=torch.complex128, device=mojo_gpu)
    with pytest.raises(NotImplementedError, match="different math bits"):
        f.conj().copy_(torch._neg_view(torch.tensor([1 + 2j, 3j], device=mojo_gpu)))
    assert f.cpu().tolist() == [0j, 0j]
    # (stock CUDA has no kernel for these two; matched)
    with pytest.raises(NotImplementedError, match="neg_cuda"):
        torch.zeros(2, dtype=torch.bool, device=mojo_gpu).copy_(
            torch._neg_view(torch.ones(2, device=mojo_gpu))
        )
    with pytest.raises(NotImplementedError, match="neg_conj_cuda"):
        torch.zeros(2, dtype=torch.complex64, device=mojo_gpu).conj().copy_(
            torch._neg_view(torch.ones(2, device=mojo_gpu))
        )


def test_copy_overlap_checks(mojo_gpu: str):
    x = _rand(3, 3).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="unsupported operation"):
        x.copy_(x.t())
    e = torch.zeros(1, dtype=torch.complex64, device=mojo_gpu).expand(3)
    with pytest.raises(RuntimeError, match="unsupported operation"):
        e.copy_(_rand(3).to(mojo_gpu))
    with pytest.raises(RuntimeError, match="unsupported operation"):
        e.fill_(torch.tensor(1j, device=mojo_gpu))
    x.copy_(x)  # the same tensor: nothing to do


def test_out_identity_is_the_tensor_not_its_data(mojo_gpu: str):
    a = torch.zeros(0, 1, dtype=torch.complex64, device=mojo_gpu)
    b = torch.zeros(0, 2, dtype=torch.complex64, device=mojo_gpu)
    out = torch.zeros(0, 1, dtype=torch.bool, device=mojo_gpu)
    torch.logical_and(a, b, out=out)
    assert out.shape == (0, 2)
    v = torch.tensor([1j, 0j, 1 + 0j], device=mojo_gpu)
    w = _rand(2, 3).to(mojo_gpu)
    o = v.view(3)
    torch.logical_and(v, w, out=o)
    assert o.shape == (2, 3)


def test_index_put_value_overlap_and_masked_fill_dispatch(mojo_gpu: str):
    x = _rand(3, 3).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="unsupported operation"):
        x.index_put_((torch.tensor([0, 1], device=mojo_gpu),), x.t())
    with pytest.raises(RuntimeError, match="unsupported operation"):
        torch.zeros(1, 3, dtype=torch.complex64, device=mojo_gpu).expand(
            3, 3
        ).index_put_(
            (torch.tensor([0], device=mojo_gpu),),
            _rand(3).to(mojo_gpu),
            accumulate=True,
        )
    # one mask + a one-element CPU value of another dtype: masked_fill_
    a = _rand(2, 3)
    m = torch.rand(2, 3) > 0.5
    d = a.to(mojo_gpu)
    d.index_put_((m.to(mojo_gpu),), torch.tensor(2.5))
    a.index_put_((m,), torch.tensor(2.5))
    assert torch.equal(d.cpu(), a)
    rows = torch.tensor([True, False])
    d.index_put_((rows.to(mojo_gpu),), torch.tensor(-1.0))
    a.index_put_((rows,), torch.tensor(-1.0))
    assert torch.equal(d.cpu(), a)
    with pytest.raises(RuntimeError, match="Index put requires"):
        d.index_put_((torch.tensor([0], device=mojo_gpu),), torch.tensor(2.5))
    with pytest.raises(
        RuntimeError, match="expected scalar type Float but found ComplexFloat"
    ):
        d.index_put_((m.to(mojo_gpu),), torch.tensor(2.5), accumulate=True)


def test_conj_physical_out_and_in_place(mojo_gpu: str):
    z = _rand(3).to(mojo_gpu)
    with pytest.raises(
        RuntimeError, match="Found dtype ComplexDouble but expected ComplexFloat"
    ):
        torch.conj_physical(
            z, out=torch.empty(3, dtype=torch.complex128, device=mojo_gpu)
        )
    bad = torch.empty(5, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="Found dtype Float"):
        torch.conj_physical(z, out=bad)
    assert bad.shape == (5,)  # refused before any resize
    with pytest.raises(RuntimeError, match="unsupported operation"):
        torch.tensor([1j], device=mojo_gpu).expand(3).conj_physical_()


def test_scalar_overflow_and_cpu_operands(mojo_gpu: str):
    a = _rand(2, 3).to(mojo_gpu)
    m = torch.tensor([[True, False, True], [False, True, False]], device=mojo_gpu)
    with pytest.raises(RuntimeError, match="without overflow"):
        a.clone().fill_(1e100)
    with pytest.raises(RuntimeError, match="without overflow"):
        torch.where(m, a, 1e100)
    # a CPU 0-d condition, CPU indices, device errors with CUDA's type
    got = torch.where(torch.tensor(True), a, torch.zeros_like(a))
    assert torch.equal(got.cpu(), a.cpu())
    assert torch.equal(a[torch.tensor([1, 0])].cpu(), a.cpu()[torch.tensor([1, 0])])
    with pytest.raises(RuntimeError, match="same device"):
        torch.masked_select(a, m.cpu())
    with pytest.raises(RuntimeError, match="same device"):
        torch.logical_or(torch.tensor(1.0), a)
    with pytest.raises(RuntimeError, match="input types can't be cast"):
        torch.cat([a, a], out=torch.empty(0, device=mojo_gpu))


# --- Scalar checks before writes, fill_ of real tensors, mask Nones ------------


def test_math_bit_source_converts_before_its_bit(mojo_gpu: str):
    out = torch.zeros(1, dtype=torch.complex64, device=mojo_gpu)
    out.copy_(torch._neg_view(torch.tensor([255], dtype=torch.uint8, device=mojo_gpu)))
    assert out.cpu().tolist() == [-255 + 0j]
    i8 = torch.zeros(1, dtype=torch.int64, device=mojo_gpu)
    i8.copy_(torch._neg_view(torch.tensor([-128], dtype=torch.int8, device=mojo_gpu)))
    assert i8.cpu().tolist() == [128]
    # an empty copy chooses no kernel: nothing to refuse
    torch.zeros(0, dtype=torch.bool, device=mojo_gpu).copy_(
        torch._neg_view(torch.zeros(0, device=mojo_gpu))
    )


def test_scalars_are_checked_before_anything_is_written(mojo_gpu: str):
    x = _rand(2, 3).to(mojo_gpu)
    before = x.cpu()
    m = torch.ones(2, 3, dtype=torch.bool, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="without overflow"):
        x.masked_fill_(m, 7 + 1e100j)
    with pytest.raises(RuntimeError, match="without overflow"):
        x.fill_(torch.tensor(1e100, dtype=torch.float64))
    with pytest.raises(RuntimeError, match="without overflow"):
        x.fill_(torch.tensor(1e39j, dtype=torch.complex128))
    with pytest.raises(RuntimeError, match="without overflow"):
        x.index_add_(0, torch.tensor([0, 1], device=mojo_gpu), x.clone(), alpha=1e100)
    assert torch.equal(x.cpu(), before)


def test_fill_a_real_tensor_with_a_complex_value(mojo_gpu: str):
    t = torch.zeros(3, device=mojo_gpu)
    t.fill_(torch.tensor(2 + 3j, device=mojo_gpu))  # copy_: the real part
    assert t.cpu().tolist() == [2.0, 2.0, 2.0]
    with pytest.raises(RuntimeError, match="value cannot be converted to type float"):
        t.fill_(torch.tensor(2 + 3j))  # .item(): a checked conversion
    t.fill_(torch.tensor(5 + 0j))
    assert t.cpu().tolist() == [5.0, 5.0, 5.0]


def test_index_put_mask_with_a_trailing_none(mojo_gpu: str):
    x = _rand(2, 2, 4)
    d = x.to(mojo_gpu)
    m = torch.tensor([True, False])
    torch.ops.aten.index_put_(d, [m.to(mojo_gpu), None], torch.tensor(5 + 1j))
    torch.ops.aten.index_put_(x, [m, None], torch.tensor(5 + 1j))
    assert torch.equal(d.cpu(), x)


def test_conj_physical_of_a_real_tensor_into_a_complex_out(mojo_gpu: str):
    with pytest.raises(
        RuntimeError, match="Found dtype ComplexFloat but expected Float"
    ):
        torch.conj_physical(
            torch.ones(2, device=mojo_gpu),
            out=torch.empty(2, dtype=torch.complex64, device=mojo_gpu),
        )


# --- checked conversions, device-source bits, empty tensors, equal -----------


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (torch.float32, 1e100 + 0j),
        (torch.uint8, -1 + 0j),
        (torch.int32, 1e100 + 0j),
        (torch.float16, 70000 + 0j),
        (torch.float32, 5 + 1j),
    ],
)
def test_real_fill_from_a_complex_value_is_checked(
    mojo_gpu: str, dtype: torch.dtype, value: complex
):
    t = torch.zeros(3, dtype=dtype, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="without overflow"):
        t.fill_(torch.tensor(value, dtype=torch.complex128))
    assert t.cpu().tolist() == [0, 0, 0]
    with pytest.raises(RuntimeError, match="without overflow"):
        t.masked_fill_(
            torch.ones(3, dtype=torch.bool, device=mojo_gpu),
            torch.tensor(value, dtype=torch.complex128),
        )
    assert t.cpu().tolist() == [0, 0, 0]


def test_real_fill_scalar_is_checked(mojo_gpu: str):
    with pytest.raises(RuntimeError, match="value cannot be converted to type uint8_t"):
        torch.zeros(3, dtype=torch.uint8, device=mojo_gpu).fill_(-1.0)
    with pytest.raises(
        RuntimeError, match="value cannot be converted to type int without"
    ):
        torch.zeros(3, dtype=torch.int32, device=mojo_gpu).fill_(1e100)
    # C++'s unsigned wrap of an in-range negative integer is accepted
    assert torch.zeros(1, dtype=torch.uint8, device=mojo_gpu).fill_(-1).item() == 255
    assert torch.zeros(1, dtype=torch.bool, device=mojo_gpu).fill_(1e100).item() is True


def test_masked_fill_of_a_real_tensor_with_a_complex_value(mojo_gpu: str):
    m = torch.ones(3, dtype=torch.bool, device=mojo_gpu)
    for v in (torch.tensor(5 + 0j), torch.tensor(5 + 0j, device=mojo_gpu), 5 + 0j):
        assert (
            torch.zeros(3, device=mojo_gpu).masked_fill_(m, v).cpu().tolist()
            == [5.0] * 3
        )
        assert (
            torch.zeros(3, device=mojo_gpu).masked_fill(m, v).cpu().tolist()
            == [5.0] * 3
        )


def test_device_source_bits_follow_cuda(mojo_gpu: str):
    # device -> host: the bit resolves in the SOURCE dtype (uint8 wraps)
    host = torch.zeros(1, dtype=torch.int64)
    host.copy_(torch._neg_view(torch.tensor([255], dtype=torch.uint8, device=mojo_gpu)))
    assert host.tolist() == [1]
    host.copy_(torch._neg_view(torch.tensor([-128], dtype=torch.int8, device=mojo_gpu)))
    assert host.tolist() == [-128]
    # device -> device: converted first, then the bit
    dev = torch.zeros(1, dtype=torch.int64, device=mojo_gpu)
    dev.copy_(torch._neg_view(torch.tensor([255], dtype=torch.uint8, device=mojo_gpu)))
    assert dev.cpu().tolist() == [-255]
    # neg + conj into bool runs the fused kernel on the complex source
    b = torch.zeros(1, dtype=torch.bool, device=mojo_gpu)
    b.copy_(torch._neg_view(torch.tensor([1 + 2j], device=mojo_gpu).conj()))
    assert b.cpu().tolist() == [True]
    with pytest.raises(
        NotImplementedError, match="neg_conj_cuda\" not implemented for 'Long'"
    ):
        torch.zeros(1, dtype=torch.complex64, device=mojo_gpu).conj().copy_(
            torch._neg_view(torch.tensor([1], device=mojo_gpu))
        )


def test_empty_tensors_still_validate(mojo_gpu: str):
    with pytest.raises(RuntimeError, match="without overflow"):
        torch.zeros(0, dtype=torch.complex64, device=mojo_gpu).fill_(
            torch.tensor(1e100, dtype=torch.float64)
        )
    with pytest.raises(
        RuntimeError, match=r"tensor a \(0\) must match the size of tensor b \(3\)"
    ):
        torch.zeros(0, device=mojo_gpu).copy_(
            torch._neg_view(torch.randn(3, device=mojo_gpu))
        )
    with pytest.raises(
        RuntimeError, match=r"tensor a \(0\) must match the size of tensor b \(3\)"
    ):
        torch.zeros(0, dtype=torch.complex64, device=mojo_gpu).copy_(
            _rand(3).to(mojo_gpu).conj()
        )
    torch.zeros(0, device=mojo_gpu).copy_(
        torch._neg_view(torch.randn(1, device=mojo_gpu))
    )


def test_equal_of_the_same_elements_with_nan(mojo_gpu: str):
    x = torch.tensor([float("nan") + 0j], device=mojo_gpu)
    assert torch.equal(x, x)
    assert torch.equal(x, x.view(1))
    assert not torch.equal(x, x.clone())


def test_where_and_index_put_messages(mojo_gpu: str):
    with pytest.raises(
        RuntimeError,
        match=r"tensor a \(2\) must match the size of tensor b \(3\) at non-singleton dimension 0",
    ):
        torch.where(
            torch.ones(2, dtype=torch.bool, device=mojo_gpu), _rand(3).to(mojo_gpu), 0j
        )
    with pytest.raises(
        RuntimeError, match="expected scalar type ComplexDouble but found ComplexFloat"
    ):
        torch.zeros(3, dtype=torch.complex64, device=mojo_gpu).index_put_(
            (torch.tensor([0], device=mojo_gpu),),
            torch.ones(1, dtype=torch.complex128, device=mojo_gpu),
            accumulate=True,
        )


# --- masks, broadcasting, bit cancellation and declined copies ----------------


def test_masked_fill_mask_must_be_bool(mojo_gpu: str):
    # (a complex mask once routed a real self back to itself, forever)
    cases = [
        (torch.float32, torch.complex64),
        (torch.complex64, torch.complex64),
        (torch.complex64, torch.uint8),
    ]
    for self_dt, mask_dt in cases:
        x = torch.zeros(3, dtype=self_dt, device=mojo_gpu)
        with pytest.raises(RuntimeError, match="only supports boolean masks"):
            x.masked_fill_(torch.ones(3, dtype=mask_dt, device=mojo_gpu), 1.0)


def test_real_masked_fill_with_a_complex_value_broadcasts(mojo_gpu: str):
    m = torch.tensor([[True], [False]], device=mojo_gpu)
    got = torch.ones(1, 3, device=mojo_gpu).masked_fill(m, 5 + 0j)
    assert got.cpu().tolist() == [[5.0] * 3, [1.0] * 3]
    m2 = torch.ones(2, 3, dtype=torch.bool, device=mojo_gpu)
    for v in (1 + 0j, torch.tensor(1 + 0j)):
        got = torch.zeros(3, device=mojo_gpu).masked_fill(m2, v)
        assert got.shape == (2, 3) and got.cpu().eq(1).all()


def test_matching_bits_cancel_and_mixed_bits_decline(mojo_gpu: str):
    b = torch.zeros(3, dtype=torch.bool, device=mojo_gpu)
    torch._neg_view(b).copy_(
        torch._neg_view(torch.tensor([1.0, 0.0, -2.0], device=mojo_gpu))
    )
    assert b.cpu().tolist() == [True, False, True]
    o = torch.zeros(2, dtype=torch.complex64, device=mojo_gpu)
    with pytest.raises(NotImplementedError):
        torch._neg_view(o).copy_(torch.tensor([1j, 2], device=mojo_gpu).conj())
    i = torch.zeros(1, dtype=torch.int32, device=mojo_gpu)
    with pytest.raises(NotImplementedError, match="neg \\+ conj"):
        i.copy_(
            torch._neg_view(torch.tensor([-2147483648 + 1j], device=mojo_gpu).conj())
        )
    assert i.cpu().tolist() == [0]


def test_non_blocking_device_to_host_bit_copies(mojo_gpu: str):
    # with a dtype change stock CUDA drops the bit (int8 [5] negated lands
    # as [5]): declined here, before anything is written
    src = torch._neg_view(torch.tensor([5], dtype=torch.int8, device=mojo_gpu))
    h = torch.zeros(1, dtype=torch.int64)
    with pytest.raises(NotImplementedError, match="non-blocking"):
        h.copy_(src, non_blocking=True)
    assert h.tolist() == [0]
    h.copy_(src)  # blocking: resolved in the source dtype
    assert h.tolist() == [-5]
    same = torch.zeros(1, dtype=torch.int8)
    same.copy_(src, non_blocking=True)
    torch.accelerator.synchronize()
    assert same.tolist() == [-5]


def test_unsigned_fill_ranges(mojo_gpu: str):
    torch.empty(0, dtype=torch.uint64, device=mojo_gpu).fill_(1e19)
    # past int64: the exact 64 bits (as CUDA)
    assert (
        torch.zeros(2, dtype=torch.uint64, device=mojo_gpu).fill_(1e19).cpu().tolist()
        == [10**19] * 2
    )
    assert (
        torch.zeros(1, dtype=torch.uint64, device=mojo_gpu).fill_(-1).cpu().item()
        == 2**64 - 1
    )
    with pytest.raises(RuntimeError, match="uint32_t"):
        torch.zeros(1, dtype=torch.uint32, device=mojo_gpu).fill_(5e9)
    with pytest.raises(RuntimeError, match="uint8_t"):
        torch.zeros(1, dtype=torch.uint8, device=mojo_gpu).fill_(-(2**63))


def test_complex_scalar_fill_of_a_real_tensor(mojo_gpu: str):
    assert torch.zeros(3, device=mojo_gpu).fill_(1 + 0j).cpu().tolist() == [1.0] * 3
    assert torch.full(
        (2,), 1 + 0j, dtype=torch.float32, device=mojo_gpu
    ).cpu().tolist() == [1.0, 1.0]
    assert torch.zeros(2, dtype=torch.bool, device=mojo_gpu).fill_(
        1j
    ).cpu().tolist() == [True, True]
    with pytest.raises(RuntimeError, match="value cannot be converted to type float"):
        torch.zeros(3, device=mojo_gpu).fill_(1 + 1j)


# --- device checks of the multi-tensor complex ops ------------------------------


def test_masked_fill_device_checks(mojo_gpu: str):
    x = torch.tensor([1 + 1j, 2j, 3 + 0j])
    m = torch.tensor([True, False, True], device=mojo_gpu)
    for fn in (
        lambda: x.masked_fill(m, 5j),
        lambda: x.clone().masked_fill_(m, 5j),
        lambda: x[:0].masked_fill(m[:0], 5j),
    ):
        with pytest.raises(
            RuntimeError, match="expected self and mask to be on the same device"
        ):
            fn()
    with pytest.raises(RuntimeError, match="Expected inputs to be on same device"):
        x.masked_fill(m, torch.tensor(5j, device=mojo_gpu))


def test_cpu_tensor_with_device_operands(mojo_gpu: str):
    x = torch.tensor([1 + 1j, 2j, 3 + 0j])
    with pytest.raises(RuntimeError, match="indices should be either on cpu"):
        x[torch.tensor([0, 2], device=mojo_gpu)]
    y = x.clone()
    y.fill_(torch.tensor(1j, device=mojo_gpu))  # a 0-d device value: its item()
    assert y.tolist() == [1j, 1j, 1j]


def test_fill_from_a_host_tensor_value_is_checked(mojo_gpu: str):
    # one fill_.Tensor implementation, with fill_.Scalar's checked conversion
    t = torch.zeros(3, dtype=torch.uint16, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="uint16_t"):
        t.fill_(torch.tensor(-1.0, dtype=torch.float64))
    assert t.cpu().tolist() == [0, 0, 0]
    with pytest.raises(RuntimeError, match="int8_t"):
        torch.zeros(2, dtype=torch.int8, device=mojo_gpu).fill_(torch.tensor(300))
    t.fill_(torch.tensor(7.0))
    assert t.cpu().tolist() == [7, 7, 7]


# --- masked_fill outs and the rounding of fill values ---------------------------


def test_masked_fill_outs_take_a_complex_value_on_a_real_tensor(mojo_gpu: str):
    m = torch.ones(2, dtype=torch.bool, device=mojo_gpu)
    for v in (torch.tensor(5 + 0j), 5 + 0j):
        fn = (
            torch.ops.aten.masked_fill.Tensor_out
            if isinstance(v, torch.Tensor)
            else torch.ops.aten.masked_fill.Scalar_out
        )
        out = torch.empty(2, device=mojo_gpu)
        fn(torch.zeros(2, device=mojo_gpu), m, v, out=out)
        assert out.cpu().tolist() == [5.0, 5.0]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_fills_round_through_float(mojo_gpu: str, dtype: torch.dtype):
    # c10's Half / BFloat16 from double go through float: two roundings
    m = torch.ones(1, dtype=torch.bool, device=mojo_gpu)
    for v in (1 + 2**-11 + 2**-30, 1 + 2**-8 + 2**-30, -(1 + 2**-11 + 2**-30)):
        want = torch.tensor(v, dtype=torch.float64).to(dtype)  # CPU torch: same chain
        for got in (
            torch.zeros(1, dtype=dtype, device=mojo_gpu).fill_(v),
            torch.zeros(1, dtype=dtype, device=mojo_gpu).fill_(complex(v, 0)),
            torch.zeros(1, dtype=dtype, device=mojo_gpu).fill_(
                torch.tensor(complex(v, 0), dtype=torch.complex128)
            ),
            torch.zeros(1, dtype=dtype, device=mojo_gpu).masked_fill_(m, complex(v, 0)),
            torch.full((1,), complex(v, 0), dtype=dtype, device=mojo_gpu),
        ):
            assert torch.equal(got.cpu(), want.reshape(1))


def test_fill_a_host_tensor_with_a_device_value(mojo_gpu: str):
    t = torch.zeros(3)
    t.fill_(torch.tensor(2.0, device=mojo_gpu))
    assert t.tolist() == [2.0, 2.0, 2.0]


# --- uint64 fill values, masked_fill out dtypes, half masked_fill rounding ------


def test_uint64_fill_values(mojo_gpu: str):
    big = torch.tensor(2**63, dtype=torch.uint64)
    dev = big.to(mojo_gpu)
    # the same device: a copy_ (unsigned into a float, wrapping into an int),
    # empty destinations included
    assert (
        torch.zeros(1, dtype=torch.float64, device=mojo_gpu).fill_(dev).cpu().item()
        == 2.0**63
    )
    assert torch.zeros(1, dtype=torch.int64, device=mojo_gpu).fill_(
        dev
    ).cpu().item() == -(2**63)
    assert (
        torch.zeros(1, dtype=torch.int32, device=mojo_gpu).fill_(dev).cpu().item() == 0
    )
    torch.zeros(0, dtype=torch.int64, device=mojo_gpu).fill_(dev)
    # another device, a uint64 destination: the same 64 bits (strided too)
    u = torch.zeros(4, dtype=torch.uint64, device=mojo_gpu)
    u[::2].fill_(big)
    assert u.cpu().tolist() == [2**63, 0, 2**63, 0]
    # another device, any other destination: declined before writing
    for host_self in (True, False):
        for dt in (torch.float64, torch.int64, torch.complex64, torch.bool):
            t = (
                torch.zeros(1, dtype=dt)
                if host_self
                else torch.zeros(1, dtype=dt, device=mojo_gpu)
            )
            with pytest.raises(NotImplementedError, match="2\\*\\*63"):
                t.fill_(dev if host_self else big)
            assert t.cpu().item() == 0
    # below 2**63: exact, one rounding into float
    small = torch.tensor(2**62 + 2**39 + 1, dtype=torch.uint64)
    want = torch.zeros(1).fill_(small).item()
    assert torch.zeros(1, device=mojo_gpu).fill_(small).cpu().item() == want
    assert torch.zeros(1).fill_(small.to(mojo_gpu)).item() == want


@pytest.mark.parametrize(
    "out_dtype", [torch.int32, torch.float64, torch.complex64, torch.bool]
)
def test_masked_fill_out_resizes_before_its_dtype_check(
    mojo_gpu: str, out_dtype: torch.dtype
):
    m = torch.ones(2, dtype=torch.bool, device=mojo_gpu)
    for call in (
        lambda o: torch.ops.aten.masked_fill.Scalar_out(
            torch.zeros(2, device=mojo_gpu), m, 5 + 0j, out=o
        ),
        lambda o: torch.ops.aten.masked_fill.Tensor_out(
            torch.zeros(2, device=mojo_gpu), m, torch.tensor(5 + 0j), out=o
        ),
        lambda o: torch.ops.aten.masked_fill.Scalar_out(
            torch.zeros(2, device=mojo_gpu), m, 5.0, out=o
        ),
        lambda o: torch.ops.aten.masked_fill.Tensor_out(
            torch.zeros(2, device=mojo_gpu), m, torch.tensor(5.0), out=o
        ),
    ):
        out = torch.empty(0, dtype=out_dtype, device=mojo_gpu)
        with pytest.raises(
            RuntimeError, match="Expected out tensor to have dtype float"
        ):
            call(out)
        assert out.shape == (2,)


@pytest.mark.parametrize(
    ("dtype", "value"),
    [(torch.float16, 1 + 2**-11 + 2**-30), (torch.bfloat16, 1 + 2**-8 + 2**-30)],
)
def test_real_masked_fill_rounds_half_through_float(
    mojo_gpu: str, dtype: torch.dtype, value: float
):
    want = torch.tensor(value, dtype=torch.float64).to(dtype).item()
    m = torch.ones(3, dtype=torch.bool, device=mojo_gpu)
    got = torch.zeros(3, dtype=dtype, device=mojo_gpu).masked_fill_(m, value)
    assert got.cpu().tolist() == [want] * 3
    out = torch.empty(3, dtype=dtype, device=mojo_gpu)
    torch.ops.aten.masked_fill.Scalar_out(
        torch.zeros(3, dtype=dtype, device=mojo_gpu), m, value, out=out
    )
    assert out.cpu().tolist() == [want] * 3


def test_masked_fill_out_checks_the_value_first_and_resizes_host_outs(mojo_gpu: str):
    m = torch.ones(2, dtype=torch.bool, device=mojo_gpu)
    out = torch.empty(0, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="int8_t without overflow"):
        torch.ops.aten.masked_fill.Scalar_out(
            torch.zeros(2, dtype=torch.int8, device=mojo_gpu), m, 1000, out=out
        )
    assert out.shape == (0,)
    out64 = torch.empty(0, dtype=torch.float64, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="float without overflow"):
        torch.ops.aten.masked_fill.Scalar_out(
            torch.zeros(2, device=mojo_gpu), m, 1e100, out=out64
        )
    assert out64.shape == (0,)
    for self_dt, out_dt in (
        (torch.float32, torch.int32),
        (torch.complex64, torch.float32),
    ):
        host = torch.empty(0, dtype=out_dt)
        with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
            torch.ops.aten.masked_fill.Scalar_out(
                torch.zeros(2, dtype=self_dt, device=mojo_gpu), m, 5.0, out=host
            )
        assert host.shape == (2,)


# --- values read from uint64 tensors, on every fill entry point -----------------


def test_uint64_values_on_every_fill_entry_point(mojo_gpu: str):
    m = torch.ones(2, dtype=torch.bool, device=mojo_gpu)
    top = torch.tensor(2**64 - 1, dtype=torch.uint64, device=mojo_gpu)
    for dt in (torch.complex64, torch.float32):
        want = torch.tensor(2**64 - 1, dtype=torch.uint64).to(
            dt
        )  # CPU torch: read unsigned
        got = torch.zeros(2, dtype=dt, device=mojo_gpu).masked_fill_(m, top)
        assert torch.equal(got.cpu(), want.expand(2))
        got = torch.zeros(2, dtype=dt, device=mojo_gpu).masked_fill(m, top)
        assert torch.equal(got.cpu(), want.expand(2))
    with pytest.raises(RuntimeError, match="int64_t without overflow"):
        torch.zeros(2, dtype=torch.int64, device=mojo_gpu).masked_fill_(m, top)
    # one rounding from a uint64 below 2**63 into a single-precision dtype
    mid = torch.tensor(2**62 + 2**38 + 1, dtype=torch.uint64)
    got = torch.zeros(2, dtype=torch.complex64, device=mojo_gpu).fill_(mid)
    assert got.cpu()[0].real.item() == 2**62 + 2**39
    # another device, 2**63 or more: declined before writing
    t = torch.zeros(2, dtype=torch.complex64, device=mojo_gpu)
    with pytest.raises(NotImplementedError, match="2\\*\\*63 or more"):
        t.masked_fill_(m, top.cpu())
    assert t.cpu().tolist() == [0j, 0j]
    # an index_put_ mask with a host value: its item() (a uint64 one too)
    x = torch.zeros(2, dtype=torch.int64, device=mojo_gpu)
    x[m] = torch.tensor(2**40 + 1)
    assert x.cpu().tolist() == [2**40 + 1] * 2
    x[m] = torch.tensor(7, dtype=torch.uint64)
    assert x.cpu().tolist() == [7, 7]
    # a device value of another dtype is index_put_'s dtype error, not a fill
    with pytest.raises(
        RuntimeError, match="Index put requires the source and destination dtypes match"
    ):
        torch.zeros(2, device=mojo_gpu).index_put_(
            (m,), torch.tensor(5, device=mojo_gpu)
        )


def test_masked_fill_tensor_out_value_first_and_host_out_device(mojo_gpu: str):
    m = torch.ones(2, dtype=torch.bool, device=mojo_gpu)
    out = torch.empty(0, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="int8_t without overflow"):
        torch.ops.aten.masked_fill.Tensor_out(
            torch.zeros(2, dtype=torch.int8, device=mojo_gpu),
            m,
            torch.tensor(1000),
            out=out,
        )
    assert out.shape == (0,)
    host = torch.empty(0)
    with pytest.raises(RuntimeError, match="Expected out tensor to have device"):
        torch.ops.aten.masked_fill.Scalar_out(
            torch.zeros(2, device=mojo_gpu), m, 5.0, out=host
        )
    assert host.shape == (2,)


# --- integers meet floating components in one rounding; item assignment ---------


@pytest.mark.parametrize(
    "v", [2**53 + 1, 2**62 + 2**38 + 1, 2**63 - 1, -(2**62 + 2**38 + 1)]
)
def test_integers_round_once_into_complex64(mojo_gpu: str, v: int):
    want = torch.tensor(v).to(torch.complex64)  # int64 -> float, one rounding
    m = torch.ones(2, dtype=torch.bool, device=mojo_gpu)
    for src in (v, torch.tensor(v), torch.tensor(v, device=mojo_gpu)):
        got = torch.zeros(2, dtype=torch.complex64, device=mojo_gpu).masked_fill_(
            m, src
        )
        assert torch.equal(got.cpu(), want.expand(2))
        x = torch.full(
            (2,), complex(want.item()), dtype=torch.complex64, device=mojo_gpu
        )
        assert torch.eq(x, src).cpu().all()
        w = torch.where(m, src, torch.zeros(2, dtype=torch.complex64, device=mojo_gpu))
        assert torch.equal(w.cpu(), want.expand(2))
    y = torch.zeros(2, dtype=torch.complex64, device=mojo_gpu)
    y[m] = torch.tensor(v)
    assert torch.equal(y.cpu(), want.expand(2))


@pytest.mark.parametrize(
    ("dtype", "value", "assigns"),
    [
        (torch.float16, 70000.0, True),
        (torch.bfloat16, 1e39, True),
        (torch.int8, 1000, False),
        (torch.float32, 1e39, False),
    ],
)
def test_item_assignment_converts_like_cuda(
    mojo_gpu: str, dtype: torch.dtype, value: float, assigns: bool
):
    # Python item assignment builds its value with scalar_tensor, which
    # (like CUDA's host-side scalar_tensor_static) lets a half / bfloat16
    # overflow to inf; an explicit fill_ stays checked
    m = torch.ones(2, dtype=torch.bool, device=mojo_gpu)
    for key in (0, m, slice(None)):
        x = torch.zeros(2, dtype=dtype, device=mojo_gpu)
        if assigns:
            x[key] = value
            assert x.cpu().float()[0].item() == float("inf")
        else:
            with pytest.raises(RuntimeError, match="without overflow"):
                x[key] = value
    with pytest.raises(RuntimeError, match="without overflow"):
        torch.zeros(2, dtype=dtype, device=mojo_gpu).fill_(value)


def test_index_put_on_uint64_is_cudas_not_implemented(mojo_gpu: str):
    x = torch.zeros(2, dtype=torch.uint64, device=mojo_gpu)
    with pytest.raises(NotImplementedError, match="index_put"):
        x[torch.tensor([0], device=mojo_gpu)] = -1


# --- scalar_tensor and Python item assignment, as CUDA builds them --------------


_ITEM_VALUES = [
    True,
    3,
    -1,
    2.5,
    1e100,
    1 + 2j,
    1 + 0j,
    1j,
    2**63 - 1,
    1e19,
    70000.0,
    -(2**63),
]
_ITEM_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.uint64,
    torch.bool,
    torch.complex64,
    torch.complex128,
]


@pytest.mark.parametrize("dtype", _ITEM_DTYPES, ids=str)
def test_item_assignment_matches_the_host_conversion(mojo_gpu: str, dtype: torch.dtype):
    # CPU torch's item assignment uses the same scalar_tensor_static
    # conversion CUDA's does (on the host), so it is the oracle here
    if dtype in (torch.float64, torch.complex128):
        _skip_c128(mojo_gpu, torch.complex128)
    m = torch.tensor([True, False])
    for v in _ITEM_VALUES:
        for key in (0, m, slice(None)):
            ref = torch.zeros(2, dtype=dtype)
            errs = []
            try:
                ref[key] = v  # ty: ignore[invalid-assignment]
            except RuntimeError as e:  # NotImplementedError included
                errs.append(type(e))
            x = torch.zeros(2, dtype=dtype, device=mojo_gpu)
            dkey = key.to(mojo_gpu) if isinstance(key, torch.Tensor) else key
            if errs:
                with pytest.raises(errs[0]):
                    x[dkey] = v  # ty: ignore[invalid-assignment]
                assert torch.equal(x.cpu(), torch.zeros(2, dtype=dtype))
            else:
                x[dkey] = v  # ty: ignore[invalid-assignment]
                assert torch.equal(x.cpu(), ref), (v, key)


def test_scalar_tensor_defaults_to_the_float_dtype(mojo_gpu: str):
    assert torch.scalar_tensor(3, device=mojo_gpu).dtype == torch.get_default_dtype()
    assert torch.scalar_tensor(True, device=mojo_gpu).dtype == torch.get_default_dtype()
    with pytest.raises(RuntimeError, match="without overflow"):
        torch.scalar_tensor(1 + 2j, device=mojo_gpu)
    got = torch.scalar_tensor(1e19, dtype=torch.uint64, device=mojo_gpu)
    assert got.cpu().item() == 10**19
    assert (
        torch.zeros(2, dtype=torch.uint64, device=mojo_gpu).fill_(1e19).cpu().tolist()
        == [10**19] * 2
    )


_REAL_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.bool,
]


def _outcome(fn):
    try:
        return ("ok", fn().cpu())
    except Exception as e:  # noqa: BLE001 -- the exception type is the result
        return ("raised", type(e))


@pytest.mark.parametrize(
    "v", [1e19 + 0j, 2**63 + 0j, -1 + 0j, 300 + 0j, 2.5 + 0j, 2.0**64 + 0j, 2.0**64]
)
@pytest.mark.parametrize("dtype", _REAL_DTYPES)
def test_complex_scalar_into_a_real_dtype_matches_the_host(mojo_gpu: str, v, dtype):
    """A complex Scalar with no imaginary part meets a real destination as
    CUDA's c10 conversion has it, at every Scalar entry. The CPU is the
    oracle: on this whole matrix it agrees with stock CUDA, except an
    explicit out-of-range half scalar_tensor (inf on the CPU and here, an
    error on CUDA)."""

    def entries(dev):
        msk = torch.tensor([True, False], device=dev)

        def setitem(key):
            x = torch.zeros(2, dtype=dtype, device=dev)
            x[msk if key == "mask" else key] = v
            return x

        return {
            "scalar_tensor": lambda: torch.scalar_tensor(v, dtype=dtype, device=dev),
            "full": lambda: torch.full((2,), v, dtype=dtype, device=dev),
            "fill_": lambda: torch.zeros(2, dtype=dtype, device=dev).fill_(v),
            "masked_fill": lambda: torch.zeros(2, dtype=dtype, device=dev).masked_fill(
                msk, v
            ),
            "x[0]": lambda: setitem(0),
            "x[:]": lambda: setitem(slice(None)),
            "x[mask]": lambda: setitem("mask"),
        }

    want_all = entries("cpu")
    for name, fn in entries(mojo_gpu).items():
        got = _outcome(fn)
        want = _outcome(want_all[name])
        assert got[0] == want[0], (name, got, want)
        if got[0] == "ok":
            assert torch.equal(got[1], want[1]), (name, got, want)
        else:
            assert got[1] is want[1], (name, got, want)


def test_complex_scalar_into_uint64_is_exact(mojo_gpu: str):
    got = torch.scalar_tensor(1e19 + 0j, dtype=torch.uint64, device=mojo_gpu)
    assert got.cpu().tolist() == 10**19
    x = torch.zeros(2, dtype=torch.uint64, device=mojo_gpu)
    x[0] = 1e19 + 0j  # ty: ignore[invalid-assignment]
    assert x.cpu().tolist() == [10**19, 0]


def test_uint64_fill_of_two_to_the_64(mojo_gpu: str):
    """2.0**64 passes c10's check (it compares with double(UINT64_MAX)); the
    host static_cast then gives 0, as on CUDA."""
    for v in (2.0**64, 2.0**64 + 0j):
        assert (
            torch.scalar_tensor(v, dtype=torch.uint64, device=mojo_gpu).cpu().tolist()
            == 0
        )
        assert torch.zeros(2, dtype=torch.uint64, device=mojo_gpu).fill_(
            v
        ).cpu().tolist() == [0, 0]
    with pytest.raises(RuntimeError, match="uint64_t"):
        torch.zeros(1, dtype=torch.uint64, device=mojo_gpu).fill_(1.9e19)


def test_uint64_fill_from_a_same_device_float(mojo_gpu: str):
    """A same-device value is a copy_ (exact past 2**63)."""
    for vdt, want in ((torch.float64, 10**19), (torch.float32, 9999999980506447872)):
        val = torch.tensor(1e19, dtype=vdt, device=mojo_gpu)
        got = torch.zeros(2, dtype=torch.uint64, device=mojo_gpu).fill_(val)
        assert got.cpu().tolist() == [want] * 2


def test_int64_masked_and_index_fill_past_int64(mojo_gpu: str):
    msk = torch.tensor([True, False], device=mojo_gpu)
    for v in (2.0**63, 2**63 + 0j):
        x = torch.zeros(2, dtype=torch.int64, device=mojo_gpu)
        assert x.masked_fill(msk, v).cpu().tolist() == [-(2**63), 0]
    x = torch.zeros(2, dtype=torch.int64, device=mojo_gpu)
    x.index_fill_(0, torch.tensor([0], device=mojo_gpu), 2.0**63)
    assert x.cpu().tolist() == [-(2**63), 0]
    x = torch.zeros(2, dtype=torch.int64, device=mojo_gpu)
    assert x.masked_fill(msk, 2**62 + 1).cpu().tolist() == [2**62 + 1, 0]


@pytest.mark.parametrize("dtype", [torch.uint16, torch.uint32, torch.uint64])
def test_wide_unsigned_masked_and_index_fill_decline(mojo_gpu: str, dtype):
    """CUDA's masked_fill / index_fill / index_put kernels are not built for
    uint16/32/64: the dtype dispatch refuses before the value converts."""
    name = {torch.uint16: "UInt16", torch.uint32: "UInt32"}.get(dtype, "UInt64")
    msk = torch.tensor([True, False], device=mojo_gpu)
    idx = torch.tensor([0], device=mojo_gpu)
    for v in (3, 1e19, -1 + 0j, 1 + 2j):
        x = torch.zeros(2, dtype=dtype, device=mojo_gpu)
        with pytest.raises(NotImplementedError, match=f"masked_fill_.*{name}"):
            x.masked_fill(msk, v)
        with pytest.raises(NotImplementedError, match=f"masked_fill_.*{name}"):
            x.masked_fill_(msk, v)
        assert x.cpu().tolist() == [0, 0]
    x = torch.zeros(2, dtype=dtype, device=mojo_gpu)
    # (item assignment builds its value first: one that does not fit raises
    # the conversion error there, as on CUDA)
    with pytest.raises(NotImplementedError, match=f"masked_fill_.*{name}"):
        x[msk] = 3
    with pytest.raises(NotImplementedError, match=f"index_fill_cuda.*{name}"):
        x.index_fill_(0, idx, 1e19)
    with pytest.raises(NotImplementedError, match=f"index_put.*{name}"):
        x[torch.tensor([True, False])] = 3  # a CPU mask: index_put's kernel
    with pytest.raises(NotImplementedError, match=f"index_put.*{name}"):
        x[idx] = 3
    assert x.cpu().tolist() == [0, 0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.uint64])
def test_index_fill_refuses_a_complex_scalar(mojo_gpu: str, dtype):
    x = torch.zeros(2, dtype=dtype, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="Converting complex Scalar"):
        x.index_fill_(0, torch.tensor([0], device=mojo_gpu), 1 + 0j)
    assert x.cpu().tolist() == [0, 0]


def _part_overlapping_mask(x):
    """A bool mask over the first bytes of `x`, all true."""
    m = x.view(torch.bool)[: x.numel()]
    m.fill_(True)
    return m


@pytest.mark.parametrize("dtype", [torch.complex64, torch.float32])
def test_masked_fill_refuses_a_mask_overlapping_self(mojo_gpu: str, dtype):
    """masked_fill_'s assert_no_partial_overlap(self, mask): CUDA raises
    before writing (the complex route would otherwise change the mask with
    its real-part write)."""
    vals = [1 + 2j, 3 + 4j] if dtype.is_complex else [1.0, 2.0]
    fill = 5 + 6j if dtype.is_complex else 5.0
    for value in (fill, torch.tensor(fill, device=mojo_gpu)):
        x = torch.tensor(vals, dtype=dtype, device=mojo_gpu)
        m = _part_overlapping_mask(x)
        before = x.cpu().clone()
        with pytest.raises(RuntimeError, match="refer to a single memory"):
            x.masked_fill_(m, value)
        assert torch.equal(x.cpu(), before)
    # the functional form writes a new tensor: no overlap
    x = torch.tensor(vals, dtype=dtype, device=mojo_gpu)
    got = x.masked_fill(_part_overlapping_mask(x), fill)
    assert got.cpu().tolist() == [fill, fill]


@pytest.mark.parametrize("dtype", [torch.complex64, torch.float32])
def test_where_out_refuses_an_out_overlapping_the_condition(mojo_gpu: str, dtype):
    vals = [1 + 2j, 3 + 4j] if dtype.is_complex else [1.0, 2.0]
    x = torch.tensor(vals, dtype=dtype, device=mojo_gpu)
    c = _part_overlapping_mask(x)
    before = x.cpu().clone()
    with pytest.raises(RuntimeError, match="refer to a single memory"):
        torch.where(c, torch.full((2,), 5, dtype=dtype, device=mojo_gpu), x, out=x)
    assert torch.equal(x.cpu(), before)


def test_where_out_checks_overlap_only_on_unconverted_operands(mojo_gpu: str):
    """where_self_out converts an operand of another dtype into a new
    tensor before its TensorIterator checks overlap: such an operand may
    alias the out (stock CUDA values below)."""
    c = torch.tensor([True, False], device=mojo_gpu)
    x = torch.tensor([1.0, 2.0], device=mojo_gpu)
    torch.where(c, x.view(torch.bool)[:2], x, out=x)
    assert x.cpu().tolist() == [0.0, 2.0]
    z = torch.tensor([1 + 1j, 2 + 2j], dtype=torch.complex64, device=mojo_gpu)
    torch.where(c, z.view(torch.float32)[:2], z, out=z)
    assert z.cpu().tolist() == [1 + 0j, 2 + 2j]
    # an operand of the out's dtype is still checked
    for dt in (torch.float32, torch.complex64):
        y = torch.tensor([1, 2, 3], dtype=dt, device=mojo_gpu)
        with pytest.raises(RuntimeError, match="refer to a single memory"):
            torch.where(c, y[1:], torch.zeros(2, dtype=dt, device=mojo_gpu), out=y[:2])
        assert y.cpu().tolist() == [1, 2, 3]


@pytest.mark.parametrize("dtype", [torch.complex64, torch.float32, torch.int64])
@pytest.mark.parametrize(
    ("self_shape", "mask_shape"),
    [((), (1,)), ((2,), (2, 2)), ((2,), (3,)), ((0,), (3,)), ((2, 0), (3, 1))],
)
def test_masked_fill_inplace_mask_must_expand_to_self(
    mojo_gpu: str, dtype, self_shape, mask_shape
):
    """masked_fill_ expands the mask to self (expand_inplace): a mask that
    does not raises before anything is written, empty tensors included."""
    fill = 5 + 6j if dtype.is_complex else 5
    for value in (fill, torch.tensor(fill, device=mojo_gpu)):
        x = torch.ones(self_shape, dtype=dtype, device=mojo_gpu)
        m = torch.ones(mask_shape, dtype=torch.bool, device=mojo_gpu)
        with pytest.raises(RuntimeError, match="expand"):
            x.masked_fill_(m, value)
        assert torch.equal(x.cpu(), torch.ones(self_shape, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.complex64, torch.float32])
def test_masked_fill_inplace_mask_broadcasts_from_the_right(mojo_gpu: str, dtype):
    fill = 5 + 6j if dtype.is_complex else 5.0
    x = torch.ones(2, 3, dtype=dtype, device=mojo_gpu)
    x.masked_fill_(torch.tensor([True, False, True], device=mojo_gpu), fill)
    want = torch.ones(2, 3, dtype=dtype)
    want[:, [0, 2]] = fill  # ty: ignore[invalid-assignment]
    assert torch.equal(x.cpu(), want)
    y = torch.ones(2, dtype=dtype, device=mojo_gpu)
    y.masked_fill_(torch.tensor([True], device=mojo_gpu), fill)
    assert y.cpu().tolist() == [fill, fill]


def test_index_add_integer_alpha_rounds_once(mojo_gpu: str):
    got = torch.zeros(1, dtype=torch.complex64, device=mojo_gpu).index_add(
        0,
        torch.tensor([0], device=mojo_gpu),
        torch.ones(1, dtype=torch.complex64, device=mojo_gpu),
        alpha=2**62 + 2**38 + 1,
    )
    assert got.cpu()[0].real.item() == 2**62 + 2**39
