"""Native backend: the float8 dtypes (float8_e4m3fn, float8_e5m2, the two
fnuz variants and float8_e8m0fnu).

Stock CUDA's set is the oracle: allocation, conversions (c10's bit-exact
float8 <-> float32 functions; stock CUDA's sm_89+ e4m3fn saturation on
copies), fills, `.item()`, the dtype-agnostic data movement CUDA dispatches
over AT_FLOAT8_TYPES, eq / ne / where and their type promotion. Arithmetic
CUDA has no float8 kernel for is declined. References are CPU torch, with the
stock-CUDA saturation rule applied where it differs (`_stock_cast`).
"""

import itertools

import pytest
import torch

from tests.native.conftest import is_metal
from torch_mojo_backend.native import device_module

FP8 = [
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
]
FP8_IDS = [str(d).removeprefix("torch.") for d in FP8]
OTHERS = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float64,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
]

SPECIALS = [
    0.0, -0.0, 1.0, -1.0, 0.3, 0.75, 1.5, 3.0, 5.0, 6.0, -2.0, 255.0, 256.0,
    300.0, -300.0, 447.0, 448.0, 449.0, 464.0, 465.0, 480.0, 500.0, 1e4,
    57344.0, 61440.0, 65536.0, 1e6, -1e6, 2.0**-7 * 1.5, 2.0**-9, 2.0**-10,
    2.0**-16 * 1.5, 2.0**-17, 1e-10, 2.0**-127, 2.0**-128, 2.0**127, 3e38,
    float("inf"), float("-inf"), float("nan"),
]  # fmt: skip


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.element_size() == 1 else t


def _assert_same(got: torch.Tensor, want: torch.Tensor):
    """Bit-identical for float8 (NaN encodings included), NaN-aware equality
    for everything else."""
    assert got.dtype == want.dtype and got.shape == want.shape
    if got.dtype in FP8:
        assert torch.equal(_bits(got.cpu()), _bits(want)), (got, want)
    else:
        torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0, equal_nan=True)


def _all_codes(dtype: torch.dtype) -> torch.Tensor:
    return torch.arange(256, dtype=torch.int32).to(torch.uint8).view(dtype)


def _skip_f64(device: str, *dtypes: torch.dtype):
    if torch.float64 in dtypes and is_metal(device):
        pytest.skip("Apple GPUs have no float64")


# ---------------------------------------------------------------------------
# allocation, transfers, fills, item
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", FP8, ids=FP8_IDS)
def test_float8_roundtrip_and_capability(mojo_device, dtype):
    host = _all_codes(dtype).reshape(16, 16)
    dev = host.to(mojo_device)
    assert dev.dtype == dtype and dev.device.type == "mojo"
    _assert_same(dev, host)
    _assert_same(dev.t().contiguous(), host.t().contiguous())
    _assert_same(dev.clone(), host)
    empty = torch.empty(3, 5, dtype=dtype, device=mojo_device)
    assert empty.shape == (3, 5) and empty.dtype == dtype
    if hasattr(torch.accelerator, "get_device_capability"):
        index = torch.device(mojo_device).index
        caps = torch.accelerator.get_device_capability(index)["supported_dtypes"]
        assert dtype in caps


@pytest.mark.parametrize("dtype", FP8, ids=FP8_IDS)
def test_float8_factories_and_fills(mojo_device, dtype):
    for make in (torch.zeros, torch.ones):
        _assert_same(
            make(4, 7, dtype=dtype, device=mojo_device), make(4, 7, dtype=dtype)
        )
    for value in (0, 1, -1, 3, 0.75, 1e-30, 470.0, float("nan"), True):
        try:
            want = torch.empty(5, dtype=dtype).fill_(value)
        except RuntimeError as e:
            with pytest.raises(RuntimeError, match="without overflow"):
                torch.empty(5, dtype=dtype, device=mojo_device).fill_(value)
            assert "without overflow" in str(e)
            continue
        got = torch.empty(5, dtype=dtype, device=mojo_device).fill_(value)
        _assert_same(got, want)
        _assert_same(
            torch.full((2, 3), value, dtype=dtype, device=mojo_device),
            torch.full((2, 3), value, dtype=dtype),
        )
    # A strided fill and zero_.
    base = torch.zeros(4, 6, dtype=dtype, device=mojo_device)
    base[:, ::2].fill_(2)
    want = torch.zeros(4, 6, dtype=dtype)
    want[:, ::2].fill_(2)
    _assert_same(base, want)
    _assert_same(base.zero_(), torch.zeros(4, 6, dtype=dtype))


@pytest.mark.parametrize("dtype", FP8, ids=FP8_IDS)
def test_float8_item(mojo_device, dtype):
    codes = _all_codes(dtype)
    dev = codes.to(mojo_device)
    for i in range(0, 256, 7):
        got, want = dev[i].item(), codes[i].item()
        assert got == want or (got != got and want != want), (i, got, want)


# ---------------------------------------------------------------------------
# conversions on the device
# ---------------------------------------------------------------------------


def _sources(dtype: torch.dtype) -> torch.Tensor:
    vals = torch.tensor(SPECIALS, dtype=torch.float64)
    if dtype == torch.bool:
        return vals != 0
    if not dtype.is_floating_point:
        finite = vals[vals.isfinite() & (vals.abs() < 100)]
        return torch.cat([finite, torch.arange(-40, 40.0)]).to(dtype)
    return vals.to(dtype)


@pytest.mark.parametrize("dst", FP8, ids=FP8_IDS)
@pytest.mark.parametrize("src", OTHERS, ids=lambda d: str(d).removeprefix("torch."))
def test_to_float8(mojo_device, src, dst):
    _skip_f64(mojo_device, src)
    host = _sources(src)
    want = _stock_cast(host, dst, mojo_device)
    _assert_same(host.to(mojo_device).to(dst), want)
    # copy_ into an existing float8 tensor (non-contiguous destination too).
    out = torch.empty(2 * host.numel(), dtype=dst, device=mojo_device)[::2]
    out.copy_(host.to(mojo_device))
    _assert_same(out, want)


def _saturating_e4m3(device: str) -> bool:
    """Stock CUDA torch >= 2.12 (Copy.cu, #177870) converts float / half /
    bfloat16 -> e4m3fn with the saturating sm_89+ intrinsic; CPU, ROCm, MPS
    and older GPUs use c10's conversion, where an overflow is NaN."""
    props = device_module.get_device_properties(device)
    if props.api != "cuda" or props.major is None or props.minor is None:
        return False
    return (props.major, props.minor) >= (8, 9)


def _stock_cast(host: torch.Tensor, dst: torch.dtype, device: str) -> torch.Tensor:
    """What stock torch on this device's hardware class gives for
    `host.to(dst)` (CPU torch, plus the sm_89+ e4m3fn saturation)."""
    floats = (torch.float32, torch.float16, torch.bfloat16)
    if dst == torch.float8_e4m3fn and host.dtype in floats and _saturating_e4m3(device):
        return host.float().clamp(-448.0, 448.0).to(dst)
    return host.to(dst)


@pytest.mark.parametrize("src", FP8, ids=FP8_IDS)
@pytest.mark.parametrize(
    "dst", OTHERS + FP8, ids=lambda d: str(d).removeprefix("torch.")
)
def test_from_float8(mojo_device, src, dst):
    _skip_f64(mojo_device, dst)
    host = _all_codes(src)
    want = host.to(dst)
    if not dst.is_floating_point and dst != torch.bool:
        # NaN / inf / out-of-range float -> int is implementation-defined
        # (CPU and CUDA disagree); compare the values both define.
        f = host.double()
        info = torch.iinfo(dst)
        keep = f.isfinite() & (f > info.min - 1.0) & (f < info.max + 1.0)
        if dst in (torch.uint8, torch.int8, torch.int16):
            keep &= f.abs() < 2.0**31
        host, want = host[keep], want[keep]
    _assert_same(host.to(mojo_device).to(dst), want)


# ---------------------------------------------------------------------------
# data movement and comparisons (what CUDA dispatches over float8)
# ---------------------------------------------------------------------------


def _x(dtype: torch.dtype) -> torch.Tensor:
    return (torch.arange(24, dtype=torch.float32).reshape(4, 6) / 4 + 1).to(dtype)


def _ref(fn, x: torch.Tensor) -> torch.Tensor:
    """`fn` on CPU over the raw bytes: data movement does not look at values."""
    out = fn(x.view(torch.uint8))
    return out.view(x.dtype) if out.dtype == torch.uint8 else out


MOVES = {
    "clone": lambda x: x.clone(),
    "transpose_contiguous": lambda x: x.t().contiguous(),
    "cat": lambda x: torch.cat([x, x[1:]]),
    "stack": lambda x: torch.stack([x, x], dim=1),
    "index_select": lambda x: x.index_select(
        1, torch.tensor([5, 0, 2], device=x.device)
    ),
    "index": lambda x: x[torch.tensor([3, 0], device=x.device)],
    "gather": lambda x: x.gather(
        1, torch.tensor([[1, 0], [5, 5], [2, 3], [0, 4]], device=x.device)
    ),
    "flip": lambda x: x.flip(0, 1),
    "repeat": lambda x: x.repeat(2, 3),
    "narrow_copy": lambda x: x.narrow_copy(1, 2, 3),
    "split": lambda x: x.split(4, 1)[1].contiguous(),
    "diag": lambda x: x[:, :4].diag(),
    "constant_pad_nd": lambda x: torch.constant_pad_nd(x, (1, 2)),
    "unfold": lambda x: x.unfold(1, 2, 1).contiguous(),
    "permute_reshape": lambda x: x.reshape(2, 2, 6).permute(2, 0, 1).reshape(6, 4),
}


@pytest.mark.parametrize("dtype", FP8, ids=FP8_IDS)
@pytest.mark.parametrize("name", list(MOVES))
def test_float8_data_movement(mojo_device, dtype, name):
    x = _x(dtype)
    _assert_same(MOVES[name](x.to(mojo_device)), _ref(MOVES[name], x))


@pytest.mark.parametrize("dtype", FP8, ids=FP8_IDS)
def test_float8_index_put_scatter_masked(mojo_device, dtype):
    x = _x(dtype)
    dev = x.to(mojo_device)
    idx = torch.tensor([2, 0])
    vals = _ref(lambda u: u[:2].flip(1), _x(dtype))
    got = dev.clone().index_put_((idx.to(mojo_device),), vals.to(mojo_device))
    want = _ref(lambda u: u.clone().index_put_((idx,), vals.view(torch.uint8)), x)
    _assert_same(got, want)
    sidx = torch.tensor([[3, 1, 0, 2, 1, 0]])
    got = dev.clone().scatter_(0, sidx.to(mojo_device), dev[:1])
    want = _ref(lambda u: u.clone().scatter_(0, sidx, u[:1]), x)
    _assert_same(got, want)
    mask = torch.arange(24).reshape(4, 6) % 3 == 0
    _assert_same(
        dev.masked_select(mask.to(mojo_device)),
        _ref(lambda u: u.masked_select(mask), x),
    )
    other = torch.zeros_like(x)
    got = torch.where(mask.to(mojo_device), dev, other.to(mojo_device))
    want = _ref(lambda u: torch.where(mask, u, other.view(torch.uint8)), x)
    _assert_same(got, want)


@pytest.mark.parametrize("dtype", FP8, ids=FP8_IDS)
def test_float8_eq_ne_isnan_equal(mojo_device, dtype):
    a = _all_codes(dtype)
    b = a.roll(1)
    da, db = a.to(mojo_device), b.to(mojo_device)
    # CUDA compares the float values (NaN != NaN, +0 == -0).
    fa, fb = a.float(), b.float()
    _assert_same(da == db, fa == fb)
    _assert_same(da != db, fa != fb)
    _assert_same(da == da, fa == fa)
    _assert_same(da.isnan(), fa.isnan())
    finite = a[a.float().isfinite()].to(mojo_device)
    assert torch.equal(finite, finite.clone())
    assert not torch.equal(da, db)


@pytest.mark.parametrize("dtype", FP8, ids=FP8_IDS)
def test_float8_views_and_misc(mojo_device, dtype):
    x = _x(dtype)
    dev = x.to(mojo_device)
    assert torch.equal(dev.view(torch.uint8).cpu(), x.view(torch.uint8))
    _assert_same(torch.empty_like(dev).copy_(dev), x)
    _assert_same(
        torch.eye(3, dtype=dtype, device=mojo_device), torch.eye(3, dtype=dtype)
    )
    if dtype != torch.float8_e8m0fnu:  # CUDA's _foreach_copy_ declines e8m0
        dst = [torch.zeros_like(dev)]
        torch._foreach_copy_(dst, [dev])
        _assert_same(dst[0], x)


@pytest.mark.parametrize("dtype", FP8, ids=FP8_IDS)
@pytest.mark.parametrize(
    "op",
    [
        lambda x: x + x,
        lambda x: x * x,
        lambda x: x.sum(),
        lambda x: x.abs(),
        lambda x: x < x,
        lambda x: x.sort(),
        lambda x: x @ x.t(),
    ],
    ids=["add", "mul", "sum", "abs", "lt", "sort", "mm"],
)
def test_float8_arithmetic_declined(mojo_device, dtype, op):
    """CUDA has no float8 arithmetic kernels: neither do we."""
    with pytest.raises(NotImplementedError):
        op(_x(dtype).to(mojo_device))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_nan_reads_as_true_and_unequal(mojo_device, dtype):
    """bool(nan) is True and nan != x is True, as on CUDA (the float8 paths
    lean on both; the fast-math build had folded them to False)."""
    a = torch.tensor([0.0, float("nan"), 1.0, -0.0, float("nan")], dtype=dtype)
    b = torch.tensor([float("nan"), float("nan"), 1.0, 0.0, 2.0], dtype=dtype)
    da, db = a.to(mojo_device), b.to(mojo_device)
    _assert_same(da.to(torch.bool), a.to(torch.bool))
    _assert_same(da != db, a != b)
    _assert_same(da == db, a == b)


# ---------------------------------------------------------------------------
# review regressions
# ---------------------------------------------------------------------------


def test_float8_compare_with_scalars_and_0d(mojo_device):
    """A Python scalar or a 0-d tensor rounds into the float8 dtype (CUDA
    compares `x == 1000.` against NaN instead of raising); a dimensioned
    tensor of any other dtype is c10's float8 promotion error."""
    x = torch.tensor([1.0, 1.06, 500.0, -2.0]).to(torch.float8_e4m3fn)
    d = x.to(mojo_device)
    for other in (1.0, 1, 1.0625, 1000.0, float("nan"), True):
        _assert_same(d == other, x == other)
        _assert_same(d != other, x != other)
    for other in (
        torch.tensor(1.0),
        torch.tensor(1),
        torch.tensor(True),
        torch.tensor(1.0).to(torch.float8_e5m2),
    ):
        _assert_same(d == other.to(mojo_device), x == other)
    for other in (
        torch.ones(4),
        torch.ones(4, dtype=torch.bool),
        x.to(torch.float8_e5m2),
    ):
        with pytest.raises(RuntimeError, match="Promotion for Float8 Types"):
            d == other.to(mojo_device)  # noqa: B015


def test_float8_scatter_value_declined(mojo_device):
    """CUDA's scatter fill kernel has no float8 case (scatter.src has)."""
    x = _x(torch.float8_e4m3fn).to(mojo_device)
    idx = torch.zeros(1, 6, dtype=torch.long, device=mojo_device)
    for value in (2.0, 500.0):
        with pytest.raises(NotImplementedError, match="scatter_fill"):
            x.clone().scatter_(0, idx, value)
        with pytest.raises(NotImplementedError, match="scatter_fill"):
            x.scatter(0, idx, value)


def test_e4m3fn_saturation_matches_stock_cuda(mojo_device):
    """float32 [500, inf, -1e9, nan] -> e4m3fn: [448, 448, -448, nan] on an
    sm_89+ NVIDIA GPU (stock torch 2.14), c10's [nan, nan, nan, nan]
    elsewhere; a float64 source is never saturated."""
    x = torch.tensor([500.0, float("inf"), -1e9, float("nan"), 3.0])
    got = x.to(mojo_device).to(torch.float8_e4m3fn).cpu().float()
    if _saturating_e4m3(mojo_device):
        want = torch.tensor([448.0, 448.0, -448.0, float("nan"), 3.0])
    else:
        want = x.to(torch.float8_e4m3fn).float()
    torch.testing.assert_close(got, want, equal_nan=True)
    if not is_metal(mojo_device):
        d64 = x.double().to(mojo_device).to(torch.float8_e4m3fn).cpu()
        _assert_same(d64, x.double().to(torch.float8_e4m3fn))


def test_float8_compare_with_cpu_0d_tensor(mojo_device):
    """A 0-d CPU tensor is a scalar operand (CUDA takes it)."""
    x = torch.tensor([1.0, 2.0, 500.0]).to(torch.float8_e4m3fn)
    d = x.to(mojo_device)
    for other in (torch.tensor(1.0), torch.tensor(2), torch.tensor(True)):
        _assert_same(d == other, x == other)
        _assert_same(d != other, x != other)
        _assert_same(other == d, other == x)
    f = torch.tensor([1.0, 2.0]).to(mojo_device)
    _assert_same(f == torch.tensor(1.0), torch.tensor([True, False]))


def test_float8_compare_complex_scalar(mojo_device):
    d = _x(torch.float8_e4m3fn).to(mojo_device)
    with pytest.raises(RuntimeError, match="Unknown Complex ScalarType"):
        d == 1 + 1j  # noqa: B015


def test_float8_zero_dim_vs_dimensioned_float(mojo_device):
    """A 0-d float8 meets a dimensioned half / float in that float's dtype."""
    z = torch.tensor(1.5).to(torch.float8_e4m3fn)
    for dt in (torch.float16, torch.float32, torch.bfloat16):
        o = torch.tensor([1.5, 2.0], dtype=dt)
        _assert_same(z.to(mojo_device) == o.to(mojo_device), z == o)
        _assert_same(o.to(mojo_device) != z.to(mojo_device), o != z)


@pytest.mark.parametrize("dtype", FP8, ids=FP8_IDS)
def test_float8_masked_fill_declined_where_works(mojo_device, dtype):
    """CUDA has no float8 masked_fill kernel (where.self has float8)."""
    x = _x(dtype).to(mojo_device)
    mask = (torch.arange(24).reshape(4, 6) % 2 == 0).to(mojo_device)
    with pytest.raises(NotImplementedError, match="masked_fill_"):
        x.masked_fill(mask, 2.0)
    with pytest.raises(NotImplementedError, match="masked_fill_"):
        x.masked_fill(mask, torch.tensor(2.0, device=mojo_device))
    with pytest.raises(NotImplementedError, match="masked_fill_"):
        x.clone().masked_fill_(mask, 2.0)
    got = torch.where(mask, x, x.flip(0))
    want = _ref(lambda u: torch.where(mask.cpu(), u, u.flip(0)), _x(dtype))
    _assert_same(got, want)


def test_float8_scatter_reduce_names_cuda_kernel(mojo_device):
    x = _x(torch.float8_e4m3fn).to(mojo_device)
    idx = torch.zeros(1, 6, dtype=torch.long, device=mojo_device)
    with pytest.raises(NotImplementedError, match="scatter_gather_base"):
        x.clone().scatter_add_(0, idx, x[:1])
    with pytest.raises(NotImplementedError, match="scatter_gather_base"):
        x.clone().scatter_reduce_(0, idx, x[:1], "amax")


# ---------------------------------------------------------------------------
# review round 3
# ---------------------------------------------------------------------------


def test_compare_with_cpu_0d_tensor_promotes_like_a_tensor(mojo_device):
    """A 0-d CPU operand takes part in result_type as a 0-d tensor, on
    either side of every comparison (values measured on stock CUDA)."""
    dev = mojo_device
    h = torch.tensor([1.0], dtype=torch.float16)
    # Same category: the dimensioned half wins, 1.0001 rounds to 1.
    _assert_same(h.to(dev) == torch.tensor(1.0001), torch.tensor([True]))
    _assert_same(h.to(dev) == torch.tensor(1.0001, device=dev), torch.tensor([True]))
    i = torch.tensor([1, 2], dtype=torch.int32)
    _assert_same(i.to(dev) < torch.tensor(1.5), torch.tensor([True, False]))
    f = torch.tensor([1.0, 2.0])
    _assert_same(torch.tensor(1.5) >= f.to(dev), torch.tensor([True, False]))
    _assert_same(torch.tensor(1.5) < f.to(dev), torch.tensor([False, True]))
    z8 = torch.tensor(1.0).to(torch.float8_e4m3fn).to(dev)
    with pytest.raises(RuntimeError, match="Promotion for Float8 Types"):
        z8 == torch.tensor(1.0)  # noqa: B015
    x8 = torch.tensor([1.0, 2.0]).to(torch.float8_e4m3fn).to(dev)
    with pytest.raises(NotImplementedError, match="compare_cuda"):
        torch.tensor(1.5) < x8  # noqa: B015


def test_float8_promotion_never_saturates(mojo_device):
    """Saturation is copy_ / to()'s; a comparison or where rounds the other
    operand like c10 (500 -> NaN), even where copy_ saturates."""
    dev = mojo_device
    x = torch.tensor([448.0, 1.0]).to(torch.float8_e4m3fn).to(dev)
    for other in (torch.tensor(500.0), torch.tensor(500.0, device=dev), 500.0):
        _assert_same(x == other, torch.tensor([False, False]))
        _assert_same(x != other, torch.tensor([True, True]))
    c = torch.tensor([False, True], device=dev)
    with pytest.raises(RuntimeError, match="Promotion for Float8 Types"):
        torch.where(c, x, torch.ones(2, device=dev))


def test_where_converts_like_to(mojo_device):
    """where_self_out converts `other` with `.to(result_type)`: on the device
    that is a copy (stock CUDA's sm_89+ e4m3fn saturation applies), a 0-d
    CPU other converts on the CPU (c10: 500 -> NaN)."""
    dev = mojo_device
    x = torch.tensor([448.0, 1.0]).to(torch.float8_e4m3fn).to(dev)
    c = torch.tensor([False, True], device=dev)
    got = torch.where(c, x, torch.tensor(500.0, device=dev)).cpu().float()
    want0 = _stock_cast(torch.tensor([500.0]), torch.float8_e4m3fn, dev).float()
    torch.testing.assert_close(got[:1], want0, equal_nan=True)
    assert got[1] == 1.0
    got = torch.where(c, x, torch.tensor(500.0)).cpu().float()
    assert torch.isnan(got[0]) and got[1] == 1.0


@pytest.mark.parametrize("non_blocking", [False, True])
def test_float8_cross_device_casts(mojo_device, non_blocking):
    """`.to(device, dtype)` and `copy_` across host and device with a float8
    end: a blocking transfer converts on the CPU (c10: 500 -> NaN), a
    non-blocking one on the GPU (stock CUDA's rule, see _stock_cast)."""
    dev = mojo_device
    f8 = torch.float8_e4m3fn
    for src in (
        torch.tensor([3.0, 500.0, -0.25]),
        torch.tensor([3.0, 500.0, -0.25], dtype=torch.bfloat16),
    ):
        want = _stock_cast(src, f8, dev) if non_blocking else src.to(f8)
        got = src.to(dev, dtype=f8, non_blocking=non_blocking)
        torch.accelerator.synchronize()
        _assert_same(got, want)
        dst = torch.empty(3, dtype=f8, device=dev)
        dst.copy_(src, non_blocking=non_blocking)
        torch.accelerator.synchronize()
        _assert_same(dst, want)
    cpu8 = torch.tensor([3.0, 448.0]).to(f8)
    got = cpu8.to(dev, dtype=torch.float32, non_blocking=non_blocking)
    torch.accelerator.synchronize()
    _assert_same(got, cpu8.float())
    on_dev = torch.tensor([3.0, 500.0], device=dev)
    want = _stock_cast(on_dev.cpu(), f8, dev) if non_blocking else on_dev.cpu().to(f8)
    host = torch.empty(2, dtype=f8)
    host.copy_(on_dev, non_blocking=non_blocking)
    torch.accelerator.synchronize()
    _assert_same(host, want)
    if not non_blocking:
        _assert_same(on_dev.to("cpu", dtype=f8), want)


def test_float8_index_reduce_and_nonzero_declined(mojo_device):
    x = _x(torch.float8_e4m3fn).to(mojo_device)
    idx = torch.tensor([0], device=mojo_device)
    with pytest.raises(NotImplementedError, match='"index_reduce"'):
        x.clone().index_reduce_(0, idx, x[:1], "amax")
    with pytest.raises(NotImplementedError, match="exclude_input_init"):
        x.clone().index_reduce_(0, idx, x[:1], "amax", include_self=False)
    with pytest.raises(NotImplementedError, match="nonzero_cuda"):
        x.nonzero()


# ---------------------------------------------------------------------------
# review round 4
# ---------------------------------------------------------------------------


def test_float8_scatter_empty_index_is_a_copy(mojo_device):
    """An empty index never reaches CUDA's float8 dtype dispatch."""
    dev = mojo_device
    x = _x(torch.float8_e4m3fn).to(dev)
    e = torch.zeros(0, 6, dtype=torch.long, device=dev)
    for fn in (
        lambda t: t.scatter_add_(0, e, t[:0]),
        lambda t: t.scatter_reduce_(0, e, t[:0], "amax"),
        lambda t: t.scatter_(0, e, 2.0, reduce="add"),
        lambda t: t.scatter_(0, e, 2.0),
    ):
        _assert_same(fn(x.clone()), _x(torch.float8_e4m3fn))


def test_compare_where_cpu_0d_converts_on_cpu(mojo_device):
    """A 0-d CPU operand converts to the result dtype on the CPU (no
    float64 kernel needed on Metal), and where follows the 0-d rule."""
    dev = mojo_device
    h = torch.tensor([1.0, 2.0], dtype=torch.float16)
    _assert_same(
        h.to(dev) == torch.tensor(1.0, dtype=torch.float64), torch.tensor([True, False])
    )
    c = torch.tensor([True, False])
    got = torch.where(c.to(dev), h.to(dev), torch.tensor(5.0, dtype=torch.float64))
    _assert_same(got, torch.where(c, h, torch.tensor(5.0, dtype=torch.float64)))
    f = torch.tensor([1.5, 2.5])
    got = torch.where(c.to(dev), f.to(dev), torch.tensor(3))
    _assert_same(got, torch.where(c, f, torch.tensor(3)))


def test_float8_promotion_error_names_dimensioned_first(mojo_device):
    dev = mojo_device
    z8 = torch.tensor(1.0).to(torch.float8_e4m3fn).to(dev)
    c = torch.tensor([True, False], device=dev)
    with pytest.raises(RuntimeError, match="promote Int and Float8_e4m3fn"):
        torch.where(c, z8, torch.tensor([1, 2], dtype=torch.int32, device=dev))


# ---------------------------------------------------------------------------
# PR review
# ---------------------------------------------------------------------------


def test_compare_out_alias_checked_on_original_operands(mojo_device):
    """eq(a, b, out=a) with a half `a` and a float `b`: the overlap / resize
    rules see the caller's `a`, not a cast copy (stock CUDA raises)."""
    dev = mojo_device
    a = torch.tensor([1.0], dtype=torch.half, device=dev)
    b = torch.tensor([1.0, 2.0], device=dev)
    with pytest.raises(RuntimeError):
        torch.eq(a, b, out=a)
    assert a.cpu().tolist() == [1.0]


@pytest.mark.parametrize("non_blocking", [False, True])
def test_float8_download_to_int_converts_where_cuda_does(mojo_device, non_blocking):
    """Measured on stock CUDA: a blocking download with a dtype change
    converts on the CPU (e5m2 inf / nan -> int32 is INT_MIN), a
    non-blocking one on the GPU (INT_MAX / 0)."""
    x = torch.tensor([float("inf"), float("nan"), 512.0, -3.0]).to(torch.float8_e5m2)
    got = x.to(mojo_device).to("cpu", dtype=torch.int32, non_blocking=non_blocking)
    torch.accelerator.synchronize()
    imin, imax = -(2**31), 2**31 - 1
    want = [imax, 0, 512, -3] if non_blocking else [imin, imin, 512, -3]
    assert got.tolist() == want


def test_float8_fill_tensor_same_device_is_a_copy(mojo_device):
    """fill_(device tensor) is copy_ (no range check, copy's conversion);
    fill_(CPU tensor) is fill_(Scalar), which range-checks."""
    dev = mojo_device
    for dt, v in (
        (torch.float8_e4m3fn, 1000.0),
        (torch.float8_e5m2, 1e6),
        (torch.float8_e8m0fnu, 0.0),
    ):
        got = torch.zeros(3, dtype=dt, device=dev).fill_(torch.tensor(v, device=dev))
        _assert_same(got, _stock_cast(torch.full((3,), v), dt, dev))
    with pytest.raises(RuntimeError, match="without overflow"):
        torch.zeros(3, dtype=torch.float8_e4m3fn, device=dev).fill_(
            torch.tensor(1000.0)
        )


def test_float8_mask_assign_device_value(mojo_device):
    """t[mask] = device value takes CUDA's index_put kernel (float8 ok), not
    masked_fill (CPU values only, refused for float8)."""
    dev = mojo_device
    t = torch.zeros(3).to(torch.float8_e4m3fn).to(dev)
    mask = torch.tensor([True, False, True], device=dev)
    t[mask] = torch.tensor(2.0).to(torch.float8_e4m3fn).to(dev)
    assert t.view(torch.uint8).cpu().tolist() == [64, 0, 64]


@pytest.mark.parametrize(
    "op,kernel",
    [
        (lambda x: x.roll(1, 0), "roll_cuda"),
        (lambda x: x.take(torch.tensor([0], device=x.device)), "take_cuda"),
        (lambda x: x.tril(), "triu_tril_cuda_template"),
        (lambda x: x.triu(), "triu_tril_cuda_template"),
        (
            lambda x: x.clone().masked_scatter_(
                torch.ones_like(x, dtype=torch.bool), x
            ),
            "masked_scatter_",
        ),
        (lambda x: x + x, "ufunc_add_CUDA"),
        (lambda x: x - x, "ufunc_add_CUDA"),
        (lambda x: x * x, "mul_cuda"),
    ],
    ids=["roll", "take", "tril", "triu", "masked_scatter", "add", "sub", "mul"],
)
def test_float8_refusals_name_cuda_kernels(mojo_device, op, kernel):
    x = _x(torch.float8_e4m3fn).to(mojo_device)
    with pytest.raises(NotImplementedError, match=kernel):
        op(x)


# Measured on stock CUDA (torch 2.11 / H100): same-device float8 -> integer
# conversions of _CONV_SOURCES, which go through float32 and PTX's
# saturating cvt (NaN is 0 for int32 / uint32, INT64_MIN for int64 and the
# types narrowed from it, 2**63 for uint64).
_CONV_SOURCES = [
    float("inf"), float("-inf"), float("nan"), -1.0, -3.5, 300.0, 70000.0,
    1e9, 3.5, 0.0, -0.0, 57344.0,
]  # fmt: skip
_CUDA_FLOAT8_TO_INT = {
    ("float8_e4m3fn", "uint8"): [0, 0, 0, 255, 253, 32, 0, 0, 3, 0, 0, 0],
    ("float8_e4m3fn", "uint16"): [0, 0, 0, 0, 0, 288, 0, 0, 3, 0, 0, 0],
    ("float8_e4m3fn", "uint32"): [0, 0, 0, 0, 0, 288, 0, 0, 3, 0, 0, 0],
    ("float8_e4m3fn", "uint64"): [9223372036854775808, 9223372036854775808, 9223372036854775808, 0, 0, 288, 9223372036854775808, 9223372036854775808, 3, 0, 0, 9223372036854775808],
    ("float8_e4m3fn", "int8"): [0, 0, 0, -1, -3, 32, 0, 0, 3, 0, 0, 0],
    ("float8_e4m3fn", "int16"): [0, 0, 0, -1, -3, 288, 0, 0, 3, 0, 0, 0],
    ("float8_e4m3fn", "int32"): [0, 0, 0, -1, -3, 288, 0, 0, 3, 0, 0, 0],
    ("float8_e4m3fn", "int64"): [-9223372036854775808, -9223372036854775808, -9223372036854775808, -1, -3, 288, -9223372036854775808, -9223372036854775808, 3, 0, 0, -9223372036854775808],
    ("float8_e5m2", "uint8"): [255, 0, 0, 255, 253, 64, 255, 255, 3, 0, 0, 0],
    ("float8_e5m2", "uint16"): [65535, 0, 0, 0, 0, 320, 65535, 65535, 3, 0, 0, 57344],
    ("float8_e5m2", "uint32"): [4294967295, 0, 0, 0, 0, 320, 4294967295, 4294967295, 3, 0, 0, 57344],
    ("float8_e5m2", "uint64"): [18446744073709551615, 0, 9223372036854775808, 0, 0, 320, 18446744073709551615, 18446744073709551615, 3, 0, 0, 57344],
    ("float8_e5m2", "int8"): [-1, 0, 0, -1, -3, 64, -1, -1, 3, 0, 0, 0],
    ("float8_e5m2", "int16"): [-1, 0, 0, -1, -3, 320, -1, -1, 3, 0, 0, -8192],
    ("float8_e5m2", "int32"): [2147483647, -2147483648, 0, -1, -3, 320, 2147483647, 2147483647, 3, 0, 0, 57344],
    ("float8_e5m2", "int64"): [9223372036854775807, -9223372036854775808, -9223372036854775808, -1, -3, 320, 9223372036854775807, 9223372036854775807, 3, 0, 0, 57344],
    ("float8_e4m3fnuz", "uint8"): [0, 0, 0, 255, 253, 0, 0, 0, 3, 0, 0, 0],
    ("float8_e4m3fnuz", "uint16"): [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
    ("float8_e4m3fnuz", "uint32"): [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
    ("float8_e4m3fnuz", "uint64"): [9223372036854775808, 9223372036854775808, 9223372036854775808, 0, 0, 9223372036854775808, 9223372036854775808, 9223372036854775808, 3, 0, 0, 9223372036854775808],
    ("float8_e4m3fnuz", "int8"): [0, 0, 0, -1, -3, 0, 0, 0, 3, 0, 0, 0],
    ("float8_e4m3fnuz", "int16"): [0, 0, 0, -1, -3, 0, 0, 0, 3, 0, 0, 0],
    ("float8_e4m3fnuz", "int32"): [0, 0, 0, -1, -3, 0, 0, 0, 3, 0, 0, 0],
    ("float8_e4m3fnuz", "int64"): [-9223372036854775808, -9223372036854775808, -9223372036854775808, -1, -3, -9223372036854775808, -9223372036854775808, -9223372036854775808, 3, 0, 0, -9223372036854775808],
    ("float8_e5m2fnuz", "uint8"): [0, 0, 0, 255, 253, 64, 0, 0, 3, 0, 0, 0],
    ("float8_e5m2fnuz", "uint16"): [0, 0, 0, 0, 0, 320, 0, 0, 3, 0, 0, 57344],
    ("float8_e5m2fnuz", "uint32"): [0, 0, 0, 0, 0, 320, 0, 0, 3, 0, 0, 57344],
    ("float8_e5m2fnuz", "uint64"): [9223372036854775808, 9223372036854775808, 9223372036854775808, 0, 0, 320, 9223372036854775808, 9223372036854775808, 3, 0, 0, 57344],
    ("float8_e5m2fnuz", "int8"): [0, 0, 0, -1, -3, 64, 0, 0, 3, 0, 0, 0],
    ("float8_e5m2fnuz", "int16"): [0, 0, 0, -1, -3, 320, 0, 0, 3, 0, 0, -8192],
    ("float8_e5m2fnuz", "int32"): [0, 0, 0, -1, -3, 320, 0, 0, 3, 0, 0, 57344],
    ("float8_e5m2fnuz", "int64"): [-9223372036854775808, -9223372036854775808, -9223372036854775808, -1, -3, 320, -9223372036854775808, -9223372036854775808, 3, 0, 0, 57344],
    ("float8_e8m0fnu", "uint8"): [0, 0, 0, 1, 4, 0, 0, 0, 4, 0, 0, 0],
    ("float8_e8m0fnu", "uint16"): [0, 0, 0, 1, 4, 256, 0, 0, 4, 0, 0, 0],
    ("float8_e8m0fnu", "uint32"): [0, 0, 0, 1, 4, 256, 65536, 1073741824, 4, 0, 0, 65536],
    ("float8_e8m0fnu", "uint64"): [9223372036854775808, 9223372036854775808, 9223372036854775808, 1, 4, 256, 65536, 1073741824, 4, 0, 0, 65536],
    ("float8_e8m0fnu", "int8"): [0, 0, 0, 1, 4, 0, 0, 0, 4, 0, 0, 0],
    ("float8_e8m0fnu", "int16"): [0, 0, 0, 1, 4, 256, 0, 0, 4, 0, 0, 0],
    ("float8_e8m0fnu", "int32"): [0, 0, 0, 1, 4, 256, 65536, 1073741824, 4, 0, 0, 65536],
    ("float8_e8m0fnu", "int64"): [-9223372036854775808, -9223372036854775808, -9223372036854775808, 1, 4, 256, 65536, 1073741824, 4, 0, 0, 65536],
}  # fmt: skip


@pytest.mark.parametrize("key", list(_CUDA_FLOAT8_TO_INT), ids=lambda k: "-".join(k))
def test_float8_to_int_on_device_matches_stock_cuda(mojo_device, key):
    f8 = getattr(torch, key[0])
    dt = getattr(torch, key[1])
    x = torch.tensor(_CONV_SOURCES).to(f8).to(mojo_device)
    assert x.to(dt).cpu().tolist() == _CUDA_FLOAT8_TO_INT[key]


def test_mask_assign_device_value_checks(mojo_device):
    """index_put_'s checks before any route: a device value must match
    self's dtype and not overlap self (stock CUDA, any dtype)."""
    dev = mojo_device
    mask = torch.tensor([True, False, True], device=dev)
    for dt in (torch.float8_e4m3fn, torch.float32):
        t = torch.zeros(3).to(dt).to(dev)
        with pytest.raises(RuntimeError, match="dtypes match"):
            t[mask] = torch.tensor(2.0, dtype=torch.float64, device=dev)
        with pytest.raises(RuntimeError, match="single memory location"):
            t[mask] = t[0]
    t = torch.zeros(3, device=dev)
    t[mask] = torch.tensor(2.0, dtype=torch.float64)  # a CPU value converts
    assert t.cpu().tolist() == [2.0, 0.0, 2.0]


def test_float8_refusals_wait_for_empty_inputs(mojo_device):
    """CUDA returns empty results before its float8 dispatch for these."""
    dev = mojo_device
    f8 = torch.float8_e4m3fn
    e = torch.empty((0, 2), dtype=f8, device=dev)
    assert e.roll(1, 0).shape == (0, 2)
    assert e.tril().shape == (0, 2) and e.triu().shape == (0, 2)
    assert e.clone().tril_().shape == (0, 2)
    x = torch.ones(2, 2).to(f8).to(dev)
    none = torch.zeros(0, dtype=torch.long, device=dev)
    assert x.take(none).shape == (0,)
    m = torch.zeros(0, 2, dtype=torch.bool, device=dev)
    assert e.clone().masked_scatter_(m, e).shape == (0, 2)
    # ... while these refuse even when empty, as on CUDA.
    for op in (
        lambda t: t + t,
        lambda t: t * t,
        lambda t: t.nonzero(),
        lambda t: t < t,
    ):
        with pytest.raises(NotImplementedError):
            op(e)


def _cuda_f2i(f: float, dt: str) -> int:
    """Stock CUDA's float -> integer conversion, measured on every float8 code
    (torch 2.11 / H100): PTX's saturating truncation to int32 (NaN 0) narrowed
    by truncation for int8 / int16; to uint32 (NaN 0) narrowed for uint16; to
    int64 (NaN INT64_MIN) narrowed for uint8; uint64 saturates (NaN 2**63)."""

    def sat(lo: int, hi: int) -> int:
        return lo if f <= lo else hi if f >= hi else int(f)

    nan = f != f
    if dt in ("int32", "int16", "int8"):
        bits = {"int32": 32, "int16": 16, "int8": 8}[dt]
        r = (0 if nan else sat(-(2**31), 2**31 - 1)) & ((1 << bits) - 1)
        return r - (1 << bits) if r >= 1 << (bits - 1) else r
    if dt in ("uint32", "uint16"):
        r = 0 if nan else sat(0, 2**32 - 1)
        return r & (0xFFFF if dt == "uint16" else 0xFFFFFFFF)
    if dt == "uint64":
        return 2**63 if nan else sat(0, 2**64 - 1)
    r = -(2**63) if nan else sat(-(2**63), 2**63 - 1)
    return r if dt == "int64" else r & 0xFF


_INTS = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]


@pytest.mark.parametrize("dt", _INTS)
@pytest.mark.parametrize("src", FP8, ids=FP8_IDS)
def test_float8_all_codes_to_int_match_stock_cuda(mojo_device, src, dt):
    """Every float8 code, e8m0's [2**31, 2**127] range included: int8 / int16
    narrow the saturated int32 result, not the int64 one."""
    codes = _all_codes(src)
    want = [_cuda_f2i(f, dt) for f in codes.float().tolist()]
    assert codes.to(mojo_device).to(getattr(torch, dt)).cpu().tolist() == want


def test_float8_mask_assign_overlapping_mask_raises(mojo_device):
    """index_put_ checks the mask for overlap with self too, whatever it
    selects (stock CUDA)."""
    t = torch.zeros(4).to(torch.float8_e4m3fn).to(mojo_device)
    v = torch.tensor(2.0).to(torch.float8_e4m3fn).to(mojo_device)
    with pytest.raises(RuntimeError, match="single memory location|overlap"):
        t[t.view(torch.bool)] = v


@pytest.mark.parametrize(
    "op,kernel",
    [
        (lambda x: torch.nn.functional.pad(x, (1, 1), "reflect"), "reflection_pad"),
        (lambda x: torch.nn.functional.pad(x, (1, 1), "replicate"), "replication_pad"),
        (lambda x: torch.copysign(x, x), "copysign_cuda"),
        (lambda x: torch.unique(x), "unique"),
        (lambda x: torch.unique(x[:0]), "unique"),
        (lambda x: torch.unique(x, dim=0), "unique_dim"),
        (lambda x: torch.unique_consecutive(x, dim=0), "unique_dim"),
        (lambda x: torch.lerp(x, x, 0.5), "lerp_cuda"),
        (lambda x: torch.lerp(x, x, x), "lerp_cuda"),
        (lambda x: torch.addcmul(x, x, x), "addcmul_cuda"),
        (lambda x: torch.addcdiv(x, x, x), "addcdiv_cuda"),
        (lambda x: torch.clamp(x, 0, 1), "clamp_scalar_cuda"),
        (lambda x: torch.clamp_min(x, 0), "clamp_scalar_cuda"),
        (lambda x: torch.clamp(x, x, x), "clamp_cuda"),
        (lambda x: torch.hypot(x, x), "hypot_cuda"),
        (lambda x: torch.fmod(x, x), "fmod_cuda"),
        (lambda x: torch.remainder(x, x), "remainder_cuda"),
        (lambda x: torch.maximum(x, x), "max_elementwise_cuda"),
        (lambda x: torch.minimum(x, x), "min_elementwise_cuda"),
        (
            lambda x: torch.copysign(x, torch.tensor(-1.0, device=x.device)),
            "copysign_cuda",
        ),
    ],
    ids=[
        "reflect",
        "replicate",
        "copysign",
        "unique",
        "unique_empty",
        "unique_dim",
        "unique_consecutive_dim",
        "lerp_scalar",
        "lerp_tensor",
        "addcmul",
        "addcdiv",
        "clamp",
        "clamp_min",
        "clamp_tensor",
        "hypot",
        "fmod",
        "remainder",
        "maximum",
        "minimum",
        "copysign_0d",
    ],
)
def test_float8_more_refusals_name_cuda_kernels(mojo_device, op, kernel):
    x = _x(torch.float8_e4m3fn).to(mojo_device)
    with pytest.raises(NotImplementedError, match=kernel):
        op(x)


def test_float8_tril_out_resized_before_refusal(mojo_device):
    x = _x(torch.float8_e4m3fn).to(mojo_device)
    out = torch.empty(0, dtype=x.dtype, device=mojo_device)
    with pytest.raises(NotImplementedError, match="triu_tril_cuda_template"):
        torch.tril(x, out=out)
    assert out.shape == x.shape


def test_float8_runtime_errors_come_before_the_refusal(mojo_device):
    """CUDA's argument checks run before its float8 dispatch."""
    x = _x(torch.float8_e4m3fn).to(mojo_device)
    with pytest.raises(RuntimeError, match="long tensor"):
        x.take(torch.tensor([0], dtype=torch.int32, device=mojo_device))
    with pytest.raises(RuntimeError, match="masked_scatter_"):
        x.clone().masked_scatter_(torch.ones_like(x, dtype=torch.bool), x.float())


def test_index_put_check_order(mojo_device):
    """Overlap, then mask shape, then dtype (stock CUDA)."""
    dev = mojo_device
    t = torch.zeros(3).to(torch.float8_e4m3fn).to(dev)
    bad_mask = torch.tensor([True, False], device=dev)
    with pytest.raises((IndexError, RuntimeError), match="does not match the shape"):
        t[bad_mask] = torch.tensor(2.0, dtype=torch.float64, device=dev)
    with pytest.raises(RuntimeError, match="single memory location"):
        t[bad_mask] = t[0]


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float32])
@pytest.mark.parametrize("mask", list(itertools.product([False, True], repeat=3)))
def test_mask_assign_into_expanded_self(mojo_device, dtype, mask):
    """CUDA's index_put kernel writes only the selected positions, so every
    alias of an expanded self sees the value as soon as one is selected."""
    dev = mojo_device
    t = torch.ones(1).to(dtype).to(dev).expand(3)
    t[torch.tensor(mask, device=dev)] = torch.tensor(3.0).to(dtype).to(dev)
    assert t.float().cpu().tolist() == [3.0 if any(mask) else 1.0] * 3


def test_mask_assign_into_expanded_self_2d(mojo_device):
    dev = mojo_device
    t = torch.ones(2, 1).to(torch.float8_e4m3fn).to(dev).expand(2, 3)
    mask = torch.tensor([[False, True, False], [False, False, False]], device=dev)
    t[mask] = torch.tensor(3.0).to(torch.float8_e4m3fn).to(dev)
    assert t.float().cpu().tolist() == [[3.0] * 3, [1.0] * 3]


@pytest.mark.parametrize("src", [torch.float32, torch.float16, torch.bfloat16])
def test_float8_foreach_copy_does_not_saturate(mojo_device, src):
    """CUDA's _foreach_copy_ fast route converts with c10's static_cast (an
    out-of-range value is NaN), unlike copy_'s sm_89+ saturation."""
    dev = mojo_device
    dst = [torch.zeros(3).to(torch.float8_e4m3fn).to(dev)]
    torch._foreach_copy_(dst, [torch.tensor([500.0, 1.0, -1e6]).to(src).to(dev)])
    assert dst[0].view(torch.uint8).cpu().tolist() == [127, 56, 255]


def test_copysign_float8_promotion(mojo_device):
    """A float8 0-d operand of a float32 copysign converts; a dimensioned one
    does not promote (c10::promoteTypes)."""
    dev = mojo_device
    a = torch.tensor([1.0, 2.0], device=dev)
    got = torch.copysign(a, torch.tensor(-1.0).to(torch.float8_e4m3fn).to(dev))
    assert got.dtype == torch.float32 and got.cpu().tolist() == [-1.0, -2.0]
    with pytest.raises(RuntimeError, match="Promotion for Float8 Types"):
        torch.copysign(a, torch.tensor([-1.0, 1.0]).to(torch.float8_e4m3fn).to(dev))


def test_float8_take_out_checks_before_refusal(mojo_device):
    """take(out=): overlap checks and out's resize come before take_cuda's
    float8 refusal."""
    dev = mojo_device
    x = torch.ones(2, 3).to(torch.float8_e4m3fn).to(dev)
    with pytest.raises(RuntimeError, match="single memory location"):
        torch.take(x, torch.tensor([0, 1], device=dev), out=x.view(-1)[:2])
    out = torch.empty(0, dtype=x.dtype, device=dev)
    with pytest.raises(NotImplementedError, match="take_cuda"):
        torch.take(x, torch.tensor([0, 1, 2], device=dev), out=out)
    assert out.shape == (3,)
