"""Native backend: the float8 dtypes (float8_e4m3fn, float8_e5m2, the two
fnuz variants and float8_e8m0fnu).

Stock CUDA's set is the oracle: allocation, conversions (c10's bit-exact
float8 <-> float32 functions; stock CUDA's sm_89+ e4m3fn saturation on
copies), fills, `.item()`, the dtype-agnostic data movement CUDA dispatches
over AT_FLOAT8_TYPES, eq / ne / where and their type promotion. Arithmetic
CUDA has no float8 kernel for is declined. References are CPU torch, with the
stock-CUDA saturation rule applied where it differs (`_stock_cast`).
"""

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
