"""Native backend: the float8 dtypes (float8_e4m3fn, float8_e5m2, the two
fnuz variants and float8_e8m0fnu) and torch._scaled_mm / _scaled_mm_v2.

Stock CUDA's set is the oracle: allocation, conversions (c10's bit-exact
float8 <-> float32 functions; stock CUDA's sm_89+ e4m3fn saturation on
copies), fills, `.item()`, the dtype-agnostic data movement CUDA dispatches
over AT_FLOAT8_TYPES, eq / ne / where and their type promotion, and the
scaled GEMMs. Arithmetic CUDA has no float8 kernel for is declined.
References are CPU torch, with the stock-CUDA saturation rule applied where
it differs (`_stock_cast`), and, for the GEMMs, a float64 product of the
dequantized operands.
"""

import itertools
import json
import pathlib
import re

import pytest
import torch
from torch.nn.functional import ScalingType, SwizzleType

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


# ---------------------------------------------------------------------------
# torch._scaled_mm / _scaled_mm_v2
# ---------------------------------------------------------------------------

F8_MAX = {
    torch.float8_e4m3fn: 448.0,
    torch.float8_e5m2: 57344.0,
    torch.float8_e4m3fnuz: 240.0,
    torch.float8_e5m2fnuz: 57344.0,
}


def _q(shape: tuple[int, ...], dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g) * 2).to(dtype)


def _expected(a32, b32, sa, sb, bias, out_dtype):
    p = a32.double() @ b32.double()
    p = p * sa.double() * sb.double()
    if bias is not None:
        p = p + bias.double()
    if out_dtype in F8_MAX:
        top = F8_MAX[out_dtype]
        p = p.clamp(-top, top)
    return p.float().to(out_dtype)


def _check(got, want):
    assert got.dtype == want.dtype and got.shape == want.shape
    g, w = got.cpu().double(), want.double()
    tol = {
        torch.float32: (1e-5, 1e-4),
        torch.bfloat16: (1.6e-2, 1e-2),
        torch.float16: (2e-3, 1e-2),
    }.get(got.dtype, (0.13, 1e-2))
    torch.testing.assert_close(g, w, rtol=tol[0], atol=tol[1])


@pytest.mark.parametrize(
    "a_dt,b_dt",
    [
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float8_e4m3fn, torch.float8_e5m2),
        (torch.float8_e5m2, torch.float8_e4m3fn),
    ],
    ids=["e4m3xe4m3", "e4m3xe5m2", "e5m2xe4m3"],
)
@pytest.mark.parametrize(
    "out_dtype",
    [torch.bfloat16, torch.float16, torch.float32, None],
    ids=["bf16", "f16", "f32", "default"],
)
@pytest.mark.parametrize("m,k,n", [(16, 32, 48), (100, 96, 80), (257, 528, 112)])
def test_scaled_mm_tensorwise(mojo_device, a_dt, b_dt, out_dtype, m, k, n):
    a = _q((m, k), a_dt, 1)
    b = _q((n, k), b_dt, 2).t()  # column-major, as cuBLASLt wants it
    sa, sb = torch.tensor(0.5), torch.tensor(1.75)
    got = torch._scaled_mm(
        a.to(mojo_device),
        b.to(mojo_device),
        sa.to(mojo_device),
        sb.to(mojo_device),
        out_dtype=out_dtype,
    )
    _check(got, _expected(a.float(), b.float(), sa, sb, None, out_dtype or a_dt))


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("m,k,n", [(32, 64, 16), (130, 160, 96)])
def test_scaled_mm_rowwise(mojo_device, out_dtype, bias, m, k, n):
    if bias and out_dtype == torch.float32:
        pytest.skip("CUDA refuses a bias with a float32 output")
    a = _q((m, k), torch.float8_e4m3fn, 3)
    b = _q((n, k), torch.float8_e4m3fn, 4).t()
    sa = torch.rand(m, 1) + 0.25
    sb = torch.rand(1, n) + 0.25
    bias_t = (torch.randn(n) * 4).to(out_dtype) if bias else None
    dev = mojo_device
    got = torch._scaled_mm(
        a.to(dev),
        b.to(dev),
        sa.to(dev),
        sb.to(dev),
        bias=None if bias_t is None else bias_t.to(dev),
        out_dtype=out_dtype,
    )
    _check(got, _expected(a.float(), b.float(), sa, sb, bias_t, out_dtype))
    # The out= overload, into a transposed destination.
    out = torch.empty(n, m, dtype=out_dtype, device=dev).t()
    r = torch._scaled_mm(
        a.to(dev),
        b.to(dev),
        sa.to(dev),
        sb.to(dev),
        bias=None if bias_t is None else bias_t.to(dev),
        out_dtype=out_dtype,
        out=out,
    )
    assert r.data_ptr() == out.data_ptr()
    _check(out, _expected(a.float(), b.float(), sa, sb, bias_t, out_dtype))


@pytest.mark.parametrize("out_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_scaled_mm_float8_output_saturates(mojo_device, out_dtype):
    m, k, n = 16, 32, 16
    a = torch.full((m, k), 8.0).to(torch.float8_e4m3fn)
    b = torch.full((n, k), 8.0).to(torch.float8_e4m3fn).t()
    b = b if out_dtype == torch.float8_e4m3fn else b.to(torch.float8_e5m2)
    one = torch.tensor(1.0, device=mojo_device)
    scale = torch.tensor(-1.0 if out_dtype == torch.float8_e4m3fn else 100.0)
    got = torch._scaled_mm(
        a.to(mojo_device),
        b.to(mojo_device),
        scale.to(mojo_device),
        one,
        out_dtype=out_dtype,
    )
    _check(
        got, _expected(a.float(), b.float(), scale, torch.tensor(1.0), None, out_dtype)
    )
    assert got.float().abs().max().item() == F8_MAX[out_dtype]


def _blockwise_case(kind_a: str, kind_b: str, m: int, k: int, n: int):
    """Operands and CUDA-layout scales for a DeepSeek recipe, plus the
    dequantized float32 operands. v1 and v2 lay the scales out differently:
    return both."""
    kb = (k + 127) // 128
    kb4 = (kb + 3) // 4 * 4
    a = _q((m, k), torch.float8_e4m3fn, 5)
    bt = _q((n, k), torch.float8_e4m3fn, 6)
    g = torch.Generator().manual_seed(7)
    rows = {"a": m, "b": n}

    def scales(side: str, kind: str):
        r = rows[side]
        if kind == "1x128":
            s = torch.rand(r, kb, generator=g) + 0.5  # (outer, kb)
            full = s.repeat_interleave(128, 1)[:, :k]
            v1 = s.t().contiguous().t()  # outer-dim-major: strides (1, outer)
            v2 = v1
        else:
            rb = (r + 127) // 128
            s = torch.rand(rb, kb, generator=g) + 0.5
            full = s.repeat_interleave(128, 0)[:r].repeat_interleave(128, 1)[:, :k]
            v1 = s.contiguous()  # (outer blocks, kb) row-major
            pad = torch.zeros(rb, kb4)
            pad[:, :kb] = s
            v2 = pad.t()  # (kb4, outer blocks), strides (1, kb4)
        return full, v1, v2

    fa, a_v1, a_v2 = scales("a", kind_a)
    fb, b_v1, b_v2 = scales("b", kind_b)
    a32 = a.float() * fa
    b32 = (bt.float() * fb).t()
    # v1 checks B's scale through a transpose: hand it (kb, N)-shaped.
    return a, bt.t(), a32, b32, (a_v1, b_v1.t()), (a_v2, b_v2)


@pytest.mark.parametrize(
    "kinds",
    [("1x128", "1x128"), ("1x128", "128x128"), ("128x128", "1x128")],
    ids=["1x128-1x128", "1x128-128x128", "128x128-1x128"],
)
@pytest.mark.parametrize("m,k,n", [(128, 256, 128), (200, 384, 144)])
def test_scaled_mm_blockwise(mojo_device, kinds, m, k, n):
    a, b, a32, b32, v1, v2 = _blockwise_case(kinds[0], kinds[1], m, k, n)
    one = torch.tensor(1.0)
    want = _expected(a32, b32, one, one, None, torch.bfloat16)
    dev = mojo_device
    got = torch._scaled_mm(
        a.to(dev), b.to(dev), v1[0].to(dev), v1[1].to(dev), out_dtype=torch.bfloat16
    )
    _check(got, want)
    rec = {"1x128": ScalingType.BlockWise1x128, "128x128": ScalingType.BlockWise128x128}
    got = torch.nn.functional.scaled_mm(
        a.to(dev),
        b.to(dev),
        v2[0].to(dev),
        rec[kinds[0]],
        v2[1].to(dev),
        rec[kinds[1]],
        output_dtype=torch.bfloat16,
    )
    _check(got, want)


def _to_blocked(s: torch.Tensor) -> torch.Tensor:
    """torch's SWIZZLE_32_4_4 (`to_blocked`) of a (rows, K/32) scale matrix."""
    rows, cols = s.shape
    rb, cb = (rows + 127) // 128, (cols + 3) // 4
    padded = torch.zeros(rb * 128, cb * 4, dtype=s.dtype)
    padded[:rows, :cols] = s
    blocks = padded.view(rb, 128, cb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


@pytest.mark.parametrize("m,k,n", [(128, 128, 128), (160, 320, 48)])
def test_scaled_mm_mxfp8(mojo_device, m, k, n):
    a = _q((m, k), torch.float8_e4m3fn, 8)
    bt = _q((n, k), torch.float8_e4m3fn, 9)
    g = torch.Generator().manual_seed(10)
    ea = torch.randint(120, 134, (m, (k + 31) // 32), generator=g, dtype=torch.uint8)
    eb = torch.randint(120, 134, (n, (k + 31) // 32), generator=g, dtype=torch.uint8)
    fa = ea.view(torch.float8_e8m0fnu).float().repeat_interleave(32, 1)[:, :k]
    fb = eb.view(torch.float8_e8m0fnu).float().repeat_interleave(32, 1)[:, :k]
    sa = _to_blocked(ea).view(torch.float8_e8m0fnu)
    sb = _to_blocked(eb).view(torch.float8_e8m0fnu)
    one = torch.tensor(1.0)
    want = _expected(
        a.float() * fa, (bt.float() * fb).t(), one, one, None, torch.bfloat16
    )
    dev = mojo_device
    got = torch._scaled_mm(
        a.to(dev), bt.t().to(dev), sa.to(dev), sb.to(dev), out_dtype=torch.bfloat16
    )
    _check(got, want)
    got = torch.nn.functional.scaled_mm(
        a.to(dev),
        bt.t().to(dev),
        sa.to(dev),
        ScalingType.BlockWise1x32,
        sb.to(dev),
        ScalingType.BlockWise1x32,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        output_dtype=torch.bfloat16,
    )
    _check(got, want)


def test_scaled_mm_v2_tensorwise_rowwise(mojo_device):
    m, k, n = 48, 64, 32
    a = _q((m, k), torch.float8_e4m3fn, 11)
    b = _q((n, k), torch.float8_e5m2, 12).t()
    dev = mojo_device
    sa, sb = torch.tensor(2.0), torch.tensor(0.25)
    got = torch.nn.functional.scaled_mm(
        a.to(dev),
        b.to(dev),
        sa.to(dev),
        ScalingType.TensorWise,
        sb.to(dev),
        ScalingType.TensorWise,
        output_dtype=torch.float32,
    )
    _check(got, _expected(a.float(), b.float(), sa, sb, None, torch.float32))
    sa, sb = torch.rand(m, 1) + 0.5, torch.rand(1, n) + 0.5
    bias = torch.randn(n).to(torch.bfloat16)
    got = torch.nn.functional.scaled_mm(
        a.to(dev),
        b.to(dev),
        sa.to(dev),
        ScalingType.RowWise,
        sb.to(dev),
        ScalingType.RowWise,
        bias=bias.to(dev),
        output_dtype=torch.bfloat16,
    )
    _check(got, _expected(a.float(), b.float(), sa, sb, bias, torch.bfloat16))


def test_scaled_mm_empty_and_errors(mojo_device):
    dev = mojo_device
    one = torch.tensor(1.0, device=dev)
    a = torch.zeros(16, 0, dtype=torch.float8_e4m3fn, device=dev)
    b = torch.zeros(16, 0, dtype=torch.float8_e4m3fn, device=dev).t()
    out = torch._scaled_mm(a, b, one, one, out_dtype=torch.bfloat16)
    assert out.shape == (16, 16) and torch.equal(
        out.cpu(), torch.zeros(16, 16, dtype=torch.bfloat16)
    )
    a = torch.zeros(16, 32, dtype=torch.float8_e4m3fn, device=dev)
    b = torch.zeros(16, 32, dtype=torch.float8_e4m3fn, device=dev).t()
    err = (RuntimeError, ValueError)
    with pytest.raises(err, match="shapes cannot be multiplied"):
        torch._scaled_mm(a, a, one, one)
    with pytest.raises(err, match="Multiplication of two Float8_e5m2"):
        torch._scaled_mm(a.to(torch.float8_e5m2), b.to(torch.float8_e5m2), one, one)
    with pytest.raises(err, match="Invalid scaling configuration"):
        torch._scaled_mm(a, b, torch.ones(3, device=dev), one)
    with pytest.raises(err, match="divisible by 16"):
        a2 = torch.zeros(16, 24, dtype=torch.float8_e4m3fn, device=dev)
        torch._scaled_mm(a2, a2.t(), one, one)
    with pytest.raises(
        err, match="Bias is not supported when out_dtype is set to Float32"
    ):
        torch._scaled_mm(
            a,
            b,
            one,
            one,
            bias=torch.zeros(16, dtype=torch.bfloat16, device=dev),
            out_dtype=torch.float32,
        )
    with pytest.raises(err, match="Bias must be BFloat16 to compute BFloat16"):
        torch._scaled_mm(
            a,
            b,
            one,
            one,
            bias=torch.zeros(16, dtype=torch.half, device=dev),
            out_dtype=torch.bfloat16,
        )


def _grouped_ref(groups):
    """bfloat16 result of a list of (a, b, sa, sb) rowwise groups."""
    return [
        _expected(a.float(), b.float(), sa[:, None], sb[None, :], None, torch.bfloat16)
        for a, b, sa, sb in groups
    ]


@pytest.mark.parametrize("case", ["3d3d", "2d3d", "2d2d", "3d2d"])
def test_scaled_grouped_mm_rowwise(mojo_device, case):
    dev = mojo_device
    f8 = torch.float8_e4m3fn
    G, M, K, N = 3, 32, 64, 48
    g = torch.Generator().manual_seed(13)

    def rnd(*shape):
        return torch.rand(*shape, generator=g) + 0.5

    if case == "3d3d":
        a = _q((G, M, K), f8, 14)
        b = _q((G, N, K), f8, 15).transpose(1, 2)
        sa, sb, offs = rnd(G, M), rnd(G, N), None
        want = torch.stack(_grouped_ref([(a[i], b[i], sa[i], sb[i]) for i in range(G)]))
    elif case == "2d3d":
        ends = [16, 16, 80]  # an empty group in the middle
        a = _q((80, K), f8, 16)
        b = _q((G, N, K), f8, 17).transpose(1, 2)
        sa, sb = rnd(80), rnd(G, N)
        offs = torch.tensor(ends, dtype=torch.int32)
        starts = [0] + ends[:-1]
        want = torch.cat(
            _grouped_ref(
                [
                    (a[s:e], b[i], sa[s:e], sb[i])
                    for i, (s, e) in enumerate(zip(starts, ends))
                ]
            )
        )
    elif case == "2d2d":
        ends = [16, 48, 64]
        a = _q((M, 64), f8, 18)
        b = _q((N, 64), f8, 19).t()
        sa, sb = rnd(G * M), rnd(G * N)
        offs = torch.tensor(ends, dtype=torch.int32)
        starts = [0] + ends[:-1]
        want = torch.stack(
            _grouped_ref(
                [
                    (
                        a[:, s:e],
                        b[s:e],
                        sa[i * M : (i + 1) * M],
                        sb[i * N : (i + 1) * N],
                    )
                    for i, (s, e) in enumerate(zip(starts, ends))
                ]
            )
        )
    else:
        ends = [16, 32, 48]
        a = _q((G, M, K), f8, 20)
        b = _q((48, K), f8, 21).t()
        sa, sb = rnd(G, M), rnd(48)
        offs = torch.tensor(ends, dtype=torch.int32)
        starts = [0] + ends[:-1]
        want = torch.cat(
            _grouped_ref(
                [
                    (a[i], b[:, s:e], sa[i], sb[s:e])
                    for i, (s, e) in enumerate(zip(starts, ends))
                ]
            ),
            dim=1,
        )

    da, db, dsa, dsb = (t.to(dev) for t in (a, b, sa, sb))
    doffs = None if offs is None else offs.to(dev)
    got = torch._scaled_grouped_mm(
        da, db, dsa, dsb, offs=doffs, out_dtype=torch.bfloat16
    )
    _check(got, want)
    # torch.nn.functional.scaled_grouped_mm (the v2 op), looked up at run
    # time: torch's stubs do not declare it.
    scaled_grouped_mm = vars(torch.nn.functional)["scaled_grouped_mm"]
    got = scaled_grouped_mm(
        da,
        db,
        dsa,
        ScalingType.RowWise,
        dsb,
        ScalingType.RowWise,
        offs=doffs,
        output_dtype=torch.bfloat16,
    )
    _check(got, want)


def test_scaled_grouped_mm_errors(mojo_device):
    dev = mojo_device
    f8 = torch.float8_e4m3fn
    a = torch.zeros(2, 16, 32, dtype=f8, device=dev)
    b = torch.zeros(2, 16, 32, dtype=f8, device=dev).transpose(1, 2)
    s = torch.ones(2, 16, device=dev)
    err = (RuntimeError, ValueError)
    with pytest.raises(err, match="Expected mat2 to be transposed"):
        torch._scaled_grouped_mm(a, b.contiguous(), s, s)
    with pytest.raises(err, match="Only bf16"):
        torch._scaled_grouped_mm(a, b, s, s, out_dtype=torch.float32)
    with pytest.raises(err, match="Have to provide offsets"):
        torch._scaled_grouped_mm(a[0], b, s[0], s)


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


def test_scaled_mm_tensorwise_scales_combine(mojo_device):
    """scale_a * scale_b is one factor (cuBLAS's alpha): 64 * 1e37 then
    * 1e-37 one at a time would overflow to inf. (1e-37, not 1e-38: the
    latter is a float32 subnormal, which an Apple GPU flushes.)"""
    a = torch.full((16, 16), 2.0).to(torch.float8_e4m3fn)
    b = torch.full((16, 16), 2.0).to(torch.float8_e4m3fn).t()
    sa = torch.tensor(1e37, device=mojo_device)
    sb = torch.tensor(1e-37, device=mojo_device)
    got = torch._scaled_mm(
        a.to(mojo_device), b.to(mojo_device), sa, sb, out_dtype=torch.float32
    )
    torch.testing.assert_close(got.cpu(), torch.full((16, 16), 64.0), rtol=1e-5, atol=0)


def test_scaled_mm_blockwise_ignores_tf32(mojo_device):
    """The block-scaled GEMM is strict float32 even when TF32 is allowed:
    128 * 1.0001 - 128 survives (TF32 would round 1.0001 to 1)."""
    m = n = 128
    k = 256
    a = torch.ones(m, k).to(torch.float8_e4m3fn)
    bt = torch.cat([torch.ones(n, 128), -torch.ones(n, 128)], 1)
    bt = bt.to(torch.float8_e4m3fn)
    sa = torch.tensor([1.0001, 1.0]).repeat(m, 1)  # (M, K/128)
    sa = sa.t().contiguous().t()  # outer-dim-major
    sb = torch.ones(n, 2).t().contiguous()  # v1: (K/128, N), stride (N, 1)
    want = torch.full((m, n), 128 * (torch.tensor(1.0001).item() - 1.0))
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        got = torch._scaled_mm(
            a.to(mojo_device),
            bt.t().to(mojo_device),
            sa.to(mojo_device),
            sb.to(mojo_device),
            out_dtype=torch.float32,
        )
    finally:
        torch.set_float32_matmul_precision(previous)
    # float32 summation order costs ~1e-4 against 0.0128; TF32 gives 0.
    torch.testing.assert_close(got.cpu(), want, rtol=0, atol=2e-4)


def test_scaled_mm_scale_result(mojo_device):
    """torch 2.14's test_float8_scale_result: a float8 output is multiplied
    by scale_result (cuBLAS's D scale); a high-precision one ignores it."""
    g = torch.Generator().manual_seed(0)
    a = torch.randint(-1, 2, (32, 16), generator=g).float().to(torch.float8_e4m3fn)
    b = torch.randint(-1, 2, (16, 32), generator=g).float()
    b = b.t().contiguous().t().to(torch.float8_e4m3fn)
    dev = mojo_device
    one = torch.ones((), device=dev)
    sr = torch.full((), 0.5, device=dev)
    da, db = a.to(dev), b.to(dev)
    got = torch._scaled_mm(
        da, db, one, one, scale_result=sr, out_dtype=torch.float8_e4m3fn
    )
    want = (a.float() @ b.float()).mul(0.5).to(torch.float8_e4m3fn)
    _assert_same(got, want)
    hp = torch._scaled_mm(da, db, one, one, scale_result=sr, out_dtype=torch.bfloat16)
    plain = torch._scaled_mm(da, db, one, one, out_dtype=torch.bfloat16)
    _assert_same(hp, plain.cpu())


def test_scaled_mm_rowwise_float8_output(mojo_device):
    """Rowwise scaling into a float8 output (cuBLAS >= 12.9 takes it)."""
    m, k, n = 32, 64, 48
    a = _q((m, k), torch.float8_e4m3fn, 30)
    b = _q((n, k), torch.float8_e4m3fn, 31).t()
    sa = torch.rand(m, 1) + 0.25
    sb = torch.rand(1, n) + 0.25
    dev = mojo_device
    got = torch._scaled_mm(
        a.to(dev), b.to(dev), sa.to(dev), sb.to(dev), out_dtype=torch.float8_e4m3fn
    )
    _check(got, _expected(a.float(), b.float(), sa, sb, None, torch.float8_e4m3fn))


def test_scaled_grouped_mm_device_and_bounds_checks(mojo_device):
    dev = mojo_device
    f8 = torch.float8_e4m3fn
    a = torch.zeros(2, 16, 32, dtype=f8, device=dev)
    b = torch.zeros(2, 16, 32, dtype=f8, device=dev).transpose(1, 2)
    s = torch.ones(2, 16, device=dev)
    with pytest.raises(RuntimeError, match="same device"):
        torch._scaled_grouped_mm(a, b, s.cpu(), s, out_dtype=torch.bfloat16)
    # 2d x 2d: an offset past K is refused before any device read.
    a2 = torch.zeros(16, 32, dtype=f8, device=dev)
    b2 = torch.zeros(16, 32, dtype=f8, device=dev).t()
    offs = torch.tensor([16, 64], dtype=torch.int32, device=dev)
    ones = torch.ones(32, device=dev)
    with pytest.raises(RuntimeError, match="offsets exceed"):
        torch._scaled_grouped_mm(
            a2, b2, ones, ones, offs=offs, out_dtype=torch.bfloat16
        )


def test_scaled_grouped_mm_v2_out(mojo_device):
    if "out" not in torch.ops.aten._scaled_grouped_mm_v2.overloads():
        pytest.skip("this torch has no _scaled_grouped_mm_v2.out overload")
    G, M, K, N = 2, 32, 64, 48
    a = _q((G, M, K), torch.float8_e4m3fn, 32)
    b = _q((G, N, K), torch.float8_e4m3fn, 33).transpose(1, 2)
    g = torch.Generator().manual_seed(34)
    sa = torch.rand(G, M, generator=g) + 0.5
    sb = torch.rand(G, N, generator=g) + 0.5
    want = torch.stack(_grouped_ref([(a[i], b[i], sa[i], sb[i]) for i in range(G)]))
    dev = mojo_device
    out = torch.empty(0, dtype=torch.bfloat16, device=dev)
    r = torch.ops.aten._scaled_grouped_mm_v2.out(
        a.to(dev),
        b.to(dev),
        [sa.to(dev)],
        [1],
        [],
        [sb.to(dev)],
        [1],
        [],
        None,
        None,
        torch.bfloat16,
        [],
        False,
        out=out,
    )
    assert r.data_ptr() == out.data_ptr()
    _check(out, want)


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


def test_float8_meets_complex(mojo_device):
    """float8 against complex64 (stock CUDA 2.11): no promotion either way
    (promoteTypes' error, a wrapped complex number's toComplexType error),
    while conversions between the two work."""
    d = torch.tensor([1.0, 2.0]).to(torch.float8_e4m3fn).to(mojo_device)
    c = torch.ones(2, dtype=torch.complex64, device=mojo_device)
    msg = "Promotion for Float8 Types is not supported, attempted to promote "
    with pytest.raises(RuntimeError, match=msg + "Float8_e4m3fn and ComplexFloat"):
        d == c  # noqa: B015
    with pytest.raises(RuntimeError, match=msg + "ComplexFloat and Float8_e4m3fn"):
        c == d  # noqa: B015
    with pytest.raises(RuntimeError, match=msg + "Float8_e4m3fn and ComplexFloat"):
        d + c  # noqa: B015
    zc = torch.tensor(1j, device=mojo_device)
    with pytest.raises(
        RuntimeError, match="Unknown Complex ScalarType for Float8_e4m3fn"
    ):
        d == zc  # noqa: B015
    m = torch.tensor([True, False], device=mojo_device)
    with pytest.raises(RuntimeError, match=msg + "Float8_e4m3fn and ComplexFloat"):
        torch.where(m, d, c)
    assert d.to(torch.complex64).cpu().tolist() == [1 + 0j, 2 + 0j]
    assert c.clone().copy_(d).cpu().tolist() == [1 + 0j, 2 + 0j]
    assert c.to(torch.float8_e5m2).float().cpu().tolist() == [1.0, 1.0]


@pytest.mark.parametrize(
    "dtype,value,ok",
    [
        (torch.float8_e4m3fn, 1 + 1j, None),
        (torch.float8_e4m3fn, 2 + 0j, 2.0),
        (torch.float8_e4m3fn, 1000 + 0j, None),
        (torch.float8_e8m0fnu, 2 + 0j, None),
        (torch.float8_e8m0fnu, True, 1.0),
    ],
)
def test_float8_fill_complex_scalar(mojo_device, dtype, value, ok):
    """A complex Scalar into float8 (stock CUDA 2.11): each part checked as
    a double, a nonzero imaginary part overflowing (e8m0fnu even a zero one:
    its lowest is 2**-127)."""
    msg = f"value cannot be converted to type c10::{str(dtype)[6:].capitalize()}"
    for make in (
        lambda: torch.empty(2, dtype=dtype, device=mojo_device).fill_(value),
        lambda: torch.full((2,), value, dtype=dtype, device=mojo_device),
    ):
        if ok is None:
            with pytest.raises(RuntimeError, match=msg):
                make()
        else:
            assert make().float().cpu().tolist() == [ok, ok]


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


def test_scaled_grouped_mm_scales_apply_sequentially(mojo_device):
    """The grouped CUTLASS epilogue is ScaleB * (ScaleA * acc): 1e37 then
    1e-37 overflows to inf, as on stock CUDA (unlike _scaled_mm's alpha)."""
    f8 = torch.float8_e4m3fn
    a = torch.full((2, 16, 32), 2.0).to(f8)
    b = torch.full((2, 16, 32), 2.0).to(f8).transpose(1, 2)
    sa = torch.full((2, 16), 1e37)
    sb = torch.full((2, 16), 1e-37)
    dev = mojo_device
    got = torch._scaled_grouped_mm(
        a.to(dev), b.to(dev), sa.to(dev), sb.to(dev), out_dtype=torch.bfloat16
    )
    assert torch.isinf(got.cpu().float()).all()


def test_scaled_grouped_mm_offs_stride(mojo_device):
    f8 = torch.float8_e4m3fn
    dev = mojo_device
    a = torch.zeros(32, 32, dtype=f8, device=dev)
    b = torch.zeros(2, 16, 32, dtype=f8, device=dev).transpose(1, 2)
    offs = torch.tensor([16, 0, 32, 0], dtype=torch.int32, device=dev)[::2]
    with pytest.raises(RuntimeError, match=r"Offsets must have stride \(1,\)"):
        torch._scaled_grouped_mm(
            a,
            b,
            torch.ones(32, device=dev),
            torch.ones(2, 16, device=dev),
            offs=offs,
            out_dtype=torch.bfloat16,
        )


# ---------------------------------------------------------------------------
# review round 3
# ---------------------------------------------------------------------------


def test_scaled_mm_bias_any_shape_reads_flat(mojo_device):
    """torch only checks bias.numel() == N and cuBLAS reads N consecutive
    elements: a (2, 8) bias of a 16-column GEMM is its 16 values in order."""
    a = _q((16, 32), torch.float8_e4m3fn, 40)
    b = _q((16, 32), torch.float8_e4m3fn, 41).t()
    bias = torch.arange(16, dtype=torch.float32).to(torch.bfloat16)
    one = torch.tensor(1.0)
    dev = mojo_device
    got = torch._scaled_mm(
        a.to(dev),
        b.to(dev),
        one.to(dev),
        one.to(dev),
        bias=bias.reshape(2, 8).to(dev),
        out_dtype=torch.bfloat16,
    )
    _check(got, _expected(a.float(), b.float(), one, one, bias, torch.bfloat16))


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


def test_scaled_mm_v2_checks_devices_first(mojo_device):
    dev = mojo_device
    f8 = torch.float8_e4m3fn
    for m in (0, 16):
        a = torch.zeros(m, 32, dtype=f8, device=dev)
        b = torch.zeros(16, 32, dtype=f8, device=dev).t()
        out = torch.empty(3, dtype=torch.bfloat16, device=dev)
        with pytest.raises(RuntimeError, match="same device"):
            torch.ops.aten._scaled_mm_v2.out(
                a,
                b,
                [torch.tensor(1.0)],
                [0],
                [],
                [torch.tensor(1.0, device=dev)],
                [0],
                [],
                None,
                torch.bfloat16,
                [],
                False,
                out=out,
            )
        assert out.shape == (3,)


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


def test_scaled_mm_v2_rowwise_scale_any_shape(mojo_device):
    """A rowwise v2 scale_b only has to hold N floats with stride(1) == 1:
    (2, 8) for 16 columns is read in its logical order, contiguous or not."""
    a = _q((32, 32), torch.float8_e4m3fn, 50)
    b = _q((16, 32), torch.float8_e4m3fn, 51).t()
    sa = torch.rand(32, 1) + 0.5
    flat = torch.rand(16) + 0.5
    dev = mojo_device
    want = _expected(a.float(), b.float(), sa, flat[None, :], None, torch.bfloat16)
    base = torch.zeros(4, 8)
    base[::2] = flat.reshape(2, 8)
    for sb in (flat.reshape(2, 8), base[::2]):
        got = torch.nn.functional.scaled_mm(
            a.to(dev),
            b.to(dev),
            sa.to(dev),
            ScalingType.RowWise,
            sb.to(dev) if sb.is_contiguous() else base.to(dev)[::2],
            ScalingType.RowWise,
            output_dtype=torch.bfloat16,
        )
        _check(got, want)


def test_scaled_mm_noncontiguous_bias_is_stride_correct(mojo_device):
    """Deliberate divergence: cuBLAS reads N consecutive bias elements from
    the data pointer, ignoring strides (measured, torch 2.11 / H100:
    arange(32)[::2] adds 0, 1, ..., 15). We read the view's own elements
    (0, 2, ..., 30), never past the view."""
    a = torch.ones(16, 16).to(torch.float8_e4m3fn)
    one = torch.tensor(1.0, device=mojo_device)
    full = torch.arange(32, dtype=torch.bfloat16).to(mojo_device)
    got = torch._scaled_mm(
        a.to(mojo_device),
        a.to(mojo_device).t(),
        one,
        one,
        bias=full[::2],
        out_dtype=torch.bfloat16,
    )
    want = 16.0 + torch.arange(32, dtype=torch.float32)[::2]
    torch.testing.assert_close(got.cpu().float(), want.expand(16, 16))


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


# ---------------------------------------------------------------------------
# scaled-mm PR review
# ---------------------------------------------------------------------------


def test_scaled_mm_out_aliasing_an_input(mojo_device):
    """out= sharing storage with an input (here scale_a, resized by the call)
    is computed into a temporary first: 16 * 2 * 3 everywhere, as on CUDA."""
    dev = mojo_device
    a = torch.ones(1, 16).to(torch.float8_e4m3fn).to(dev)
    b = torch.ones(16, 16).to(torch.float8_e4m3fn).to(dev).t()
    sa = torch.tensor(2.0, device=dev)
    sb = torch.tensor(3.0, device=dev)
    r = torch._scaled_mm(a, b, sa, sb, out_dtype=torch.float32, out=sa)
    assert r.data_ptr() == sa.data_ptr()
    assert sa.shape == (1, 16) and sa.cpu().tolist() == [[96.0] * 16]
    sa2 = torch.tensor(2.0, device=dev)
    out = torch.ops.aten._scaled_mm_v2.out(
        a, b, [sa2], [0], [], [sb], [0], [], None, torch.float32, [], False, out=sa2
    )
    assert out.cpu().tolist() == [[96.0] * 16]


def test_scaled_mm_block_scales_apply_per_k_block(mojo_device):
    """cuBLASLt block scaling: each K block's partial sum, times scale_a,
    times scale_b, accumulated (measured on stock CUDA: 1e37 then 1e-37
    overflows to inf; 1e-37 then 1e37 does not)."""
    dev = mojo_device
    m, k, n = 128, 256, 128
    a = torch.ones(m, k).to(torch.float8_e4m3fn).to(dev)
    b = torch.ones(n, k).to(torch.float8_e4m3fn).to(dev).t()

    def run(va: float, vb: float) -> float:
        sa = torch.full((k // 128, m), va, device=dev).t()
        sb = torch.full((k // 128, n), vb, device=dev)  # v1: (kb, N)
        out = torch._scaled_mm(a, b, sa, sb, out_dtype=torch.float32)
        return out[0, 0].item()

    assert run(1e37, 1e-37) == float("inf")
    assert run(1e-37, 1e37) == 256.0
    assert run(2.0, 3.0) == 1536.0


def test_scaled_grouped_mm_empty_k_group_runs_epilogue(mojo_device):
    """A 2d x 2d group with no K still multiplies its (zero) accumulator by
    the scales: 0 * inf is NaN, as on CUDA."""
    dev = mojo_device
    f8 = torch.float8_e4m3fn
    a = torch.ones(16, 32).to(f8).to(dev)
    b = torch.ones(16, 32).to(f8).to(dev).t()
    offs = torch.tensor([0, 32], dtype=torch.int32, device=dev)
    sa = torch.ones(32, device=dev)
    sa[:16] = float("inf")
    sb = torch.ones(32, device=dev)
    got = torch._scaled_grouped_mm(a, b, sa, sb, offs=offs, out_dtype=torch.bfloat16)
    got = got.cpu().float()
    assert torch.isnan(got[0]).all()
    assert (got[1] == 32.0).all()


def test_scaled_grouped_mm_requires_e4m3fn(mojo_device):
    dev = mojo_device
    a = torch.zeros(2, 16, 32).to(torch.float8_e5m2).to(dev)
    b = torch.zeros(2, 16, 32).to(torch.float8_e4m3fn).to(dev).transpose(1, 2)
    s = torch.ones(2, 16, device=dev)
    with pytest.raises(
        (RuntimeError, ValueError),
        match="Expected mat_a to be Float8_e4m3 matrix got Float8_e5m2",
    ):
        torch._scaled_grouped_mm(a, b, s, s, out_dtype=torch.bfloat16)
    with pytest.raises(
        (RuntimeError, ValueError),
        match="Expected mat_b to be Float8_e4m3 matrix got Float8_e5m2",
    ):
        torch._scaled_grouped_mm(
            b.transpose(1, 2).contiguous(),
            a.transpose(1, 2),
            s,
            s,
            out_dtype=torch.bfloat16,
        )


# ---------------------------------------------------------------------------
# scaled-mm PR review, round 2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_scaled_mm_rowwise_scale_order(mojo_device, out_dtype):
    """Rowwise scales apply one at a time, scale_b first (measured on stock
    CUDA: 1e-37 / 1e37 overflows, 1e37 / 1e-37 does not), in v1 and v2."""
    dev = mojo_device
    m, k, n = 128, 256, 128
    a = torch.ones(m, k).to(torch.float8_e4m3fn).to(dev)
    b = torch.ones(n, k).to(torch.float8_e4m3fn).to(dev).t()

    def run(va: float, vb: float, v2: bool) -> float:
        sa = torch.full((m, 1), va, device=dev)
        sb = torch.full((1, n), vb, device=dev)
        if v2:
            out = torch.nn.functional.scaled_mm(
                a,
                b,
                sa,
                ScalingType.RowWise,
                sb,
                ScalingType.RowWise,
                output_dtype=out_dtype,
            )
        else:
            out = torch._scaled_mm(a, b, sa, sb, out_dtype=out_dtype)
        return out[0, 0].float().item()

    for v2 in (False, True):
        assert run(1e-37, 1e37, v2) == float("inf")
        assert run(1e37, 1e-37, v2) == 256.0
        assert run(2.0, 3.0, v2) == 1536.0


def test_scaled_mm_v2_deepseek_scale_list_length(mojo_device):
    """Every recipe takes exactly one scale per side: an extra one selects no
    implementation (CUDA's invalid-configuration error)."""
    a, b, _, _, _, v2 = _blockwise_case("1x128", "1x128", 128, 256, 128)
    dev = mojo_device
    sa = v2[0].to(dev)
    with pytest.raises(
        (RuntimeError, ValueError), match="Invalid scaling configuration"
    ):
        torch.ops.aten._scaled_mm_v2(
            a.to(dev),
            b.to(dev),
            [sa, sa],
            [4],
            [],
            [v2[1].to(dev)],
            [4],
            [],
            None,
            torch.bfloat16,
            [],
            False,
        )


def test_scaled_grouped_mm_v2_out_aliasing(mojo_device):
    if "out" not in torch.ops.aten._scaled_grouped_mm_v2.overloads():
        pytest.skip("this torch has no _scaled_grouped_mm_v2.out overload")
    dev = mojo_device
    f8 = torch.float8_e4m3fn
    a = torch.ones(2, 16, 32).to(f8).to(dev)
    b = torch.ones(2, 16, 32).to(f8).to(dev).transpose(1, 2)
    sa = torch.full((2, 16), 2.0, device=dev)
    sb = torch.full((2, 16), 3.0, device=dev)
    alias = sa.view(torch.bfloat16)  # out shares scale_a's storage
    torch.ops.aten._scaled_grouped_mm_v2.out(
        a,
        b,
        [sa],
        [1],
        [],
        [sb],
        [1],
        [],
        None,
        None,
        torch.bfloat16,
        [],
        False,
        out=alias,
    )
    assert alias.shape == (2, 16, 16)
    assert (alias.cpu().float() == 192.0).all()


def test_scaled_mm_empty_with_unsupported_dtypes(mojo_device):
    """CUDA returns an empty / zero result before its output- and
    matrix-dtype limits apply."""
    dev = mojo_device
    one = torch.tensor(1.0, device=dev)
    a = torch.zeros(0, 16).to(torch.float8_e4m3fn).to(dev)
    b = torch.zeros(16, 16).to(torch.float8_e4m3fn).to(dev).t()
    out = torch._scaled_mm(a, b, one, one, out_dtype=torch.float8_e8m0fnu)
    assert out.shape == (0, 16)


def test_scaled_mm_higher_rank_scale_b_refused(mojo_device):
    """get_joint_scaling transposes scale_b first: a 3-D one raises t()'s
    error, while a 3-D singleton scale_a is still a tensorwise scale."""
    dev = mojo_device
    a = torch.randn(16, 32).to(torch.float8_e4m3fn).to(dev)
    b = torch.randn(16, 32).to(torch.float8_e4m3fn).to(dev).t()
    one = torch.ones((), device=dev)
    three = torch.ones(1, 1, 1, device=dev)
    with pytest.raises(RuntimeError, match=r"t\(\) expects a tensor with <= 2"):
        torch._scaled_mm(a, b, one, three, out_dtype=torch.bfloat16)
    assert torch._scaled_mm(a, b, three, one, out_dtype=torch.bfloat16).shape == (
        16,
        16,
    )


def test_scaled_mm_v2_scale_list_lengths(mojo_device):
    """torch 2.14's structured meta (validate_scaled_mm_v2_inputs) checks the
    lists before any kernel: an empty or extra scale names the count."""
    dev = mojo_device
    a = torch.randn(16, 32).to(torch.float8_e4m3fn).to(dev)
    b = torch.randn(16, 32).to(torch.float8_e4m3fn).to(dev).t()
    one = torch.ones((), device=dev)
    tw, rw = ScalingType.TensorWise.value, ScalingType.RowWise.value
    ns = SwizzleType.NO_SWIZZLE.value

    def v2(sa, ra, sb, rb):
        return torch.ops.aten._scaled_mm_v2(
            a, b, sa, ra, [ns] * len(sa), sb, rb, [ns] * len(sb), None,
            torch.bfloat16, [], False,
        )  # fmt: skip

    with pytest.raises(RuntimeError, match="scale_a must have 1 Float element"):
        v2([], [tw], [one], [tw])
    with pytest.raises(RuntimeError, match="scale_b must have 1 Float element"):
        v2([one], [tw], [], [tw])
    with pytest.raises(RuntimeError, match="scale_a must have 1 Float element"):
        v2([one, one], [tw], [one], [tw])
    ra = torch.ones(16, 1, device=dev)
    rb = torch.ones(1, 16, device=dev)
    with pytest.raises(RuntimeError, match="scale_a must have 16 Float elements"):
        v2([ra, ra], [rw], [rb], [rw])
    with pytest.raises(RuntimeError, match="unsupported recipe"):
        v2([one, one], [tw, tw], [one], [tw])


def test_scaled_mm_v2_deepseek_empty_scale_list(mojo_device):
    """The DeepSeek recipes skip the meta's list checks: an empty scale list
    still raises instead of indexing past the end."""
    dev = mojo_device
    a = torch.randn(128, 128).to(torch.float8_e4m3fn).to(dev)
    b = torch.randn(128, 128).to(torch.float8_e4m3fn).to(dev).t()
    r = ScalingType.BlockWise1x128.value
    ns = SwizzleType.NO_SWIZZLE.value
    sb = torch.ones(1, 128, device=dev)
    with pytest.raises(RuntimeError, match="Invalid scaling configuration"):
        torch.ops.aten._scaled_mm_v2(
            a, b, [], [r], [ns], [sb], [r], [ns], None, torch.bfloat16, [], False
        )
    with pytest.raises(RuntimeError, match="Invalid scaling configuration"):
        torch.ops.aten._scaled_mm_v2(
            a, b, [sb], [r], [ns], [], [r], [ns], None, torch.bfloat16, [], False
        )


_N_ROUND = 3 * 2**61 - 2**38 - 1  # rounds differently via float64


# Measured on stock CUDA (torch 2.11 / H100): an int64 scalar converts to
# float8 through ONE int64 -> float32 rounding (c10), not through float64.
@pytest.mark.parametrize(
    "v,fill_byte,eq",
    [
        (_N_ROUND, 189, [True, False, False, False]),
        (2**63 - 1, 190, [False, True, False, False]),
        (2**53 + 1, 180, [False, False, False, False]),
        (1000, 137, [False, False, True, False]),
    ],
)
def test_e8m0_int_scalar_rounds_once(mojo_device, v, fill_byte, eq):
    dev = mojo_device
    f8 = torch.float8_e8m0fnu
    t = torch.zeros(2).to(f8).to(dev)
    t.fill_(v)
    assert t.view(torch.uint8).cpu().tolist() == [fill_byte] * 2
    t.fill_(torch.tensor(v))
    assert t.view(torch.uint8).cpu().tolist() == [fill_byte] * 2
    x = torch.tensor([2.0**62, 2.0**63, 1000.0, 7.0]).to(f8).to(dev)
    assert (x == v).cpu().tolist() == eq
    assert (x != v).cpu().tolist() == [not e for e in eq]


def test_e5m2_compare_huge_int_scalar(mojo_device):
    x = torch.tensor([2.0**62, 2.0**63, 1000.0, 7.0]).to(torch.float8_e5m2)
    x = x.to(mojo_device)
    assert (x == _N_ROUND).cpu().tolist() == [True, True, False, False]


def test_float8_foreach_copy_empty_bumps_version(mojo_device):
    d = torch.empty(0).to(torch.float8_e4m3fn).to(mojo_device)
    v0 = d._version
    torch._foreach_copy_([d], [torch.empty(0, device=mojo_device)])
    assert d._version == v0 + 1


def test_float8_clamp_one_tensor_bound_names_cuda_kernel(mojo_device):
    x = torch.ones(2, 3).to(torch.float8_e4m3fn).to(mojo_device)
    with pytest.raises(NotImplementedError, match="max_elementwise_cuda"):
        torch.clamp(x, min=x)
    with pytest.raises(NotImplementedError, match="min_elementwise_cuda"):
        torch.clamp(x, max=x)
    with pytest.raises(NotImplementedError, match="max_elementwise_cuda"):
        torch.clamp(x, min=x, out=torch.empty_like(x))


_FOREACH_COPY_REFUSED = [torch.float8_e8m0fnu, torch.uint16, torch.uint32, torch.uint64]


@pytest.mark.parametrize("n", [3, 0])
@pytest.mark.parametrize(
    "src,dst",
    [
        (torch.float8_e8m0fnu, torch.float8_e4m3fn),
        (torch.uint16, torch.float8_e4m3fn),
        (torch.uint64, torch.float8_e5m2),
        (torch.float32, torch.float8_e8m0fnu),
        (torch.float8_e4m3fn, torch.uint32),
        (torch.float8_e8m0fnu, torch.float8_e8m0fnu),
        (torch.float8_e8m0fnu, torch.float32),
    ],
    ids=str,
)
def test_float8_foreach_copy_refuses_like_cuda(mojo_device, src, dst, n):
    """CUDA's multi_tensor_apply _foreach_copy_ dispatches dst then src over
    the standard dtypes and four float8s: e8m0fnu and uint16/32/64 refuse,
    empty lists too, before anything is written."""
    dev = mojo_device
    s = torch.tensor([2.0, 1.0, 0.0])[:n].to(src).to(dev)
    d = torch.ones(3)[:n].to(dst).to(dev)
    before = d.cpu().clone()
    v0 = d._version
    refused = dst if dst in _FOREACH_COPY_REFUSED else src
    name = str(refused).removeprefix("torch.").replace("float8", "Float8")
    name = {"uint16": "UInt16", "uint32": "UInt32", "uint64": "UInt64"}.get(name, name)
    with pytest.raises(NotImplementedError, match=f"foreach_tensor_copy.*{name}"):
        torch._foreach_copy_([d], [s])
    assert d._version == v0
    assert torch.equal(d.cpu().view(torch.uint8), before.view(torch.uint8))


# Measured on stock CUDA (torch 2.11 / H100): a float32 result stored into a
# float8 out= converts in the kernel (c10's static_cast, no saturation).
@pytest.mark.parametrize(
    "fn,dt,want",
    [
        (torch.add, torch.float8_e4m3fn, [72, 76, 127]),
        (torch.mul, torch.float8_e4m3fn, [68, 80, 127]),
        (torch.copysign, torch.float8_e4m3fn, [56, 64, 124]),
        (torch.sub, torch.float8_e4m3fn, [192, 192, 121]),
        (torch.add, torch.float8_e5m2, [68, 70, 96]),
        (torch.add, torch.float8_e8m0fnu, [129, 130, 136]),
        (torch.mul, torch.float8_e5m2fnuz, [70, 76, 125]),
    ],
    ids=lambda v: getattr(v, "__name__", str(v).removeprefix("torch.")),
)
def test_float32_op_into_float8_out(mojo_device, fn, dt, want):
    dev = mojo_device
    a = torch.tensor([1.0, 2.0, 400.0], device=dev)
    b = torch.tensor([3.0, 4.0, 100.0], device=dev)
    out = torch.empty(3, dtype=dt, device=dev)
    assert fn(a, b, out=out) is out
    assert out.view(torch.uint8).cpu().tolist() == want


def test_float8_inplace_add_scalar_names_cuda_kernel(mojo_device):
    t = torch.ones(3).to(torch.float8_e4m3fn).to(mojo_device)
    with pytest.raises(NotImplementedError, match="ufunc_add_CUDA"):
        t.add_(1.0)


def test_where_float8_out_names_torch_dtypes(mojo_device):
    dev = mojo_device
    a = torch.tensor([1.0, 2.0], device=dev)
    out = torch.empty(2, dtype=torch.float8_e4m3fn, device=dev)
    with pytest.raises(RuntimeError, match="to be Float but got Float8_e4m3fn"):
        torch.where(a > 1, a, a, out=out)


# Measured on stock CUDA (torch 2.11 / H100). A Python Scalar never promotes
# a float8 tensor: a NaN clamp bound fills the float8 result with NaN (no
# kernel), every other bound reaches clamp_scalar_cuda.
@pytest.mark.parametrize(
    "dt,nan_byte", [(torch.float8_e4m3fn, 127), (torch.float8_e8m0fnu, 255)]
)
def test_float8_clamp_scalar_bounds(mojo_device, dt, nan_byte):
    dev = mojo_device
    x = torch.tensor([1.0, 2.0]).to(dt).to(dev)
    nan = float("nan")
    for r in (
        torch.clamp(x, nan, 1),
        torch.clamp(x, 0, nan),
        torch.clamp_min(x, nan),
        torch.clamp_max(x, nan),
        torch.clamp(x, nan, 1, out=torch.empty(2, dtype=dt, device=dev)),
        x.clone().clamp_(nan, 1),
    ):
        assert r.dtype == dt
        assert r.view(torch.uint8).cpu().tolist() == [nan_byte] * 2
    for s in (float("inf"), -float("inf"), 1, 1.5, True, 0):
        for fn in (
            lambda: torch.clamp(x, s, 1),
            lambda: torch.clamp_min(x, s),
            lambda: torch.clamp_max(x, s),
            lambda: x.clone().clamp_(s, 1),
        ):
            with pytest.raises(NotImplementedError, match="clamp_scalar_cuda"):
                fn()


def test_float8_scalar_ops_match_cuda(mojo_device):
    dev = mojo_device
    x = torch.tensor([1.0, 2.0]).to(torch.float8_e4m3fn).to(dev)
    for s in (float("nan"), 1.5, 2):
        with pytest.raises(NotImplementedError, match="div_true_cuda"):
            x / s
        with pytest.raises(NotImplementedError, match="bitwise_and_cuda"):
            x & s
    with pytest.raises(NotImplementedError, match="nan_to_num_cuda"):
        torch.nan_to_num(x, 0.0)
    with pytest.raises(NotImplementedError, match="mul_cuda"):
        torch.isclose(x, x, rtol=0.0, atol=1.0)
    assert torch.isclose(x, x, rtol=0.0, atol=0.0).cpu().tolist() == [True, True]
    with pytest.raises(RuntimeError, match="Subtraction, the `-` operator"):
        x - True
    # pow: an exponent of 0 / 1 (or a base of 1) needs no kernel.
    assert (x**0).view(torch.uint8).cpu().tolist() == [56, 56]
    assert (x**1).view(torch.uint8).cpu().tolist() == [56, 64]
    assert (x**True).view(torch.uint8).cpu().tolist() == [56, 64]
    assert (1**x).view(torch.uint8).cpu().tolist() == [56, 56]
    with pytest.raises(NotImplementedError, match="pow_cuda"):
        x**1.5
    with pytest.raises(NotImplementedError, match="pow_cuda"):
        2**x


def test_float8_promotion_errors_are_cuda_runtime_errors(mojo_device):
    dev = mojo_device
    i = torch.ones(2, dtype=torch.int32, device=dev)
    for f in (torch.ones(2), torch.tensor(1.0)):
        f8 = f.to(torch.float8_e4m3fn).to(dev)
        with pytest.raises(RuntimeError, match="promote Int and Float8_e4m3fn"):
            i + f8


@pytest.mark.parametrize(
    "fn,kernel",
    [
        (lambda x, o: torch.add(x, x, out=o), "ufunc_add_CUDA"),
        (lambda x, o: torch.mul(x, x, out=o), "mul_cuda"),
        (lambda x, o: torch.div(x, x, out=o), "div_true_cuda"),
        (lambda x, o: torch.add(x, 1, out=o), "ufunc_add_CUDA"),
        (lambda x, o: torch.copysign(x, x, out=o), "copysign_cuda"),
        (lambda x, o: torch.maximum(x, x, out=o), "max_elementwise_cuda"),
        (lambda x, o: torch.atan2(x, x, out=o), "atan2_cuda"),
        (lambda x, o: torch.addcmul(x, x, x, out=o), "addcmul_cuda"),
        (lambda x, o: torch.clamp(x, 0, 1, out=o), "clamp_scalar_cuda"),
        (lambda x, o: torch.neg(x, out=o), "neg_cuda"),
        (lambda x, o: torch.exp(x, out=o), "exp_cuda"),
        (lambda x, o: torch.sqrt(x, out=o), "sqrt_cuda"),
        (lambda x, o: torch.bitwise_and(x, x, out=o), "bitwise_and_cuda"),
    ],
    ids=[
        "add",
        "mul",
        "div",
        "add_s",
        "copysign",
        "maximum",
        "atan2",
        "addcmul",
        "clamp",
        "neg",
        "exp",
        "sqrt",
        "bitand",
    ],
)
def test_float8_out_resized_before_refusal(mojo_device, fn, kernel):
    """TensorIterator resizes `out` before the kernel dispatch refuses."""
    x = torch.ones(2, 3).to(torch.float8_e4m3fn).to(mojo_device)
    out = torch.empty(0, dtype=x.dtype, device=mojo_device)
    with pytest.raises(NotImplementedError, match=kernel):
        fn(x, out)
    assert out.shape == (2, 3)


def test_float8_sum_out_resized_before_refusal(mojo_device):
    dev = mojo_device
    out = torch.empty(0, dtype=torch.float8_e4m3fn, device=dev)
    with pytest.raises(NotImplementedError, match="sum_cuda"):
        torch.sum(torch.ones(2, 3, device=dev), 0, out=out)
    assert out.shape == (3,)


def test_unary_out_dtype_mismatch_wording(mojo_device):
    x = torch.ones(2, device=mojo_device)
    for o, name in ((torch.float8_e4m3fn, "Float8_e4m3fn"), (torch.int32, "Int")):
        out = torch.empty(2, dtype=o, device=mojo_device)
        with pytest.raises(
            RuntimeError, match=f"Found dtype {name} but expected Float"
        ):
            torch.neg(x, out=out)


@pytest.mark.parametrize("strides", [(99, 3), (1, 0), (5, 7)])
def test_float8_foreach_copy_empty_any_strides_refused(mojo_device, strides):
    """An empty tensor is non-overlapping and dense whatever its strides, so
    CUDA's foreach route (and its dtype refusal) applies."""
    dev = mojo_device
    s = torch.empty_strided((0, 3), strides, device=dev)
    d = torch.empty_strided((0, 3), strides, dtype=torch.float8_e8m0fnu, device=dev)
    v0 = d._version
    with pytest.raises(NotImplementedError, match="foreach_tensor_copy"):
        torch._foreach_copy_([d], [s])
    assert d._version == v0


# Measured on stock CUDA (torch 2.11 / H100): TensorIterator's order for an
# out= of a float8 computation -- (1) the out dtype check raises
# RuntimeError with out untouched, (2) out is resized, (3) the kernel
# dispatch raises NotImplementedError. (op, out dtype, outcome, message or
# values, out's shape afterwards.)
_OUT_ORDER_CUDA = [
    ('neg', 'float8_e4m3fn', 'NotImplementedError', '"neg_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('neg', 'float32', 'RuntimeError', 'Found dtype Float but expected Float8_e4m3fn', (0,)),
    ('neg', 'int32', 'RuntimeError', 'Found dtype Int but expected Float8_e4m3fn', (0,)),
    ('neg', 'bool', 'RuntimeError', 'Found dtype Bool but expected Float8_e4m3fn', (0,)),
    ('neg', 'float16', 'RuntimeError', 'Found dtype Half but expected Float8_e4m3fn', (0,)),
    ('abs', 'float8_e4m3fn', 'NotImplementedError', '"abs_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('abs', 'float32', 'RuntimeError', 'Found dtype Float but expected Float8_e4m3fn', (0,)),
    ('abs', 'int32', 'RuntimeError', 'Found dtype Int but expected Float8_e4m3fn', (0,)),
    ('abs', 'bool', 'RuntimeError', 'Found dtype Bool but expected Float8_e4m3fn', (0,)),
    ('abs', 'float16', 'RuntimeError', 'Found dtype Half but expected Float8_e4m3fn', (0,)),
    ('exp', 'float8_e4m3fn', 'NotImplementedError', '"exp_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('exp', 'float32', 'NotImplementedError', '"exp_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('exp', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('exp', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('exp', 'float16', 'NotImplementedError', '"exp_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('sqrt', 'float8_e4m3fn', 'NotImplementedError', '"sqrt_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('sqrt', 'float32', 'NotImplementedError', '"sqrt_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('sqrt', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('sqrt', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('sqrt', 'float16', 'NotImplementedError', '"sqrt_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('sigmoid', 'float8_e4m3fn', 'NotImplementedError', '"sigmoid_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('sigmoid', 'float32', 'NotImplementedError', '"sigmoid_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('sigmoid', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('sigmoid', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('sigmoid', 'float16', 'NotImplementedError', '"sigmoid_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('sign', 'float8_e4m3fn', 'NotImplementedError', '"sign_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('sign', 'float32', 'RuntimeError', 'Found dtype Float but expected Float8_e4m3fn', (0,)),
    ('sign', 'int32', 'RuntimeError', 'Found dtype Int but expected Float8_e4m3fn', (0,)),
    ('sign', 'bool', 'RuntimeError', 'Found dtype Bool but expected Float8_e4m3fn', (0,)),
    ('sign', 'float16', 'RuntimeError', 'Found dtype Half but expected Float8_e4m3fn', (0,)),
    ('add', 'float8_e4m3fn', 'NotImplementedError', '"ufunc_add_CUDA" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('add', 'float32', 'NotImplementedError', '"ufunc_add_CUDA" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('add', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('add', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('add', 'float16', 'NotImplementedError', '"ufunc_add_CUDA" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('add_s', 'float8_e4m3fn', 'NotImplementedError', '"ufunc_add_CUDA" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('add_s', 'float32', 'NotImplementedError', '"ufunc_add_CUDA" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('add_s', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('add_s', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('add_s', 'float16', 'NotImplementedError', '"ufunc_add_CUDA" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('mul', 'float8_e4m3fn', 'NotImplementedError', '"mul_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('mul', 'float32', 'NotImplementedError', '"mul_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('mul', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('mul', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('mul', 'float16', 'NotImplementedError', '"mul_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('div', 'float8_e4m3fn', 'NotImplementedError', '"div_true_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('div', 'float32', 'NotImplementedError', '"div_true_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('div', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('div', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('div', 'float16', 'NotImplementedError', '"div_true_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('sub', 'float8_e4m3fn', 'NotImplementedError', '"ufunc_add_CUDA" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('sub', 'float32', 'NotImplementedError', '"ufunc_add_CUDA" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('sub', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('sub', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('sub', 'float16', 'NotImplementedError', '"ufunc_add_CUDA" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('maximum', 'float8_e4m3fn', 'NotImplementedError', '"max_elementwise_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('maximum', 'float32', 'ok', [1.0, 1.0], (2,)),
    ('maximum', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('maximum', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('maximum', 'float16', 'ok', [1.0, 1.0], (2,)),
    ('bitand', 'float8_e4m3fn', 'NotImplementedError', '"bitwise_and_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('bitand', 'float32', 'NotImplementedError', '"bitwise_and_cuda" not implemented for \'Float\'', (2,)),
    ('bitand', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('bitand', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('bitand', 'float16', 'NotImplementedError', '"bitwise_and_cuda" not implemented for \'Half\'', (2,)),
    ('copysign', 'float8_e4m3fn', 'NotImplementedError', '"copysign_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('copysign', 'float32', 'NotImplementedError', '"copysign_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('copysign', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('copysign', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('copysign', 'float16', 'NotImplementedError', '"copysign_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('atan2', 'float8_e4m3fn', 'NotImplementedError', '"atan2_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('atan2', 'float32', 'NotImplementedError', '"atan2_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('atan2', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('atan2', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('atan2', 'float16', 'NotImplementedError', '"atan2_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('fmod', 'float8_e4m3fn', 'NotImplementedError', '"fmod_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('fmod', 'float32', 'NotImplementedError', '"fmod_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('fmod', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('fmod', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('fmod', 'float16', 'NotImplementedError', '"fmod_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('addcmul', 'float8_e4m3fn', 'NotImplementedError', '"addcmul_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('addcmul', 'float32', 'NotImplementedError', '"addcmul_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('addcmul', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('addcmul', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('addcmul', 'float16', 'NotImplementedError', '"addcmul_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('lerp', 'float8_e4m3fn', 'NotImplementedError', '"lerp_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('lerp', 'float32', 'NotImplementedError', '"lerp_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('lerp', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('lerp', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('lerp', 'float16', 'NotImplementedError', '"lerp_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('clamp_s', 'float8_e4m3fn', 'NotImplementedError', '"clamp_scalar_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('clamp_s', 'float32', 'RuntimeError', 'Found dtype Float but expected Float8_e4m3fn', (0,)),
    ('clamp_s', 'int32', 'RuntimeError', 'Found dtype Int but expected Float8_e4m3fn', (0,)),
    ('clamp_s', 'bool', 'RuntimeError', 'Found dtype Bool but expected Float8_e4m3fn', (0,)),
    ('clamp_s', 'float16', 'RuntimeError', 'Found dtype Half but expected Float8_e4m3fn', (0,)),
    ('clamp_nan', 'float8_e4m3fn', 'ok', [127, 127], (2,)),
    ('clamp_nan', 'float32', 'RuntimeError', 'Found dtype Float but expected Float8_e4m3fn', (0,)),
    ('clamp_nan', 'int32', 'RuntimeError', 'Found dtype Int but expected Float8_e4m3fn', (0,)),
    ('clamp_nan', 'bool', 'RuntimeError', 'Found dtype Bool but expected Float8_e4m3fn', (0,)),
    ('clamp_nan', 'float16', 'RuntimeError', 'Found dtype Half but expected Float8_e4m3fn', (0,)),
    ('clamp_min_nan', 'float8_e4m3fn', 'ok', [127, 127], (2,)),
    ('clamp_min_nan', 'float32', 'RuntimeError', 'Found dtype Float but expected Float8_e4m3fn', (0,)),
    ('clamp_min_nan', 'int32', 'RuntimeError', 'Found dtype Int but expected Float8_e4m3fn', (0,)),
    ('clamp_min_nan', 'bool', 'RuntimeError', 'Found dtype Bool but expected Float8_e4m3fn', (0,)),
    ('clamp_min_nan', 'float16', 'RuntimeError', 'Found dtype Half but expected Float8_e4m3fn', (0,)),
    ('clamp_t', 'float8_e4m3fn', 'NotImplementedError', '"clamp_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('clamp_t', 'float32', 'NotImplementedError', '"clamp_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('clamp_t', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('clamp_t', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('clamp_t', 'float16', 'NotImplementedError', '"clamp_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('clamp_tmin', 'float8_e4m3fn', 'NotImplementedError', '"max_elementwise_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('clamp_tmin', 'float32', 'ok', [1.0, 1.0], (2,)),
    ('clamp_tmin', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('clamp_tmin', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('clamp_tmin', 'float16', 'ok', [1.0, 1.0], (2,)),
    ('clamp_tmax', 'float8_e4m3fn', 'NotImplementedError', '"min_elementwise_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('clamp_tmax', 'float32', 'ok', [1.0, 1.0], (2,)),
    ('clamp_tmax', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('clamp_tmax', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('clamp_tmax', 'float16', 'ok', [1.0, 1.0], (2,)),
    ('pow0', 'float8_e4m3fn', 'ok', [56, 56], (2,)),
    ('pow0', 'float32', 'ok', [1.0, 1.0], (2,)),
    ('pow0', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('pow0', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('pow0', 'float16', 'ok', [1.0, 1.0], (2,)),
    ('pow1', 'float8_e4m3fn', 'ok', [56, 56], (2,)),
    ('pow1', 'float32', 'ok', [1.0, 1.0], (2,)),
    ('pow1', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('pow1', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('pow1', 'float16', 'ok', [1.0, 1.0], (2,)),
    ('pow2', 'float8_e4m3fn', 'NotImplementedError', '"pow_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('pow2', 'float32', 'NotImplementedError', '"pow_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('pow2', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (0,)),
    ('pow2', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (0,)),
    ('pow2', 'float16', 'NotImplementedError', '"pow_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('rpow1', 'float8_e4m3fn', 'ok', [56, 56], (2,)),
    ('rpow1', 'float32', 'ok', [1.0, 1.0], (2,)),
    ('rpow1', 'int32', 'ok', [1, 1], (2,)),
    ('rpow1', 'bool', 'ok', [True, True], (2,)),
    ('rpow1', 'float16', 'ok', [1.0, 1.0], (2,)),
    ('rpow2', 'float8_e4m3fn', 'NotImplementedError', '"pow_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('rpow2', 'float32', 'NotImplementedError', '"pow_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('rpow2', 'int32', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Int", (2,)),
    ('rpow2', 'bool', 'RuntimeError', "result type Float8_e4m3fn can't be cast to the desired output type Bool", (2,)),
    ('rpow2', 'float16', 'NotImplementedError', '"pow_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('sum', 'float8_e4m3fn', 'NotImplementedError', '"sum_cuda" not implemented for \'Float8_e4m3fn\'', ()),
    ('sum', 'float32', 'ok', 2.0, ()),
    ('sum', 'float16', 'ok', 2.0, ()),
    ('eq', 'float8_e4m3fn', 'ok', [56, 56], (2,)),
    ('eq', 'float32', 'ok', [1.0, 1.0], (2,)),
    ('eq', 'int32', 'ok', [1, 1], (2,)),
    ('eq', 'bool', 'ok', [True, True], (2,)),
    ('eq', 'float16', 'ok', [1.0, 1.0], (2,)),
    ('ne_s', 'float8_e4m3fn', 'ok', [0, 0], (2,)),
    ('ne_s', 'float32', 'ok', [0.0, 0.0], (2,)),
    ('ne_s', 'int32', 'ok', [0, 0], (2,)),
    ('ne_s', 'bool', 'ok', [False, False], (2,)),
    ('ne_s', 'float16', 'ok', [0.0, 0.0], (2,)),
    ('nan_to_num', 'float8_e4m3fn', 'NotImplementedError', '"nan_to_num_cuda" not implemented for \'Float8_e4m3fn\'', (2,)),
    ('nan_to_num', 'float32', 'RuntimeError', 'nan_to_num: dtype of out: Float should be same as input: Float8_e4m3fn', (0,)),
    ('nan_to_num', 'int32', 'RuntimeError', 'nan_to_num: dtype of out: Int should be same as input: Float8_e4m3fn', (0,)),
    ('nan_to_num', 'bool', 'RuntimeError', 'nan_to_num: dtype of out: Bool should be same as input: Float8_e4m3fn', (0,)),
    ('nan_to_num', 'float16', 'RuntimeError', 'nan_to_num: dtype of out: Half should be same as input: Float8_e4m3fn', (0,)),
    ('where', 'float8_e4m3fn', 'ok', [56, 56], (2,)),
    ('where', 'float32', 'RuntimeError', 'Expected out type to be Float8_e4m3fn but got Float', (0,)),
    ('where', 'int32', 'RuntimeError', 'Expected out type to be Float8_e4m3fn but got Int', (0,)),
    ('where', 'bool', 'RuntimeError', 'Expected out type to be Float8_e4m3fn but got Bool', (0,)),
    ('where', 'float16', 'RuntimeError', 'Expected out type to be Float8_e4m3fn but got Half', (0,)),
]  # fmt: skip


def _out_order_ops(dev):
    x = torch.ones(2).to(torch.float8_e4m3fn).to(dev)
    nan = float("nan")
    t = torch.ones(2, dtype=torch.bool, device=dev)
    return {
        "neg": lambda o: torch.neg(x, out=o),
        "abs": lambda o: torch.abs(x, out=o),
        "exp": lambda o: torch.exp(x, out=o),
        "sqrt": lambda o: torch.sqrt(x, out=o),
        "sigmoid": lambda o: torch.sigmoid(x, out=o),
        "sign": lambda o: torch.sign(x, out=o),
        "add": lambda o: torch.add(x, x, out=o),
        "add_s": lambda o: torch.add(x, 1, out=o),
        "mul": lambda o: torch.mul(x, x, out=o),
        "div": lambda o: torch.div(x, x, out=o),
        "sub": lambda o: torch.sub(x, x, out=o),
        "maximum": lambda o: torch.maximum(x, x, out=o),
        "bitand": lambda o: torch.bitwise_and(x, x, out=o),
        "copysign": lambda o: torch.copysign(x, x, out=o),
        "atan2": lambda o: torch.atan2(x, x, out=o),
        "fmod": lambda o: torch.fmod(x, x, out=o),
        "addcmul": lambda o: torch.addcmul(x, x, x, out=o),
        "lerp": lambda o: torch.lerp(x, x, 0.5, out=o),
        "clamp_s": lambda o: torch.clamp(x, 0, 1, out=o),
        "clamp_nan": lambda o: torch.clamp(x, nan, 1, out=o),
        "clamp_min_nan": lambda o: torch.clamp_min(x, nan, out=o),
        "clamp_t": lambda o: torch.clamp(x, x, x, out=o),
        "clamp_tmin": lambda o: torch.clamp(x, min=x, out=o),
        "clamp_tmax": lambda o: torch.clamp(x, max=x, out=o),
        "pow0": lambda o: torch.pow(x, 0, out=o),
        "pow1": lambda o: torch.pow(x, 1, out=o),
        "pow2": lambda o: torch.pow(x, 2, out=o),
        "rpow1": lambda o: torch.pow(1, x, out=o),
        "rpow2": lambda o: torch.pow(2, x, out=o),
        "sum": lambda o: torch.sum(x, 0, out=o),
        "eq": lambda o: torch.eq(x, x, out=o),
        "ne_s": lambda o: torch.ne(x, 1.0, out=o),
        "nan_to_num": lambda o: torch.nan_to_num(x, out=o),
        "where": lambda o: torch.where(t, x, x, out=o),
    }


@pytest.mark.parametrize("case", _OUT_ORDER_CUDA, ids=lambda c: f"{c[0]}-{c[1]}")
def test_float8_out_order_matches_cuda(mojo_device, case):
    op, odt, outcome, want, shape = case
    out = torch.empty(0, dtype=getattr(torch, odt), device=mojo_device)
    fn = _out_order_ops(mojo_device)[op]
    if outcome == "ok":
        fn(out)
        got = out.view(torch.uint8) if out.dtype in FP8 else out
        assert got.cpu().tolist() == want
    else:
        exc = NotImplementedError if outcome == "NotImplementedError" else RuntimeError
        with pytest.raises(exc) as info:
            fn(out)
        assert type(info.value) is exc
        assert str(info.value).startswith(want)
    assert tuple(out.shape) == shape


def test_float8_cpu_0d_operands_match_cuda(mojo_device):
    dev = mojo_device
    f8 = torch.float8_e4m3fn
    x = torch.tensor([1.0, 2.0]).to(f8).to(dev)
    assert torch.isclose(
        x, torch.tensor(1.0).to(f8), rtol=0, atol=0
    ).cpu().tolist() == [True, False]
    nan8 = torch.tensor(float("nan")).to(f8)
    assert torch.isclose(x, nan8, rtol=0, atol=0, equal_nan=True).cpu().tolist() == [
        False,
        False,
    ]
    one = torch.tensor(1.0).to(f8).to(dev)
    r = torch.ones(2, device=dev) + one
    assert r.dtype == torch.float32 and r.cpu().tolist() == [2.0, 2.0]
    r = torch.ones(2, dtype=torch.float16, device=dev) + one
    assert r.dtype == torch.float16 and r.cpu().tolist() == [2.0, 2.0]
    with pytest.raises(NotImplementedError, match="pow_cuda"):
        torch.pow(x, torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="promote Float8_e4m3fn and Float"):
        torch.pow(x, torch.ones(2, device=dev))
    with pytest.raises(RuntimeError, match="promote Float8_e4m3fn and Float"):
        torch.cat([x, torch.ones(2, device=dev)])
    with pytest.raises(RuntimeError, match="promote Float and Float8_e4m3fn"):
        torch.cat([torch.ones(2, device=dev), x])


# ---------------------------------------------------------------------------
# TensorIterator's pre-dispatch sequence (tmb/ops/f8_iter.mojo), op by op:
# every unary / binary / comparison / logical op against stock CUDA in each
# float8 situation -- float8 self / other / out, 0-d and CPU 0-d operands,
# fresh / empty / wrong-dtype / partially and internally overlapping outs.
# float8_elementwise_cuda.json holds the lines measured on stock CUDA (torch
# 2.11 / H100) with exactly the generator below.
# ---------------------------------------------------------------------------

_F8_MATRIX = json.loads(
    (pathlib.Path(__file__).parent / "float8_elementwise_cuda.json").read_text()
)
_MATRIX_UNARY = {
    "abs", "neg", "sign", "exp", "exp2", "expm1", "log", "log2", "log10", "log1p",
    "sqrt", "rsqrt", "sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos",
    "atan", "asinh", "acosh", "atanh", "sigmoid", "erf", "erfc", "erfinv",
    "reciprocal", "ceil", "floor", "round", "trunc", "frac", "nan_to_num",
    "isnan", "isinf", "isfinite", "signbit", "bitwise_not", "logical_not", "sgn",
    "lgamma", "digamma", "i0", "square", "relu", "angle", "positive", "deg2rad",
    "rad2deg", "fix",
}  # fmt: skip


def _mx_enc(t):
    if not isinstance(t, torch.Tensor):
        return repr(t)[:40]
    v = t.view(torch.uint8) if t.dtype in FP8 else t
    return f"{str(t.dtype)[6:]}{tuple(t.shape)}{v.cpu().tolist()}"


def _mx_res(fn, out=None, inp=()):
    before = [_mx_enc(i) for i in inp]
    try:
        s = "OK " + _mx_enc(fn())
    except Exception as e:  # noqa: BLE001 -- the outcome itself is compared
        msg = str(e).split("\n")[0].split(" [aten")[0][:90]
        s = type(e).__name__ + " " + msg
    if out is not None:
        s += f" | out={str(out.dtype)[6:]}{tuple(out.shape)}"
    if [_mx_enc(i) for i in inp] != before:
        s += " | INPUT MUTATED"
    return s


def _mx_unary(name, dev):
    f = getattr(torch, name)
    f8 = torch.float8_e4m3fn

    def mk(vals):
        return torch.tensor(vals).to(f8).to(dev)

    a = mk([1.0, 2.0])
    yield f"{name} fn {_mx_res(lambda: f(a))}"
    for odt in (f8, torch.float32, torch.int32, torch.bool):
        o = torch.empty(0, dtype=odt, device=dev)
        yield f"{name} out_empty {str(odt)[6:]} {_mx_res(lambda: f(a, out=o), o)}"
    o = torch.empty(2, dtype=f8, device=dev)
    yield f"{name} out_fresh {_mx_res(lambda: f(a, out=o), o)}"
    b = mk([1.0, 2.0, 3.0, 4.0])
    o = b[1:]
    yield f"{name} out_partial_overlap {_mx_res(lambda: f(b[:2], out=o), o, (b,))}"
    o = torch.empty(1, dtype=f8, device=dev).expand(2)
    yield f"{name} out_internal_overlap {_mx_res(lambda: f(a, out=o), o)}"
    yield f"{name} fn_0d {_mx_res(lambda: f(mk(1.0)))}"


def _mx_binary(name, dev):
    f = getattr(torch, name)
    f8 = torch.float8_e4m3fn

    def mk(vals, dt):
        return torch.tensor(vals).to(dt).to(dev)

    a8, a32 = mk([1.0, 2.0], f8), mk([1.0, 2.0], torch.float32)
    z8, z32 = mk(2.0, f8), mk(2.0, torch.float32)
    combos = {
        "f8_f8": (a8, a8), "f8_f32": (a8, a32), "f32_f8": (a32, a8),
        "f8_0d32": (a8, z32), "0d8_f32": (z8, a32), "f32_0d8": (a32, z8),
        "0d32_f8": (z32, a8), "f8_scalar": (a8, 2.0), "f8_iscalar": (a8, 2),
        "f8_cpu0d8": (a8, torch.tensor(2.0).to(f8)),
    }  # fmt: skip
    for k, (x, y) in combos.items():
        yield f"{name} fn {k} {_mx_res(lambda: f(x, y))}"
    for odt in (f8, torch.float32, torch.int32, torch.bool):
        o = torch.empty(0, dtype=odt, device=dev)
        res = _mx_res(lambda: f(a8, a8, out=o), o)
        yield f"{name} out_empty_f8f8 {str(odt)[6:]} {res}"
    o = torch.empty(0, dtype=f8, device=dev)
    yield f"{name} out_empty_f32f32_to_f8 {_mx_res(lambda: f(a32, a32, out=o), o)}"
    b = mk([1.0, 2.0, 3.0, 4.0], f8)
    o = b[1:]
    res = _mx_res(lambda: f(b[:2], a8, out=o), o, (b,))
    yield f"{name} out_partial_overlap {res}"
    o = torch.empty(1, dtype=f8, device=dev).expand(2)
    yield f"{name} out_internal_overlap {_mx_res(lambda: f(a8, a8, out=o), o)}"


_F8_ITER = json.loads(
    (pathlib.Path(__file__).parent / "float8_iter_cuda.json").read_text()
)
# Lines where the mojo device differs from stock CUDA for reasons that are
# not float8's (each reproduces with float32 operands): CUDA's per-op
# dispatcher device check ("got self is on cuda:0, different from other
# tensors on cpu (when checking argument in method wrapper_CUDA_...)") for a
# CPU operand or out of an op without CPU scalars (logaddexp, nextafter,
# igamma, the backward ops, silu / mish / glu out=); sigmoid/tanh_backward's
# -0.0 where CUDA has 0.0; silu's last float32 bit; the "aten::mul:" prefix
# of a decomposed backward op's cast error; and int8 / int16 / uint8 add /
# sub with alpha != 1, which the mojo device declines (NotImplementedError)
# into any out dtype -- it has no scale kernel for those integer dtypes.
_KNOWN = set(_F8_ITER["known_differences"])


# The JSON holds what the installed stock CUDA torch (2.11) prints; where
# torch 2.14's source changes the outcome, the mojo device follows 2.14 and
# this maps the measured line to the 2.14 one (ldexp's _ldexp_int_exponent
# fast path also wants out.dtype == self.dtype since 2.14; otherwise the
# fallback's _pow2 meets the float8 promotion error, out untouched).
_V214 = _F8_ITER["torch_2_14_overrides"]


_FLOAT_LIT = re.compile(r"-?\d+\.\d+(?:e[+-]?\d+)?|-?\d+e[+-]?\d+")


def _hw_free(line):
    """A line with its non-float8 float values rounded to 5 significant
    digits and signed zeros folded: transcendental results (and the sign of
    a zero computed as a difference) may differ in the last ulp between GPU
    generations (H100 vs L4). float8 values compare exactly as bytes."""
    return _FLOAT_LIT.sub(lambda m: format(float(m.group()) + 0.0, ".5g"), line)


def _compare(got, want):
    pairs = [
        (_hw_free(g), _hw_free(_V214.get(w, w)))
        for g, w in zip(got, want, strict=True)
        if w not in _KNOWN
    ]
    assert [g for g, _ in pairs] == [w for _, w in pairs]


@pytest.mark.parametrize("op", list(_F8_MATRIX))
def test_float8_elementwise_matches_stock_cuda(mojo_device, op):
    gen = _mx_unary if op in _MATRIX_UNARY else _mx_binary
    _compare(list(gen(op, mojo_device)), _F8_MATRIX[op])


_R9_CUDA = [
    'pow_0d8_f32 OK float32(2,)[8.0, 8.0]',
    'pow_tt_out_torch.float8_e4m3fn NotImplementedError "pow_cuda" not implemented for \'Float8_e4m3fn\' | out=(2,)',
    'pow_tt_out_torch.float32 NotImplementedError "pow_cuda" not implemented for \'Float8_e4m3fn\' | out=(2,)',
    "pow_tt_out_torch.int32 RuntimeError result type Float8_e4m3fn can't be cast to the desired output type Int | out=(0,)",
    'gt_out_torch.float32 NotImplementedError "compare_cuda" not implemented for \'Float8_e4m3fn\' | out=(2,)',
    'gt_out_torch.bool NotImplementedError "compare_cuda" not implemented for \'Float8_e4m3fn\' | out=(2,)',
    'logical_and_out_torch.float32 NotImplementedError "logical_and_cuda" not implemented for \'Float8_e4m3fn\' | out=(2,)',
    'logical_and_out_torch.bool NotImplementedError "logical_and_cuda" not implemented for \'Float8_e4m3fn\' | out=(2,)',
    'logical_or_out_torch.float32 NotImplementedError "logical_or_cuda" not implemented for \'Float8_e4m3fn\' | out=(2,)',
    'logical_or_out_torch.bool NotImplementedError "logical_or_cuda" not implemented for \'Float8_e4m3fn\' | out=(2,)',
    'logical_xor_out_torch.float32 NotImplementedError "logical_xor_cuda" not implemented for \'Float8_e4m3fn\' | out=(2,)',
    'logical_xor_out_torch.bool NotImplementedError "logical_xor_cuda" not implemented for \'Float8_e4m3fn\' | out=(2,)',
    'neg_overlap RuntimeError unsupported operation: some elements of the input tensor and the written-to tensor refer to a single | out=(3,)',
    'exp_overlap RuntimeError unsupported operation: some elements of the input tensor and the written-to tensor refer to a single | out=(3,)',
    'nan_to_num_overlap RuntimeError unsupported operation: some elements of the input tensor and the written-to tensor refer to a single | out=(3,)',
    'lt_cpu0d_out NotImplementedError "compare_cuda" not implemented for \'Float8_e4m3fn\' | out=(2,)',
    'maximum_e5out NotImplementedError "max_elementwise_cuda" not implemented for \'Float8_e5m2\' | out=(2,)',
    'bitand_e5out NotImplementedError "bitwise_and_cuda" not implemented for \'Float8_e5m2\' | out=(2,)',
    'clamp_tmin_e5out NotImplementedError "max_elementwise_cuda" not implemented for \'Float8_e5m2\' | out=(2,)',
    'minimum_e5out NotImplementedError "min_elementwise_cuda" not implemented for \'Float8_e5m2\' | out=(2,)',
    'pow_int_f8 RuntimeError Promotion for Float8 Types is not supported, attempted to promote Int and Float8_e4m3fn',
    'pow_bool_f8 RuntimeError Promotion for Float8 Types is not supported, attempted to promote Bool and Float8_e4m3fn',
    'isclose_tol NotImplementedError "mul_cuda" not implemented for \'Float8_e4m3fn\'',
    'isclose_tol_cpu0d NotImplementedError "mul_cpu_reduced_float" not implemented for \'Float8_e4m3fn\'',
    'div_floor NotImplementedError "div_floor_cuda" not implemented for \'Float8_e4m3fn\'',
    'div_trunc NotImplementedError "div_trunc_cuda" not implemented for \'Float8_e4m3fn\'',
    'div_floor_out NotImplementedError "div_floor_cuda" not implemented for \'Float8_e4m3fn\' | out=(2,)',
]  # fmt: skip


def _r9_lines(dev):
    f8, e5 = torch.float8_e4m3fn, torch.float8_e5m2

    def run(name, fn, out=None):
        try:
            s = "OK " + _mx_enc(fn())
        except Exception as e:  # noqa: BLE001 -- the outcome itself is compared
            s = type(e).__name__ + " " + str(e).split("\n")[0].split(" [aten")[0][:100]
        if out is not None:
            s += f" | out={tuple(out.shape)}"
        return f"{name} {s}"

    x = torch.tensor([1.0, 2.0]).to(f8).to(dev)
    base = torch.tensor(2.0).to(f8).to(dev)
    yield run("pow_0d8_f32", lambda: torch.pow(base, torch.full((2,), 3.0, device=dev)))
    for odt in (f8, torch.float32, torch.int32):
        o = torch.empty(0, dtype=odt, device=dev)
        yield run(f"pow_tt_out_{odt}", lambda: torch.pow(x, x, out=o), o)
    for fn in ("gt", "logical_and", "logical_or", "logical_xor"):
        for odt in (torch.float32, torch.bool):
            o = torch.empty(0, dtype=odt, device=dev)
            yield run(f"{fn}_out_{odt}", lambda: getattr(torch, fn)(x, x, out=o), o)
    a = torch.ones(4).to(f8).to(dev)
    for fn in ("neg", "exp", "nan_to_num"):
        o = a[1:]
        yield run(f"{fn}_overlap", lambda: getattr(torch, fn)(a[:2], out=o), o)
    o = torch.empty(0, dtype=torch.bool, device=dev)
    yield run("lt_cpu0d_out", lambda: torch.lt(x, torch.tensor(1.5), out=o), o)
    for name, f in [
        ("maximum", lambda o: torch.maximum(x, x, out=o)),
        ("bitand", lambda o: torch.bitwise_and(x, x, out=o)),
        ("clamp_tmin", lambda o: torch.clamp(x, min=x, out=o)),
        ("minimum", lambda o: torch.minimum(x, x, out=o)),
    ]:
        o = torch.empty(0, dtype=e5, device=dev)
        yield run(f"{name}_e5out", lambda: f(o), o)
    ones_i = torch.ones(2, dtype=torch.int32, device=dev)
    yield run("pow_int_f8", lambda: torch.pow(ones_i, x))
    ones_b = torch.ones(2, dtype=torch.bool, device=dev)
    yield run("pow_bool_f8", lambda: torch.pow(ones_b, x))
    yield run("isclose_tol", lambda: torch.isclose(x, x))
    yield run("isclose_tol_cpu0d", lambda: torch.isclose(x, torch.tensor(1.0).to(f8)))
    yield run("div_floor", lambda: torch.div(x, x, rounding_mode="floor"))
    yield run("div_trunc", lambda: torch.div(x, x, rounding_mode="trunc"))
    o = torch.empty(0, dtype=f8, device=dev)
    yield run("div_floor_out", lambda: torch.div(x, x, rounding_mode="floor", out=o), o)


def test_float8_review_items_match_stock_cuda(mojo_device):
    assert list(_r9_lines(mojo_device)) == _R9_CUDA


def _r10_res(fn, out=None, watch=()):
    before = [_r10_enc(w) for w in watch]
    try:
        s = "OK " + _r10_enc(fn())
    except Exception as e:  # noqa: BLE001 -- the outcome itself is compared
        s = type(e).__name__ + " " + str(e).split("\n")[0].split(" [aten")[0][:100]
    if out is not None:
        s += f" | out={str(out.dtype)[6:]}{tuple(out.shape)}"
    if [_r10_enc(w) for w in watch] != before:
        s += " | MUTATED " + ",".join(_r10_enc(w) for w in watch)
    return s


def _r10_enc(t):
    if isinstance(t, tuple):
        return "(" + ",".join(_r10_enc(x) for x in t) + ")"
    return _mx_enc(t)


def _r10_lines(dev):
    """Per-op dtype rules beyond TensorIterator's default (lerp,
    log_sigmoid_backward's buffer, the backward ops, addcmul/addcdiv,
    where, clamp with tensors) and an out= aliasing an input element."""
    f8, a = torch.float8_e4m3fn, torch.ops.aten

    def t(vals, dt):
        return torch.tensor(vals).to(dt).to(dev)

    k = {
        "f8_0d": lambda: t(0.5, f8),
        "f8": lambda: t([0.5, 1.0], f8),
        "f32": lambda: t([1.0, 2.0], torch.float32),
        "f32_0d": lambda: t(0.5, torch.float32),
    }
    mask = torch.tensor([True, False], device=dev)
    ops3 = {
        "lerp.Tensor": torch.lerp,
        "addcmul": torch.addcmul,
        "addcdiv": torch.addcdiv,
        "clamp.Tensor": torch.clamp,
        "where": lambda x, y, z: torch.where(mask, y, z),
        "log_sigmoid_backward": a.log_sigmoid_backward,
    }
    combos3 = [
        ("f8_0d", "f32", "f32"), ("f32", "f32", "f8_0d"), ("f32", "f8_0d", "f32"),
        ("f8", "f32", "f32"), ("f32", "f32", "f8"), ("f8", "f8", "f8"),
        ("f32", "f8", "f8"), ("f8_0d", "f8_0d", "f32_0d"),
    ]  # fmt: skip
    for name, f in ops3.items():
        for c in combos3:
            yield f"{name} {'/'.join(c)} " + _r10_res(lambda: f(*[k[x]() for x in c]))
    ops2 = {
        "lerp.Scalar": lambda x, y: torch.lerp(x, y, 0.5),
        "sigmoid_backward": a.sigmoid_backward,
        "tanh_backward": a.tanh_backward,
        "gelu_backward": a.gelu_backward,
        "silu_backward": a.silu_backward,
        "hardsigmoid_backward": a.hardsigmoid_backward,
        "hardswish_backward": a.hardswish_backward,
        "mish_backward": a.mish_backward,
        "logit_backward": a.logit_backward,
        "threshold_backward": lambda x, y: a.threshold_backward(x, y, 0.5),
        "hardtanh_backward": lambda x, y: a.hardtanh_backward(x, y, -1.0, 1.0),
        "leaky_relu_backward": lambda x, y: a.leaky_relu_backward(x, y, 0.01, False),
        "softplus_backward": lambda x, y: a.softplus_backward(x, y, 1.0, 20.0),
        "elu_backward": lambda x, y: a.elu_backward(x, 1.0, 1.0, 1.0, False, y),
        "clamp_min.Tensor": torch.clamp_min,
    }
    combos2 = [
        ("f8_0d", "f32"), ("f32", "f8_0d"), ("f8", "f32"),
        ("f32", "f8"), ("f8", "f8"), ("f8_0d", "f8_0d"),
    ]  # fmt: skip
    for name, f in ops2.items():
        for c in combos2:
            yield f"{name} {'/'.join(c)} " + _r10_res(lambda: f(*[k[x]() for x in c]))
    y, w8 = t([1.0, 2.0], torch.float32), t(0.5, f8)
    for odt in (torch.float16, torch.float32, f8):
        o = torch.empty(0, dtype=odt, device=dev)
        r = _r10_res(lambda: torch.lerp(y, y, w8, out=o), o)
        yield f"lerp_w8_out {str(odt)[6:]} {r}"
        o = torch.empty(0, dtype=odt, device=dev)
        r = _r10_res(lambda: torch.lerp(y, y, 0.5, out=o), o)
        yield f"lerp_s_out {str(odt)[6:]} {r}"
        o = torch.empty(0, dtype=odt, device=dev)
        x8 = t([1.0, 2.0], f8)
        r = _r10_res(lambda: torch.lerp(x8, x8, 0.5, out=o), o)
        yield f"lerp_f8f8_out {str(odt)[6:]} {r}"
        o = torch.empty(0, dtype=odt, device=dev)
        r = _r10_res(lambda: torch.addcmul(y, y, w8, out=o), o)
        yield f"addcmul_0d8_out {str(odt)[6:]} {r}"
        o = torch.empty(0, dtype=odt, device=dev)
        b8 = t([1.0, 2.0], f8)
        r = _r10_res(lambda: a.log_sigmoid_backward(y, y, b8, grad_input=o), o)
        yield f"lsb_out {str(odt)[6:]} {r}"
        o = torch.empty(0, dtype=odt, device=dev)
        r = _r10_res(lambda: a.sigmoid_backward(y, w8, grad_input=o), o)
        yield f"sigb_out {str(odt)[6:]} {r}"
    for op in ("add", "mul", "maximum", "atan2", "lt", "logical_and", "copysign"):
        for tag, other in (("alias0d", torch.float32), ("alias0d_f8f8", f8)):
            b = t([0.5, 7.0], f8)
            x = b[0]
            yy = t([1.0, 2.0], other)
            r = _r10_res(lambda: getattr(torch, op)(x, yy, out=x), x, (b,))
            yield f"{tag} {op} {r}"
    b = t([0.5, 7.0], f8)
    x = b[0]
    yield "alias0d_neg_view " + _r10_res(lambda: torch.neg(b, out=x), x, (b,))
    b = t([0.5, 7.0, 1.0, 2.0], f8)
    x = b[:2]
    yield "alias_view_resize " + _r10_res(lambda: torch.add(b, b, out=x), x, (b,))


def _backward_lines(dev):
    """Every elementwise backward op: operand dtype pairs and out dtypes."""
    f8, a = torch.float8_e4m3fn, torch.ops.aten
    k = {
        "f8_0d": lambda: torch.tensor(0.5).to(f8).to(dev),
        "f8": lambda: torch.tensor([0.5, 1.0]).to(f8).to(dev),
        "f32": lambda: torch.tensor([1.0, 2.0], device=dev),
        "f32_0d": lambda: torch.tensor(0.5, device=dev),
    }
    ops = {
        "threshold_backward": (
            lambda x, y: a.threshold_backward(x, y, 0.5),
            lambda x, y, o: a.threshold_backward.grad_input(x, y, 0.5, grad_input=o),
        ),
        "sigmoid_backward": (
            a.sigmoid_backward,
            lambda x, y, o: a.sigmoid_backward.grad_input(x, y, grad_input=o),
        ),
        "tanh_backward": (
            a.tanh_backward,
            lambda x, y, o: a.tanh_backward.grad_input(x, y, grad_input=o),
        ),
        "elu_backward": (
            lambda x, y: a.elu_backward(x, 1.0, 1.0, 1.0, False, y),
            lambda x, y, o: a.elu_backward.grad_input(
                x, 1.0, 1.0, 1.0, False, y, grad_input=o
            ),
        ),
        "hardtanh_backward": (
            lambda x, y: a.hardtanh_backward(x, y, -1.0, 1.0),
            lambda x, y, o: a.hardtanh_backward.grad_input(
                x, y, -1.0, 1.0, grad_input=o
            ),
        ),
        "leaky_relu_backward": (
            lambda x, y: a.leaky_relu_backward(x, y, 0.01, False),
            lambda x, y, o: a.leaky_relu_backward.grad_input(
                x, y, 0.01, False, grad_input=o
            ),
        ),
        "softplus_backward": (
            lambda x, y: a.softplus_backward(x, y, 1.0, 20.0),
            lambda x, y, o: a.softplus_backward.grad_input(
                x, y, 1.0, 20.0, grad_input=o
            ),
        ),
        "mish_backward": (a.mish_backward, None),
        "hardswish_backward": (
            a.hardswish_backward,
            lambda x, y, o: a.hardswish_backward.out(x, y, out=o),
        ),
        "hardsigmoid_backward": (
            a.hardsigmoid_backward,
            lambda x, y, o: a.hardsigmoid_backward.grad_input(x, y, grad_input=o),
        ),
        "silu_backward": (
            a.silu_backward,
            lambda x, y, o: a.silu_backward.grad_input(x, y, grad_input=o),
        ),
        "gelu_backward": (
            a.gelu_backward,
            lambda x, y, o: a.gelu_backward.grad_input(x, y, grad_input=o),
        ),
        "logit_backward": (
            a.logit_backward,
            lambda x, y, o: a.logit_backward.grad_input(x, y, grad_input=o),
        ),
        "log_sigmoid_backward": (
            lambda x, y: a.log_sigmoid_backward(x, y, y),
            lambda x, y, o: a.log_sigmoid_backward.grad_input(x, y, y, grad_input=o),
        ),
    }
    fn_combos = [
        ("f8_0d", "f32"), ("f32", "f8_0d"), ("f8", "f32"), ("f32", "f8"),
        ("f8", "f8"), ("f8_0d", "f8_0d"), ("f8_0d", "f32_0d"),
    ]  # fmt: skip
    for name, (f, fo) in ops.items():
        for c in fn_combos:
            r = _r10_res(lambda: f(k[c[0]](), k[c[1]]()))
            yield f"{name} fn {'/'.join(c)} {r}"
        if fo is None:
            continue
        for c in (("f8", "f8"), ("f32", "f8_0d"), ("f32", "f32")):
            for odt in (f8, torch.float16, torch.float32, torch.int32):
                o = torch.empty(0, dtype=odt, device=dev)
                r = _r10_res(lambda: fo(k[c[0]](), k[c[1]](), o), o)
                yield f"{name} out {'/'.join(c)} {str(odt)[6:]} {r}"


def test_float8_iterator_rules_match_stock_cuda(mojo_device):
    _compare(list(_r10_lines(mojo_device)), _F8_ITER["rules_and_aliasing"])
    _compare(list(_backward_lines(mojo_device)), _F8_ITER["backward_ops"])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="regenerating needs stock CUDA torch"
)
def test_float8_cuda_data_regenerates_on_stock_cuda():
    """The JSON files are exactly what these generators print on stock CUDA
    (torch.cuda), so the comparisons above test against the real thing."""
    for op, want in _F8_MATRIX.items():
        gen = _mx_unary if op in _MATRIX_UNARY else _mx_binary
        assert list(gen(op, "cuda")) == want
    assert list(_r10_lines("cuda")) == _F8_ITER["rules_and_aliasing"]
    assert list(_backward_lines("cuda")) == _F8_ITER["backward_ops"]


def _r11_lines(dev):
    """Out-dtype dispatch converting every operand, the shrink backward ops,
    a CPU out=, the special polynomials and ldexp's own sequence."""
    f8, a = torch.float8_e4m3fn, torch.ops.aten

    def run(name, fn, o):
        r = _r10_res(fn).replace("cuda:0", "DEV").replace("mojo:0", "DEV")
        return f"{name} {r} {tuple(o.shape)}"

    g = torch.tensor(0.5).to(f8).to(dev)
    xx = torch.tensor([0.9999], device=dev)
    for odt in (torch.float16, torch.float32):
        o = torch.empty(0, dtype=odt, device=dev)
        yield run(
            f"hardtanh_b_{odt}",
            lambda: a.hardtanh_backward.grad_input(g, xx, -1.0, 1.0, grad_input=o),
            o,
        )
        o = torch.empty(0, dtype=odt, device=dev)
        y3 = torch.tensor([0.33333], device=dev)
        yield run(
            f"sigmoid_b_{odt}",
            lambda: a.sigmoid_backward.grad_input(g, y3, grad_input=o),
            o,
        )
    b = torch.tensor([1.0, 2.0], device=dev)
    x = torch.tensor([2.0, 3.0], device=dev)
    for name in ("hardshrink_backward", "softshrink_backward"):
        f = getattr(a, name)
        for odt in (f8, torch.float16):
            o = torch.empty(0, dtype=odt, device=dev)
            yield run(f"{name}_{odt}", lambda: f.grad_input(b, x, 0.5, grad_input=o), o)
        yield run(f"{name}_f8", lambda: f(b.to(f8), x.to(f8), 0.5), b)
        g8 = torch.tensor(0.5, device=dev).to(f8)
        yield run(f"{name}_f8_0d", lambda: f(g8, x, 0.5), b)
    ones = torch.ones(2, device=dev)
    for odt in (f8, torch.int32, torch.float32):
        o = torch.empty(0, dtype=odt)
        yield run(
            f"add_f8_cpuout_{odt}",
            lambda: torch.add(ones.to(f8), ones.to(f8), out=o),
            o,
        )
        o = torch.empty(0, dtype=odt)
        yield run(f"neg_f8_cpuout_{odt}", lambda: torch.neg(ones.to(f8), out=o), o)
    o = torch.empty(0, dtype=f8)
    yield run("add_f32_cpuout_f8", lambda: torch.add(ones, ones, out=o), o)
    o = torch.empty(0, dtype=f8, device=dev)
    yield run(
        "bessel_k0_out",
        lambda: torch.special.scaled_modified_bessel_k0(ones.to(f8), out=o),
        o,
    )
    o = torch.empty(0, dtype=f8, device=dev)
    yield run(
        "shifted_cheb_t_out",
        lambda: torch.special.shifted_chebyshev_polynomial_t(
            ones.to(f8), ones.to(f8), out=o
        ),
        o,
    )
    f8v = ones.to(f8)
    for odt in (torch.float32, torch.int32, f8):
        o = torch.empty(0, dtype=odt, device=dev)
        yield run(f"ldexp_f32_f8_out_{odt}", lambda: torch.ldexp(ones, f8v, out=o), o)
        o = torch.empty(0, dtype=odt, device=dev)
        yield run(f"ldexp_f8_f8_out_{odt}", lambda: torch.ldexp(f8v, f8v, out=o), o)
    yield run("ldexp_inplace_f32_f8", lambda: ones.clone().ldexp_(f8v), ones)
    x8 = torch.ones(2, 1).to(f8).to(dev)
    y13 = torch.ones(1, 3, device=dev)
    yield run("ldexp_inplace_shape", lambda: x8.clone().ldexp_(y13), x8)
    yield run("ldexp_f8_f32", lambda: torch.ldexp(f8v, ones), ones)
    z8 = torch.tensor(1.0).to(f8).to(dev)
    yield run("ldexp_f8_0d_f32", lambda: torch.ldexp(z8, ones), ones)
    z32 = torch.tensor(1.0, device=dev)
    yield run("ldexp_f8_f32_0d", lambda: torch.ldexp(f8v, z32), ones)


def test_float8_round11_shapes_match_stock_cuda(mojo_device):
    _compare(list(_r11_lines(mojo_device)), _F8_ITER["round11"])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="regenerating needs stock CUDA torch"
)
def test_float8_round11_data_regenerates_on_stock_cuda():
    assert list(_r11_lines("cuda")) == _F8_ITER["round11"]


def _act_res(fn, out=None):
    try:
        s = "OK " + _mx_enc(fn())
    except Exception as e:  # noqa: BLE001 -- the outcome itself is compared
        msg = str(e).split("\n")[0].split(" [aten")[0]
        s = type(e).__name__ + " " + _dev_free(msg)[:100]
    if out is not None:
        s += f" | out={str(out.dtype)[6:]}{tuple(out.shape)}"
    return s


def _dev_free(s):
    return s.replace("cuda:0", "DEV").replace("mojo:0", "DEV")


def _activation_lines(dev):
    """The forward activations (functional, out= of every dtype, in-place,
    a CPU out=), and broadcast errors before the dispatch."""
    f8, a = torch.float8_e4m3fn, torch.ops.aten

    def x32():
        return torch.tensor([-1.0, 0.0, 1.0, 2.0], device=dev)

    def x8():
        return torch.tensor([-1.0, 0.0, 1.0, 2.0]).to(f8).to(dev)

    ops = {
        "threshold": (
            lambda x: a.threshold(x, 0.5, -2.0),
            lambda x, o: a.threshold.out(x, 0.5, -2.0, out=o),
            lambda x: a.threshold_(x, 0.5, -2.0),
        ),
        "softshrink": (
            lambda x: a.softshrink(x, 0.5),
            lambda x, o: a.softshrink.out(x, 0.5, out=o),
            None,
        ),
        "hardshrink": (
            lambda x: a.hardshrink(x, 0.5),
            lambda x, o: a.hardshrink.out(x, 0.5, out=o),
            None,
        ),
        "hardtanh": (a.hardtanh, lambda x, o: a.hardtanh.out(x, out=o), a.hardtanh_),
        "elu": (a.elu, lambda x, o: a.elu.out(x, out=o), a.elu_),
        "leaky_relu": (
            a.leaky_relu,
            lambda x, o: a.leaky_relu.out(x, out=o),
            a.leaky_relu_,
        ),
        "gelu": (a.gelu, lambda x, o: a.gelu.out(x, out=o), a.gelu_),
        "silu": (a.silu, lambda x, o: a.silu.out(x, out=o), a.silu_),
        "mish": (a.mish, lambda x, o: a.mish.out(x, out=o), a.mish_),
        "hardsigmoid": (
            a.hardsigmoid,
            lambda x, o: a.hardsigmoid.out(x, out=o),
            a.hardsigmoid_,
        ),
        "hardswish": (
            a.hardswish,
            lambda x, o: a.hardswish.out(x, out=o),
            a.hardswish_,
        ),
        "softplus": (a.softplus, lambda x, o: a.softplus.out(x, out=o), None),
        "log_sigmoid": (a.log_sigmoid, lambda x, o: a.log_sigmoid.out(x, out=o), None),
        "relu6": (a.relu6, None, a.relu6_),
        "celu": (a.celu, None, a.celu_),
        "selu": (a.selu, None, a.selu_),
        "relu": (a.relu, lambda x, o: a.relu.out(x, out=o), a.relu_),
        "glu": (a.glu, lambda x, o: a.glu.out(x, out=o), None),
    }
    for name, (f, fo, fi) in ops.items():
        yield f"{name} fn f8 " + _act_res(lambda: f(x8()))
        if fi is not None:
            yield f"{name} inplace f8 " + _act_res(lambda: fi(x8()))
        if fo is None:
            continue
        for src, mk in (("f8", x8), ("f32", x32)):
            for odt in (f8, torch.float16, torch.float32, torch.int32):
                o = torch.empty(0, dtype=odt, device=dev)
                yield f"{name} out {src} {str(odt)[6:]} " + _act_res(
                    lambda: fo(mk(), o), o
                )
        o = torch.empty(0, dtype=f8)
        yield f"{name} out cpu f8 " + _act_res(lambda: fo(x8(), o), o)
    for sa, sb in (((2,), (3,)), ((2, 3), (3, 2))):
        for name in ("add", "mul", "lt", "maximum", "atan2"):
            f = getattr(torch, name)

            def ta():
                return torch.ones(sa).to(f8).to(dev)

            def tb():
                return torch.ones(sb).to(f8).to(dev)

            yield f"{name} bcast {sa} {sb} " + _act_res(lambda: f(ta(), tb()))
            o = torch.empty(0, dtype=f8, device=dev)
            yield f"{name} bcast_out {sa} {sb} " + _act_res(
                lambda: f(ta(), tb(), out=o), o
            )


def _r12_lines(dev):
    """ldexp's integral-exponent iterator, TensorIterator's device order and
    its CPU scalars."""
    f8, e5, a = torch.float8_e4m3fn, torch.float8_e5m2, torch.ops.aten

    def run(name, fn, o=None):
        r = _act_res(fn)
        return f"{name} {r} {tuple(o.shape) if o is not None else ''}"

    x8 = torch.ones(2).to(f8).to(dev)
    for idt in (torch.int32, torch.int64, torch.bool):
        e = torch.ones(2, dtype=idt, device=dev)
        yield run(f"ldexp_f8_{idt}", lambda: torch.ldexp(x8, e))
        for odt in (f8, torch.float32):
            o = torch.empty(0, dtype=odt, device=dev)
            yield run(f"ldexp_f8_{idt}_out_{odt}", lambda: torch.ldexp(x8, e, out=o), o)
    i64 = torch.ones(2, dtype=torch.int64, device=dev)
    yield run("ldexp_inplace_i64", lambda: x8.clone().ldexp_(i64))
    e5v = torch.ones(2).to(e5).to(dev)
    i32 = torch.ones(2, dtype=torch.int32, device=dev)
    yield run("ldexp_e5_i32", lambda: torch.ldexp(e5v, i32))
    cpu2 = torch.ones(2).to(f8)
    for odt in (torch.int64, f8):
        o = torch.empty(0, dtype=odt, device=dev)
        yield run(f"add_cpu2_dev_out_{odt}", lambda: torch.add(cpu2, x8, out=o), o)
    o = torch.empty(0, dtype=f8)
    yield run("add_dev_dev_out_cpu", lambda: torch.add(x8, x8, out=o), o)
    c0 = torch.tensor(1.0).to(f8)
    for name in ("neg", "exp"):
        o = torch.empty(0, dtype=f8, device=dev)
        yield run(f"{name}_cpu0d_out_dev", lambda: getattr(torch, name)(c0, out=o), o)
    yield run("hardtanh_b_cpu0d", lambda: a.hardtanh_backward(c0, x8, -1.0, 1.0))
    yield run("softshrink_b_cpu0d", lambda: a.softshrink_backward(c0, x8, 0.5))
    for name in ("eq", "lt"):
        o = torch.empty(0, dtype=torch.bool, device=dev)
        yield run(
            f"{name}_two_cpu0d_out", lambda: getattr(torch, name)(c0, c0, out=o), o
        )
    o = torch.empty(0, dtype=f8, device=dev)
    yield run("add_two_cpu0d_out", lambda: torch.add(c0, c0, out=o), o)
    yield run("logaddexp_cpu0d", lambda: torch.logaddexp(x8, c0))
    yield run("add_cpu0d", lambda: torch.add(x8, c0))


def test_float8_round12_shapes_match_stock_cuda(mojo_device):
    _compare(list(_activation_lines(mojo_device)), _F8_ITER["activations"])
    _compare(list(_r12_lines(mojo_device)), _F8_ITER["round12"])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="regenerating needs stock CUDA torch"
)
def test_float8_round12_data_regenerates_on_stock_cuda():
    assert list(_activation_lines("cuda")) == _F8_ITER["activations"]
    assert list(_r12_lines("cuda")) == _F8_ITER["round12"]


def _r13_lines(dev):
    """Per-op iterator flags and pre-checks (v2.14 source): ldexp's
    integral-exponent iterator, relu's bool check, threshold without
    overlap checks (self added twice), softshrink_check's dispatch."""
    f8, a = torch.float8_e4m3fn, torch.ops.aten

    def run(name, fn, o=None):
        return f"{name} {_act_res(fn)} | {tuple(o.shape) if o is not None else ''}"

    x8 = torch.ones(2).to(f8).to(dev)
    x1 = torch.ones(1).to(f8).to(dev)
    i3 = torch.ones(3, dtype=torch.int32, device=dev)
    o = torch.empty(0, dtype=f8, device=dev)
    yield run("ldexp_f8_i32_bcast_out", lambda: torch.ldexp(x8, i3, out=o), o)
    yield run("ldexp_f8_i32_bcast", lambda: torch.ldexp(x8, i3))
    yield run(
        "ldexp_cpu_exp", lambda: torch.ldexp(x8, torch.ones(2, dtype=torch.int32))
    )
    yield run(
        "ldexp_cpu0d_exp", lambda: torch.ldexp(x8, torch.tensor(1, dtype=torch.int32))
    )
    yield run("ldexp_inplace_bcast", lambda: x1.clone().ldexp_(i3), x1)
    o = torch.empty(0, dtype=f8, device=dev)
    yield run("ldexp_out_bcast_ok", lambda: torch.ldexp(x1, i3, out=o), o)
    b4 = torch.ones(4, dtype=torch.bool, device=dev)
    for odt in (f8, torch.float32):
        o = torch.empty(0, dtype=odt, device=dev)
        yield run(f"relu_out_bool_{odt}", lambda: a.relu.out(b4, out=o), o)
    yield run("relu_bool", lambda: a.relu(b4))
    ones8 = torch.ones(4).to(f8).to(dev)
    o = torch.empty(1, dtype=torch.float32, device=dev).expand(4)
    yield run("threshold_out_expanded", lambda: a.threshold.out(ones8, 2, -2, out=o), o)
    o = torch.empty(1, dtype=f8, device=dev).expand(4)
    yield run(
        "threshold_out_expanded_f8", lambda: a.threshold.out(ones8, 2, -2, out=o), o
    )
    b = torch.ones(4).to(f8).to(dev)
    yield run(
        "threshold_out_partial", lambda: a.threshold.out(b[:2], 2, -2, out=b[1:3]), b
    )
    o = torch.empty(1, device=dev).expand(4)
    yield run(
        "threshold_b_out_expanded",
        lambda: a.threshold_backward.grad_input(ones8, ones8, 2, grad_input=o),
        o,
    )
    o = torch.empty(0, dtype=torch.float32, device=dev)
    c0 = torch.tensor(1.0).to(f8)
    yield run("threshold_out_cpu0d_self", lambda: a.threshold.out(c0, 2, -2, out=o), o)
    for idt in (torch.int32, torch.bool, torch.float16):
        o = torch.empty(0, dtype=f8, device=dev)
        t = torch.ones(4, dtype=idt, device=dev)
        yield run(f"softshrink_out_{idt}", lambda: a.softshrink.out(t, 0.5, out=o), o)


def test_float8_round13_shapes_match_stock_cuda(mojo_device):
    _compare(list(_r13_lines(mojo_device)), _F8_ITER["round13"])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="regenerating needs stock CUDA torch"
)
def test_float8_round13_data_regenerates_on_stock_cuda():
    assert list(_r13_lines("cuda")) == _F8_ITER["round13"]


def _r14_lines(dev):
    """threshold's iterator config (inputs promoted, kernel dispatched on the
    out dtype, no overlap check, safe casting) for every out= shape;
    softshrink_check's lambda range; ldexp's 2.14 fast-path condition."""
    f8, a = torch.float8_e4m3fn, torch.ops.aten

    def run(name, fn, o=None):
        return f"{name} {_act_res(fn)} | {tuple(o.shape) if o is not None else ''}"

    x = torch.full((4,), 1.0001, device=dev)
    ones = torch.ones(4, device=dev)
    for odt in (torch.float16, torch.bool, torch.float32, f8, torch.int32):
        o = torch.empty(1, dtype=odt, device=dev).expand(4)
        yield run(f"thr_exp_{odt}", lambda: a.threshold.out(x, 1.00005, 0.5, out=o), o)
        o = torch.empty(4, dtype=odt, device=dev)
        yield run(
            f"thr_plain_{odt}", lambda: a.threshold.out(x, 1.00005, 0.5, out=o), o
        )
        o = torch.empty(1, dtype=odt, device=dev).expand(4)
        yield run(
            f"thrb_exp_{odt}",
            lambda: a.threshold_backward.grad_input(ones, x, 1.00005, grad_input=o),
            o,
        )
        o = torch.empty(1, dtype=odt, device=dev).expand(4)
        yield run(f"thr_ones_exp_{odt}", lambda: a.threshold.out(ones, 2, -2, out=o), o)
    for lam in (-1.0, float("nan"), 1e39, 0.5):
        for odt in (f8, torch.float32):
            o = torch.empty(0, dtype=odt, device=dev)
            yield run(
                f"softshrink_f32_lam{lam}_{odt}",
                lambda: a.softshrink.out(ones, lam, out=o),
                o,
            )
        o = torch.empty(0, dtype=f8, device=dev)
        h = torch.ones(4, device=dev, dtype=torch.float16)
        yield run(
            f"softshrink_f16_lam{lam}", lambda: a.softshrink.out(h, lam, out=o), o
        )
        yield run(f"softshrink_fn_f32_lam{lam}", lambda: a.softshrink(ones, lam))
    x8 = torch.ones(2).to(f8).to(dev)
    i32 = torch.ones(2, dtype=torch.int32, device=dev)
    for odt in (torch.float32, f8, torch.float16):
        o = torch.empty(0, dtype=odt, device=dev)
        yield run(f"ldexp_f8_i32_out_{odt}", lambda: torch.ldexp(x8, i32, out=o), o)
    e5 = torch.ones(2).to(torch.float8_e5m2).to(dev)
    o = torch.empty(0, dtype=f8, device=dev)
    yield run("ldexp_e5_i32_out_e4", lambda: torch.ldexp(e5, i32, out=o), o)
    ar = torch.arange(4.0, device=dev)
    o = torch.empty(1, device=dev).expand(4)
    yield run(
        "thr_cpu_self_exp_out",
        lambda: a.threshold.out(torch.arange(4.0), 1.5, -1, out=o),
        o,
    )
    o = torch.empty(1, device=dev).expand(4)
    yield run(
        "thrb_cpu_grad_exp_out",
        lambda: a.threshold_backward.grad_input(
            torch.arange(4.0), ar, 1.5, grad_input=o
        ),
        o,
    )
    b = torch.arange(4.0, device=dev)
    yield run("thr_partial", lambda: a.threshold.out(b[:3], 1.5, -1, out=b[1:]), b)
    e = torch.zeros(1, device=dev).expand(4)
    yield run("thr_inplace_expanded", lambda: a.threshold_(e, 1.5, -1), e)
    o = torch.empty(1, device=dev).expand(3)
    yield run("thr_exp_wrong_shape", lambda: a.threshold.out(ar, 1.5, -1, out=o), o)
    o = torch.empty(0, device=dev)
    yield run(
        "thr_cpu0d_f32_self",
        lambda: a.threshold.out(torch.tensor(1.0), 1.5, -1, out=o),
        o,
    )
    o = torch.empty(0, device=dev)
    i4 = torch.ones(4, dtype=torch.int32, device=dev)
    yield run("softshrink_i32_f32out", lambda: a.softshrink.out(i4, 0.5, out=o), o)
    yield run("softshrink_i32_fn", lambda: a.softshrink(i4, 0.5))


def test_float8_round14_shapes_match_stock_cuda(mojo_device):
    _compare(list(_r14_lines(mojo_device)), _F8_ITER["round14"])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="regenerating needs stock CUDA torch"
)
def test_float8_round14_data_regenerates_on_stock_cuda():
    assert list(_r14_lines("cuda")) == _F8_ITER["round14"]


def _threshold_value_lines(dev):
    """threshold / threshold_backward values: inf, nan and -0 in grad and
    self, float8 / float16 / float32 operands, broadcast shapes, and every
    kind of out= (none, fresh, empty, expanded, aliased, another dtype)."""
    a = torch.ops.aten
    f32, f16 = torch.float32, torch.float16
    e4, e5 = torch.float8_e4m3fn, torch.float8_e5m2
    gv = [float("inf"), float("nan"), -0.0, 2.0]
    sv = [0.0, -0.0, float("nan"), 1.0]

    def t(vals, dt):
        return torch.tensor(vals).to(dt).to(dev)

    def outs(n):
        yield "none", None
        yield "fresh_f32", lambda: torch.empty(n, device=dev)
        yield "empty_f32", lambda: torch.empty(0, device=dev)
        yield "expanded_f32", lambda: torch.empty(1, device=dev).expand(n)
        yield "empty_f16", lambda: torch.empty(0, dtype=f16, device=dev)

    for thr in (0.0, 0.5):
        for gdt, sdt in (
            (f32, f32),
            (e5, f32),
            (f32, e5),
            (e5, e5),
            (f16, f32),
            (e4, f16),
        ):
            for gshape, sshape in (((4,), (4,)), ((4,), (1,)), ((1,), (4,))):

                def grad():
                    return t(gv[: gshape[0]] if gshape[0] == 4 else [gv[0]], gdt)

                def selfv():
                    return t(sv[: sshape[0]] if sshape[0] == 4 else [sv[2]], sdt)

                tag = f"{thr} {str(gdt)[6:]}/{str(sdt)[6:]} {gshape}/{sshape}"
                for oname, mk in outs(4):
                    if mk is None:
                        yield f"thrb {tag} {oname} " + _act_res(
                            lambda: a.threshold_backward(grad(), selfv(), thr)
                        )
                        yield f"thr {tag} {oname} " + _act_res(
                            lambda: a.threshold(selfv(), thr, -2.0)
                        )
                        continue
                    o = mk()
                    yield f"thrb {tag} {oname} " + _act_res(
                        lambda: a.threshold_backward.grad_input(
                            grad(), selfv(), thr, grad_input=o
                        ),
                        o,
                    )
                    o = mk()
                    yield f"thr {tag} {oname} " + _act_res(
                        lambda: a.threshold.out(selfv(), thr, -2.0, out=o), o
                    )
            # Aliased outs: grad_input = grad, threshold out = self.
            g = t(gv, gdt)
            yield f"thrb {thr} {str(gdt)[6:]}/{str(sdt)[6:]} aliased_grad " + _act_res(
                lambda: a.threshold_backward.grad_input(
                    g, t(sv, sdt), thr, grad_input=g
                ),
                g,
            )
            s = t(sv, sdt)
            yield f"thr {thr} {str(sdt)[6:]} aliased_self " + _act_res(
                lambda: a.threshold.out(s, thr, -2.0, out=s), s
            )
            s2 = t(sv, sdt)
            yield f"thr_ {thr} {str(sdt)[6:]} inplace " + _act_res(
                lambda: a.threshold_(s2, thr, -2.0)
            )


def test_float8_threshold_values_match_stock_cuda(mojo_device):
    _compare(list(_threshold_value_lines(mojo_device)), _F8_ITER["threshold_values"])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="regenerating needs stock CUDA torch"
)
def test_float8_threshold_values_regenerate_on_stock_cuda():
    assert list(_threshold_value_lines("cuda")) == _F8_ITER["threshold_values"]


def _threshold_scalar_lines(dev):
    """threshold's Scalars convert to the kernel's dtype with c10's overflow
    check (an infinity converts)."""
    a = torch.ops.aten
    x = torch.ones(4, device=dev)
    h = torch.float16

    def out0():
        return torch.empty(0, dtype=h, device=dev)

    yield _act_res(lambda: a.threshold.out(x, 0.5, 1e6, out=out0()))
    yield _act_res(lambda: a.threshold.out(x, 1e6, 0.5, out=out0()))
    yield _act_res(lambda: a.threshold(x.half(), 0.5, 1e6))
    yield _act_res(
        lambda: a.threshold.out(
            x, 0.5, 1e6, out=torch.empty(1, dtype=h, device=dev).expand(4)
        )
    )
    yield _act_res(lambda: a.threshold.out(x, 0.5, float("inf"), out=out0()))
    yield _act_res(lambda: a.threshold_(x.half(), 0.5, 1e6))
    e5 = torch.ones(4).to(torch.float8_e5m2).to(dev)
    yield _act_res(lambda: a.threshold.out(e5, 0.5, 1e6, out=out0()))
    # Every Scalar kind (int, float, bool) against every dispatched dtype.
    e5b = torch.tensor([0.0, 1.0]).to(torch.float8_e5m2).to(dev)
    yield _act_res(lambda: a.threshold.out(e5b, 0, 100000, out=out0()))
    dts = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
    dts += [h, torch.bfloat16, torch.float32, torch.float64]
    vals = [100000, -1, -300, 300, -128, 128, 2**40, 2**63 - 1, True, False] + [
        1e6,
        1e40,
        -1.0,
        2.5,
        255.0,
        256.0,
        float("inf"),
        float("nan"),
    ]
    for dt in dts:
        for v in vals:
            xd = torch.ones(2, dtype=dt, device=dev)
            yield _act_res(lambda xd=xd, v=v: a.threshold(xd, 0, v))
            yield _act_res(lambda xd=xd, v=v: a.threshold(xd, v, 0))
    # Integers past 2**53 travel exactly and round once to the computation
    # dtype: inputs straddling each threshold expose the converted value.
    bigs = [2**53 - 1, 2**53, 2**53 + 1, 2**53 + 3, -(2**53 + 1)]
    bigs += [2**63 - 1, -(2**63)]
    for dt in dts:
        for v in bigs:
            if dt.is_floating_point:
                b = torch.tensor(float(v), dtype=dt)
                xs = torch.stack(
                    [
                        torch.nextafter(b, torch.tensor(-float("inf"), dtype=dt)),
                        b,
                        torch.nextafter(b, torch.tensor(float("inf"), dtype=dt)),
                    ]
                )
            elif dt == torch.int64:
                lo, hi = max(v - 1, -(2**63)), min(v + 1, 2**63 - 1)
                xs = torch.tensor([lo, v, hi], dtype=dt)
            else:
                xs = torch.tensor([0, 1, 2], dtype=dt)
            xb = xs.to(dev)
            gf = torch.tensor([1.0, 2.0, 3.0], device=dev)
            yield _act_res(lambda xb=xb, v=v: a.threshold(xb, v, -5))
            yield _act_res(lambda xb=xb, v=v: a.threshold_backward(gf, xb, v))
            yield _act_res(lambda xb=xb, v=v: a.threshold_backward(xb, xb, v))
            o = torch.empty(0, dtype=dt, device=dev)
            yield _act_res(lambda xb=xb, v=v, o=o: a.threshold.out(xb, 0, v, out=o), o)
    # The overflow raises in the kernel, after the iterator resized out.
    for v in (1e6, 100000):
        o = out0()
        yield _act_res(lambda o=o, v=v: a.threshold.out(x.half(), 0, v, out=o), o)
        o = out0()
        yield _act_res(lambda o=o, v=v: a.threshold.out(x, 0, v, out=o), o)
    y = x.half()
    yield _act_res(lambda: a.threshold_(y, 0, 100000), y)
    # threshold_backward converts its threshold the same way.
    for v in (1e6, 100000, True, float("inf")):
        yield _act_res(lambda v=v: a.threshold_backward(x.half(), x.half(), v))
        o = out0()
        yield _act_res(
            lambda o=o, v=v: a.threshold_backward.grad_input(x, x, v, grad_input=o), o
        )
    i8 = torch.ones(2, dtype=torch.int8, device=dev)
    yield _act_res(lambda: a.threshold_backward(i8, i8, 300))
    # The dispatcher wrapper's device check (a CPU grad_input, a CPU 0-d
    # operand), then the kernel's refusal of bool.
    d = x[:2]
    c = torch.ones(2)
    c0 = torch.tensor(1.0)
    yield _act_res(lambda: a.threshold_backward.grad_input(d, d, 0, grad_input=c))
    yield _act_res(
        lambda: a.threshold_backward.grad_input(c, d, 0, grad_input=d.clone())
    )
    yield _act_res(
        lambda: a.threshold_backward.grad_input(d, c, 0, grad_input=d.clone())
    )
    yield _act_res(lambda: a.threshold_backward(c0, d, 0))
    yield _act_res(lambda: a.threshold_backward(d, c0, 0))
    bl = torch.ones(2, dtype=torch.bool, device=dev)
    yield _act_res(lambda: a.threshold_backward(bl, bl, 0))
    yield _act_res(lambda: a.threshold_backward(d, bl, 0))
    yield _act_res(
        lambda: a.threshold_backward.grad_input(bl, bl, 0, grad_input=bl.clone())
    )
    yield _act_res(lambda: a.threshold.out(bl, 0, 0, out=bl.clone()))
    # The broadcast error comes first, in the iterator's (self, grad) order.
    for gs, ss in (((2,), (3,)), ((3,), (2,)), ((2, 3), (4, 1))):
        o = torch.empty(0, device=dev)
        yield _act_res(
            lambda o=o, gs=gs, ss=ss: a.threshold_backward.grad_input(
                torch.ones(gs, device=dev), torch.ones(ss, device=dev), 0, grad_input=o
            ),
            o,
        )


def test_threshold_scalar_overflow_matches_stock_cuda(mojo_device):
    _compare(list(_threshold_scalar_lines(mojo_device)), _F8_ITER["threshold_scalars"])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="regenerating needs stock CUDA torch"
)
def test_threshold_scalar_data_regenerates_on_stock_cuda():
    assert list(_threshold_scalar_lines("cuda")) == _F8_ITER["threshold_scalars"]


# ---------------------------------------------------------------------------
# float8 meets complex (review of the complex-tensor merge)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cdt", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("dst", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_complex_to_float8_never_saturates(mojo_device, cdt, dst):
    """A complex source takes Copy.cu's default float8 branch (c10's
    conversion of the real part), not the float / half / bfloat16 SATFINITE
    one: [1000+2j, 3+1j, inf] -> e4m3fn is [nan, 3, nan] on stock CUDA
    (measured 2.11, H100; v2.14.0's float8_copy_kernel_cuda keys the
    intrinsic on kFloat / kHalf / kBFloat16 sources only)."""
    if cdt == torch.complex128:
        _skip_f64(mojo_device, cdt)
    host = torch.tensor([1000 + 2j, 3 + 1j, complex(float("inf"), 0.0)], dtype=cdt)
    want = host.real.to(dst)
    d = host.to(mojo_device)
    _assert_same(d.to(dst), want)
    out = torch.empty(3, dtype=dst, device=mojo_device)
    _assert_same(out.copy_(d), want)


@pytest.mark.parametrize("op", ["logical_and", "logical_or", "logical_xor"])
def test_float8_logical_with_complex_0d(mojo_device, op):
    """float8[3] against a complex 0-d tensor: combine_categories needs
    toComplexType(Float8_e4m3fn), a RuntimeError raised before `out` is
    resized (stock CUDA 2.11: out keeps its shape)."""
    x = torch.tensor([1.0, 0.0, 2.0]).to(torch.float8_e4m3fn).to(mojo_device)
    z = torch.tensor(1 + 1j, device=mojo_device)
    msg = "Unknown Complex ScalarType for Float8_e4m3fn"
    with pytest.raises(RuntimeError, match=msg):
        getattr(torch, op)(x, z)
    out = torch.empty(0, dtype=torch.bool, device=mojo_device)
    with pytest.raises(RuntimeError, match=msg):
        getattr(torch, op)(x, z, out=out)
    assert out.shape == (0,)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("n", [4, 0])
@pytest.mark.parametrize("value", [1 + 1j, 1000 + 0j, 2 + 0j])
def test_float8_masked_fill_complex_value_declined(mojo_device, dtype, n, value):
    """masked_fill_'s dtype dispatch refuses float8 before the (complex)
    value converts, empty or not (stock CUDA 2.11: NotImplementedError,
    never the value's overflow RuntimeError); index_put_ with a complex
    value names both dtypes."""
    x = torch.zeros(n).to(dtype).to(mojo_device)
    m = torch.ones(n, dtype=torch.bool, device=mojo_device)
    with pytest.raises(NotImplementedError, match='"masked_fill_" not implemented'):
        x.masked_fill_(m, value)
    with pytest.raises(NotImplementedError, match='"masked_fill_" not implemented'):
        x.masked_fill(m, torch.tensor(value, device=mojo_device))
    name = "Float8_" + str(dtype).removeprefix("torch.float8_")
    with pytest.raises(
        RuntimeError,
        match=f"got {name} for the destination and ComplexFloat for the source",
    ):
        x.index_put_((m,), torch.tensor(value, device=mojo_device))


@pytest.mark.parametrize("op", ["maximum", "minimum", "fmax", "fmin", "heaviside"])
@pytest.mark.parametrize("cshape", [(), (3,)], ids=["0d", "1d"])
def test_float8_vs_complex_meta_checks_first(mojo_device, op, cshape):
    """These ops refuse a complex operand in their meta function, before
    the iterator promotes (stock CUDA 2.11), out untouched."""
    x = torch.tensor([1.0, 0.0, 2.0]).to(torch.float8_e4m3fn).to(mojo_device)
    z = torch.full(cshape, 1 + 1j, device=mojo_device)
    word = "is not yet implemented" if op == "heaviside" else "not implemented"
    msg = f"{op} {word} for complex tensors"
    for a, b in ((x, z), (z, x)):
        with pytest.raises(RuntimeError, match=msg):
            getattr(torch, op)(a, b)
        out = torch.empty(0, device=mojo_device)
        with pytest.raises(RuntimeError, match=msg):
            getattr(torch, op)(a, b, out=out)
        assert out.shape == (0,)


def test_float8_with_wrapped_complex_scalar(mojo_device):
    """A Python complex is a wrapped number of the default complex dtype in
    the result type: against float8 it needs toComplexType(float8), a
    RuntimeError on stock CUDA 2.11; a Python float still leaves float8 the
    computation dtype (the kernel's NotImplementedError)."""
    x = torch.tensor([1.0, 0.0, 2.0]).to(torch.float8_e4m3fn).to(mojo_device)
    msg = "Unknown Complex ScalarType for Float8_e4m3fn"
    z = torch.tensor(1j, device=mojo_device)
    for fn in (
        lambda: x + 1j,
        lambda: x * 2j,
        lambda: 2j * x,
        lambda: x.clone().add_(1j),
        lambda: x.clone().mul_(z),
    ):
        with pytest.raises(RuntimeError, match=msg):
            fn()
    with pytest.raises(NotImplementedError, match='"ufunc_add_CUDA" not implemented'):
        x + 2.5  # noqa: B018
    i = torch.ones(3, dtype=torch.int32, device=mojo_device)
    out = torch.empty(0, dtype=torch.float8_e4m3fn, device=mojo_device)
    assert torch.add(i, 2.5, out=out).float().cpu().tolist() == [3.5, 3.5, 3.5]
    out = torch.empty(0, dtype=torch.float8_e4m3fn, device=mojo_device)
    with pytest.raises(RuntimeError, match="ComplexFloat can't be cast"):
        torch.add(i, 1j, out=out)
    assert out.shape == (0,)


@pytest.mark.parametrize("op,kernel", [("mul", "mul_cuda"), ("add", "ufunc_add_CUDA")])
def test_float8_into_complex_out(mojo_device, op, kernel):
    """float8 operands with a complex out: canCast(float8, complex) holds,
    the out is resized, then the float8 kernel dispatch refuses."""
    x = torch.tensor([1.0, 0.0, 2.0]).to(torch.float8_e4m3fn).to(mojo_device)
    out = torch.empty(0, dtype=torch.complex64, device=mojo_device)
    with pytest.raises(NotImplementedError, match=f'"{kernel}" not implemented'):
        getattr(torch, op)(x, x, out=out)
    assert out.shape == (3,)


@pytest.mark.parametrize("src", ["float32", "int64", "complex64", "float8_e4m3fn"])
def test_logical_not_float8_out_declined(mojo_device, src):
    """logical_not_cuda dispatches on out's dtype first: a float8 out is
    refused after the resize, whatever the input (stock CUDA 2.11)."""
    x = torch.tensor([1.0, 0.0, 2.0]).to(getattr(torch, src)).to(mojo_device)
    out = torch.empty(0, dtype=torch.float8_e4m3fn, device=mojo_device)
    with pytest.raises(NotImplementedError, match='"logical_not_cuda" not implemented'):
        torch.logical_not(x, out=out)
    assert out.shape == (3,)
    if src != "float8_e4m3fn":
        b = torch.logical_not(
            x, out=torch.empty(0, dtype=torch.bool, device=mojo_device)
        )
        assert b.cpu().tolist() == [False, True, False]


def test_complex_with_float8_computes(mojo_device):
    """complex64[3] + float8 0-d computes in complex64 (the float8 operand
    converted exactly), and abs(complex, out=float8) copies the real result
    into out (stock CUDA 2.11 values)."""
    c = torch.tensor([1 + 1j, 3 + 4j, 0j], device=mojo_device)
    z8 = torch.tensor(2.0).to(torch.float8_e4m3fn).to(mojo_device)
    assert (c + z8).cpu().tolist() == [3 + 1j, 5 + 4j, 2 + 0j]
    assert (z8 * c).cpu().tolist() == [2 + 2j, 6 + 8j, 0j]
    out = torch.empty(0, dtype=torch.float8_e4m3fn, device=mojo_device)
    torch.abs(c, out=out)
    assert out.float().cpu().tolist() == [1.375, 5.0, 0.0]


def test_complex_abs_into_e4m3fn_out_is_a_copy(mojo_device):
    """abs(complex, out=e4m3fn) is unary_op_impl_with_complex_to_float_out:
    `result.copy_(at::real(complex_result))`, a float32 -> e4m3fn copy_,
    which v2.14 saturates on sm_89+ (c10's NaN elsewhere)."""
    c = torch.tensor([1000 + 0j, 460 + 0j, 470 + 0j], device=mojo_device)
    out = torch.empty(0, dtype=torch.float8_e4m3fn, device=mojo_device)
    torch.abs(c, out=out)
    want = _stock_cast(torch.tensor([1000.0, 460.0, 470.0]), out.dtype, mojo_device)
    _assert_same(out, want)


@pytest.mark.parametrize(
    "dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.uint16, torch.float32]
)
@pytest.mark.parametrize("value", [[1j], [1.0]], ids=["complex", "real"])
@pytest.mark.parametrize("on_cpu", [False, True], ids=["dev", "cpu"])
def test_masked_fill_value_rank_checked_first(mojo_device, dtype, value, on_cpu):
    """masked_fill__cuda's Tensor overload checks a 0-d value before the
    dtype dispatch (stock CUDA 2.11: RuntimeError, not the float8 / uint16
    NotImplementedError)."""
    x = torch.zeros(3).to(dtype).to(mojo_device)
    m = torch.ones(3, dtype=torch.bool, device=mojo_device)
    v = torch.tensor(value, device="cpu" if on_cpu else mojo_device)
    msg = "only supports a 0-dimensional value tensor, but got tensor with 1"
    with pytest.raises(RuntimeError, match=msg):
        x.masked_fill_(m, v)
    with pytest.raises(RuntimeError, match=msg):
        x.masked_fill(m, v)


@pytest.mark.parametrize(
    "op,exc,msg",
    [
        ("ceil", NotImplementedError, "ceil is not supported for complex inputs"),
        ("floor", NotImplementedError, "floor is not supported for complex inputs"),
        ("trunc", NotImplementedError, "trunc is not supported for complex inputs"),
        ("sign", NotImplementedError, "Unlike NumPy, torch.sign is not intended"),
        ("signbit", NotImplementedError, "signbit is not implemented for complex"),
        ("isposinf", RuntimeError, "isposinf does not support complex inputs"),
        ("isneginf", RuntimeError, "isneginf does not support complex inputs"),
    ],
)
def test_complex_self_float8_out_meta_refusal(mojo_device, op, exc, msg):
    """The op's meta refuses a complex self before the iterator checks a
    float8 out (stock CUDA 2.11), out untouched."""
    c = torch.tensor([1 + 1j, 2.5 + 0j, -3j], device=mojo_device)
    out = torch.empty(0, dtype=torch.float8_e4m3fn, device=mojo_device)
    with pytest.raises(exc, match=msg):
        getattr(torch, op)(c, out=out)
    assert out.shape == (0,)


_F8_CX_SCALAR = [
    lambda x: torch.pow(x, 2j),
    lambda x: torch.pow(2j, x),
    lambda x: x**2j,
    lambda x: 2j**x,
    lambda x: torch.remainder(x, 2j),
    lambda x: torch.remainder(2j, x),
    lambda x: x % 2j,
    lambda x: torch.fmod(x, 2j),
    lambda x: torch.copysign(x, 2j),
    lambda x: torch.bitwise_and(x, 2j),
    lambda x: torch.bitwise_or(x, 2j),
    lambda x: torch.bitwise_xor(x, 2j),
    lambda x: x & 2j,
]


@pytest.mark.parametrize("i", range(len(_F8_CX_SCALAR)))
def test_float8_complex_scalar_overloads(mojo_device, i):
    """The Scalar overloads wrap the complex number as the Tensor ones do:
    toComplexType(float8) fails (stock CUDA 2.11 RuntimeError)."""
    x = torch.tensor([1.0, 0.0, 2.0]).to(torch.float8_e4m3fn).to(mojo_device)
    with pytest.raises(
        RuntimeError, match="Unknown Complex ScalarType for Float8_e4m3fn"
    ):
        _F8_CX_SCALAR[i](x)


def test_float8_complex_reductions(mojo_device):
    """Reductions where float8 meets complex (stock CUDA 2.11): float8 is a
    floating dtype for mean's checks, a float8 accumulation dtype is the
    kernel's dtype dispatch error, and a float8 self accumulates in a
    complex out."""
    x8 = torch.tensor([1.0, 0.5, 2.0]).to(torch.float8_e4m3fn).to(mojo_device)
    c = torch.tensor([1 + 1j, 3 + 4j, -2j], device=mojo_device)
    out = torch.empty(0, dtype=torch.complex64, device=mojo_device)
    torch.mean(x8, 0, out=out)
    assert out.cpu().tolist() == pytest.approx(7 / 6 + 0j)
    for fn, kernel in (
        (lambda: torch.mean(c, dtype=torch.float8_e4m3fn), "mean_cuda"),
        (lambda: torch.sum(c, dtype=torch.float8_e4m3fn), "sum_cuda"),
        (lambda: torch.cumsum(c, 0, dtype=torch.float8_e4m3fn), "cumsum_cuda"),
    ):
        with pytest.raises(NotImplementedError, match=f'"{kernel}" not implemented'):
            fn()


@pytest.mark.parametrize(
    "fn",
    [lambda x, o: torch.pow(x, 2, out=o), lambda x, o: torch.square(x, out=o)],
    ids=["pow", "square"],
)
def test_float8_pow_scalar_into_complex_out(mojo_device, fn):
    """pow.Tensor_Scalar_out of a float8 base with a complex out: the
    iterator resizes out, then pow_cuda refuses float8 (stock CUDA 2.11)."""
    x8 = torch.tensor([1.0, 0.5, 2.0]).to(torch.float8_e4m3fn).to(mojo_device)
    out = torch.empty(0, dtype=torch.complex64, device=mojo_device)
    with pytest.raises(NotImplementedError, match='"pow_cuda" not implemented'):
        fn(x8, out)
    assert out.shape == (3,)


def test_float8_rsub_promotes_other_first(mojo_device):
    """rsub(self, other) is sub(other, self): the promotion error names
    other's dtype first (stock CUDA 2.11)."""
    x8 = torch.tensor([1.0, 0.5, 2.0]).to(torch.float8_e4m3fn).to(mojo_device)
    f = torch.ones(3, device=mojo_device)
    msg = "attempted to promote "
    with pytest.raises(RuntimeError, match=msg + "Float and Float8_e4m3fn"):
        torch.rsub(x8, f)
    with pytest.raises(RuntimeError, match=msg + "Float8_e4m3fn and Float"):
        torch.rsub(f, x8)
    c = torch.ones(3, dtype=torch.complex64, device=mojo_device)
    with pytest.raises(RuntimeError, match=msg + "ComplexFloat and Float8_e4m3fn"):
        torch.rsub(x8, c)


@pytest.mark.parametrize(
    "fn",
    [
        lambda x, o: torch.eq(x, x, out=o),
        lambda x, o: torch.ne(x, 2, out=o),
        lambda x, o: torch.pow(x, 2, out=o),
    ],
    ids=["eq", "ne_scalar", "pow"],
)
def test_float8_into_complex_out_overlap_checked_before_resize(mojo_device, fn):
    """TensorIterator checks the out's internal overlap before resizing it
    (stock CUDA 2.11: RuntimeError, out keeps its expanded shape)."""
    x8 = torch.tensor([1.0, 0.5, 2.0]).to(torch.float8_e4m3fn).to(mojo_device)
    out = torch.zeros(1, dtype=torch.complex64, device=mojo_device).expand(6)
    with pytest.raises(RuntimeError, match="more than one element of the written-to"):
        fn(x8, out)
    assert out.shape == (6,)


@pytest.mark.parametrize("odt", [torch.complex64, torch.float8_e4m3fn])
def test_float8_rsub_out_is_a_generated_wrapper(mojo_device, odt):
    """rsub's out= overloads compute the functional result first: its
    float8 refusal leaves out untouched (stock CUDA 2.11)."""
    x8 = torch.tensor([1.0, 0.5, 2.0]).to(torch.float8_e4m3fn).to(mojo_device)
    aten = torch.ops.aten
    for fn in (
        lambda o: aten.rsub.Scalar_out(x8, 2, out=o),
        lambda o: aten.rsub.Tensor_out(x8, x8, out=o),
    ):
        out = torch.empty(0, dtype=odt, device=mojo_device)
        with pytest.raises(
            NotImplementedError, match='"ufunc_add_CUDA" not implemented'
        ):
            fn(out)
        assert out.shape == (0,)


@pytest.mark.parametrize("odt", [torch.complex64, torch.float8_e4m3fn, torch.float32])
def test_float8_pow_complex_scalar_base_out(mojo_device, odt):
    """pow.Scalar_out takes its dtype from out (no result type): the out is
    resized, a base equal to 1 fills ones, any other complex base meets the
    float8 exponent in pow.Tensor_Tensor (stock CUDA 2.11)."""
    x8 = torch.tensor([1.0, 0.5, 2.0]).to(torch.float8_e4m3fn).to(mojo_device)
    out = torch.empty(0, dtype=odt, device=mojo_device)
    torch.pow(1 + 0j, x8, out=out)
    assert out.shape == (3,)
    vals = out.cpu().tolist() if odt.is_complex else out.float().cpu().tolist()
    assert vals == [1, 1, 1]
    out = torch.empty(0, dtype=odt, device=mojo_device)
    with pytest.raises(
        RuntimeError, match="Unknown Complex ScalarType for Float8_e4m3fn"
    ):
        torch.pow(2j, x8, out=out)
    assert out.shape == (3,)
    with pytest.raises(
        RuntimeError, match="Unknown Complex ScalarType for Float8_e4m3fn"
    ):
        torch.pow(1 + 0j, x8)


@pytest.mark.parametrize("odt", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fft_into_float8_out(mojo_device, odt):
    """A real transform into a float8 out: castable, stored by mul_out's
    TensorIterator with c10's conversion (never saturating: 2000 / 4 is
    NaN in e4m3fn), as stock CUDA 2.11 does."""
    for vals in ([1 + 1j, 0.5 - 1j, 2 + 0j], [2000 + 0j, 0j, 0j]):
        c = torch.tensor(vals)
        for name, fn in (
            ("irfft", lambda t, o: torch.fft.irfft(t, out=o)),
            ("hfft", lambda t, o: torch.fft.hfft(t, out=o)),
            ("c2r", lambda t, o: torch.ops.aten._fft_c2r.out(t, [0], 2, 4, out=o)),
        ):
            want = fn(c, torch.empty(0)).to(odt)
            out = torch.empty(0, dtype=odt, device=mojo_device)
            fn(c.to(mojo_device), out)
            torch.testing.assert_close(
                out.float().cpu(), want.float(), equal_nan=True, msg=name
            )


def test_complex_meta_refusals_without_float8(mojo_device):
    """A complex operand meets these ops' own checks before anything else,
    float8 or not (stock CUDA 2.11 error kinds and messages)."""
    c = torch.tensor([1 + 1j, 0.5 - 1j, 2 + 0j], device=mojo_device)
    x8 = torch.tensor([1.0, 0.5, 2.0]).to(torch.float8_e4m3fn).to(mojo_device)
    for fn, exc, msg in (
        (lambda: torch.isposinf(c), RuntimeError, "isposinf does not support complex"),
        (lambda: torch.isneginf(c), RuntimeError, "isneginf does not support complex"),
        (lambda: torch.signbit(c), NotImplementedError, "signbit is not implemented"),
        (lambda: torch.ceil(c), NotImplementedError, "ceil is not supported for"),
        (lambda: torch.maximum(c, c), RuntimeError, "maximum not implemented for"),
        (
            lambda: torch.dot(c, x8),
            RuntimeError,
            "same dtype, but found ComplexFloat and Float8_e4m3fn",
        ),
        (
            lambda: torch.vdot(x8, c),
            RuntimeError,
            "same dtype, but found Float8_e4m3fn and ComplexFloat",
        ),
        (
            lambda: torch.ops.aten._fft_r2c(x8, [0], 0, True),
            RuntimeError,
            "Unknown Complex ScalarType for Float8_e4m3fn",
        ),
    ):
        with pytest.raises(exc, match=msg):
            fn()


# ---------------------------------------------------------------------------
# review round 4: rsub's generated out= wrappers, alpha_check, pow.Scalar_out
# ---------------------------------------------------------------------------


def _rsub_pow_alpha_lines(dev):
    """rsub.Scalar_out / rsub.Tensor_out over other x alpha x out dtype x
    out shape (empty, right-sized, an expanded one-element tensor), the
    functional rsub, add / sub's alpha_check against float8, and
    pow.Scalar_out's resize / base-equal-to-1 fill (stock CUDA 2.11)."""
    f8, a = torch.float8_e4m3fn, torch.ops.aten
    selves = {
        "x8": lambda: torch.tensor([1.0, 0.5, 2.0]).to(f8).to(dev),
        "f32": lambda: torch.tensor([1.0, 0.5, 2.0], device=dev),
    }
    scalars = {"2": 2, "True": True, "2j": 2j, "2.5": 2.5}
    tensors = {
        "x8": lambda: torch.tensor([1.0, 0.5, 2.0]).to(f8).to(dev),
        "f32": lambda: torch.tensor([3.0, 1.0, 0.0], device=dev),
        "bool": lambda: torch.tensor([True, False, True], device=dev),
        "c64": lambda: torch.tensor([1 + 1j, 2j, 3 + 0j], device=dev),
    }
    alphas = {"1": 1, "True": True, "2j": 2j}
    odts = {"f8": f8, "c64": torch.complex64, "f32": torch.float32}

    def out_of(dt, shape):
        if shape == "exp":
            return torch.zeros(1, dtype=dt, device=dev).expand(6)
        return torch.zeros(int(shape), dtype=dt, device=dev)

    for s, al in itertools.product(selves, alphas):
        for o in scalars:
            yield f"rsub {s} S{o} a={al} " + _act_res(
                lambda s=s, o=o, al=al: torch.rsub(
                    selves[s](), scalars[o], alpha=alphas[al]
                )
            )
            for od, shp in itertools.product(odts, ("0", "3", "exp")):
                out = out_of(odts[od], shp)
                yield f"rsub.Scalar_out {s} {o} a={al} {od}{shp} " + _act_res(
                    lambda s=s, o=o, al=al, out=out: a.rsub.Scalar_out(
                        selves[s](), scalars[o], alphas[al], out=out
                    ),
                    out,
                )
        for o in tensors:
            yield f"rsub {s} T{o} a={al} " + _act_res(
                lambda s=s, o=o, al=al: torch.rsub(
                    selves[s](), tensors[o](), alpha=alphas[al]
                )
            )
            for od, shp in itertools.product(odts, ("0", "3", "exp")):
                out = out_of(odts[od], shp)
                yield f"rsub.Tensor_out {s} {o} a={al} {od}{shp} " + _act_res(
                    lambda s=s, o=o, al=al, out=out: a.rsub.Tensor_out(
                        selves[s](), tensors[o](), alpha=alphas[al], out=out
                    ),
                    out,
                )
    for op, other, al in itertools.product(
        ("add", "sub"), ("x8", "2.5"), (1, True, 2j, 2.0)
    ):

        def oth(other=other):
            return selves["x8"]() if other == "x8" else 2.5

        fn = getattr(torch, op)
        yield f"{op} x8 {other} a={al} " + _act_res(
            lambda fn=fn, oth=oth, al=al: fn(selves["x8"](), oth(), alpha=al)
        )
        out = torch.empty(0, device=dev)
        yield f"{op}.out x8 {other} a={al} " + _act_res(
            lambda fn=fn, oth=oth, al=al, out=out: fn(
                selves["x8"](), oth(), alpha=al, out=out
            ),
            out,
        )
    for base, ex, od, shp in itertools.product(
        (1, 1.0, 1 + 0j, True, 2, 2j),
        ("f32", "x8"),
        ("i32", "c64", "f8", "f32"),
        ("0", "exp3"),
    ):
        dt = {"i32": torch.int32, "c64": torch.complex64, "f8": f8}.get(
            od, torch.float32
        )
        out = (
            torch.zeros(1, dtype=dt, device=dev).expand(3)
            if shp == "exp3"
            else torch.zeros(0, dtype=dt, device=dev)
        )
        yield f"pow.Scalar_out {base} {ex} {od}{shp} " + _act_res(
            lambda base=base, ex=ex, out=out: torch.pow(base, selves[ex](), out=out),
            out,
        )


def test_rsub_pow_alpha_match_stock_cuda(mojo_device):
    _compare(list(_rsub_pow_alpha_lines(mojo_device)), _F8_ITER["rsub_pow_alpha"])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="regenerating needs stock CUDA torch"
)
def test_rsub_pow_alpha_data_regenerates_on_stock_cuda():
    assert list(_rsub_pow_alpha_lines("cuda")) == _F8_ITER["rsub_pow_alpha"]


# ---------------------------------------------------------------------------
# review round 5: out= validation (device, internal / partial overlap)
# ---------------------------------------------------------------------------


def _ov_res(fn, out, extra=()):
    """`_act_res` plus where the out's storage lives (a host out must keep
    host storage) and the state of any tensor the out overlaps."""
    s = _act_res(fn, out)
    kind = out.untyped_storage().device.type
    s += " storage=" + ("host" if kind == "cpu" else "dev")
    for e in extra:
        s += " in=" + _mx_enc(e)
    return s


def _out_validation_lines(dev):
    """Every out= path of this PR's float8 / complex work with an out on the
    host, an expanded out (internal overlap), an out partially overlapping
    an input and an out aliasing it, plus add / sub's alpha_check on the
    out's dtype (stock CUDA 2.11)."""
    f8, a = torch.float8_e4m3fn, torch.ops.aten

    def x8():
        return torch.tensor([1.0, 0.5, 2.0]).to(f8).to(dev)

    def f32():
        return torch.tensor([1.0, 2.0, 3.0], device=dev)

    def cx():
        return torch.tensor([1 + 1j, 0.5 - 1j, 2 + 0j], device=dev)

    forms = []
    for odt in (torch.float32, f8, torch.complex64):
        on = str(odt)[6:]
        for xn, xs in (("f32", f32), ("x8", x8)):
            forms += [
                (f"rsub.Scalar_out {xn} {on}", odt,
                 lambda o, xs=xs: a.rsub.Scalar_out(xs(), 4.5, out=o)),
                (f"rsub.Tensor_out {xn} {on}", odt,
                 lambda o, xs=xs: a.rsub.Tensor_out(xs(), f32(), out=o)),
            ]  # fmt: skip
            for base in (1, 2, 1 + 0j, 2j):
                forms.append(
                    (f"pow.Scalar_out {base} {xn} {on}", odt,
                     lambda o, xs=xs, base=base: torch.pow(base, xs(), out=o))
                )  # fmt: skip
        forms += [
            (f"eq x8 {on}", odt, lambda o: torch.eq(x8(), 2, out=o)),
            (f"add x8 {on}", odt, lambda o: torch.add(x8(), x8(), out=o)),
            (f"pow x8 {on}", odt, lambda o: torch.pow(x8(), 2, out=o)),
            (f"abs cx {on}", odt, lambda o: torch.abs(cx(), out=o)),
            (f"neg x8 {on}", odt, lambda o: torch.neg(x8(), out=o)),
            (f"relu f32 {on}", odt, lambda o: a.relu.out(f32(), out=o)),
            (f"irfft cx {on}", odt, lambda o: torch.fft.irfft(cx(), out=o)),
            (f"hfft cx {on}", odt, lambda o: torch.fft.hfft(cx(), out=o)),
        ]  # fmt: skip
    outs = {
        "host0": lambda dt: torch.empty(0, dtype=dt),
        "host3": lambda dt: torch.zeros(3, dtype=dt),
        "exp3": lambda dt: torch.zeros(1, dtype=dt, device=dev).expand(3),
        "exp6": lambda dt: torch.zeros(1, dtype=dt, device=dev).expand(6),
    }
    for name, odt, fn in forms:
        for oname, mk in outs.items():
            out = mk(odt)
            yield f"{name} out={oname} " + _ov_res(lambda fn=fn, out=out: fn(out), out)
    for name, fn in (
        ("rsub.Scalar_out", lambda x, o: a.rsub.Scalar_out(x, 4.5, out=o)),
        ("rsub.Tensor_out", lambda x, o: a.rsub.Tensor_out(x, x.flip(0), out=o)),
        ("pow.Scalar_out 1", lambda x, o: torch.pow(1, x, out=o)),
        ("pow.Scalar_out 2", lambda x, o: torch.pow(2, x, out=o)),
        ("add", lambda x, o: torch.add(x, 1, out=o)),
    ):
        buf = torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)
        yield f"{name} partial " + _ov_res(
            lambda fn=fn, buf=buf: fn(buf[:3], buf[1:4]), buf[1:4], (buf,)
        )
        buf = torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)
        yield f"{name} alias " + _ov_res(
            lambda fn=fn, buf=buf: fn(buf[:3], buf[:3]), buf[:3], (buf,)
        )
    ins = {
        "i32": lambda: torch.ones(2, dtype=torch.int32, device=dev),
        "bool": lambda: torch.ones(2, dtype=torch.bool, device=dev),
        "f32": lambda: torch.ones(2, device=dev),
        "x8": lambda: torch.ones(2).to(f8).to(dev),
    }
    # float8 on either side: a float8 out of any input, a float8 input
    # into any out
    pairs = [(xn, "f8") for xn in ins] + [("x8", "f32"), ("x8", "i32")]
    for op, (xn, od), al in itertools.product(
        ("add", "sub"), pairs, (1, 1.5, True, 2j)
    ):
        dt = {"f8": f8, "f32": torch.float32, "i32": torch.int32}[od]
        out = torch.empty(0, dtype=dt, device=dev)
        yield f"{op}.out {xn} a={al} {od} " + _act_res(
            lambda op=op, xn=xn, al=al, out=out: getattr(torch, op)(
                ins[xn](), ins[xn](), alpha=al, out=out
            ),
            out,
        )


def test_out_validation_matches_stock_cuda(mojo_device):
    _compare(list(_out_validation_lines(mojo_device)), _F8_ITER["out_validation"])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="regenerating needs stock CUDA torch"
)
def test_out_validation_data_regenerates_on_stock_cuda():
    assert list(_out_validation_lines("cuda")) == _F8_ITER["out_validation"]


# ---------------------------------------------------------------------------
# review round 6: alpha converted to an integral / bool computation dtype
# ---------------------------------------------------------------------------


def _alpha_convert_lines(dev):
    """add / sub of integer and bool operands into a float8 out (alpha_check
    passes on the out's floating dtype), so the kernel converts alpha to the
    integral / bool computation dtype with c10's checked_convert: an alpha
    out of range, a negative fraction into an unsigned dtype, NaN / inf
    (stock CUDA 2.11). Fractions within a unit of an integer bound are left
    out: torch 2.11 and 2.14 round that boundary differently. So are sub's
    +128 (int8) and +2**31 (int32), whose -alpha fits: the non-float8 sub
    path checks +alpha (a pre-existing difference of that path)."""
    f8 = torch.float8_e4m3fn
    dts = (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        torch.bool, torch.uint16, torch.uint32, torch.uint64,
    )  # fmt: skip
    alphas = (
        1e20, -1e20, -0.5, 0.5, 2.5, -2.5, float("nan"), float("inf"),
        float("-inf"), 300, -1, -300, 2**40, -128, 255, -255,
    )  # fmt: skip
    for op, dt, al, other, shp in itertools.product(
        ("add", "sub"), dts, alphas, ("t", "3"), ("0", "2")
    ):

        def x(dt=dt):
            return torch.tensor([1, 2], device=dev).to(dt)

        out = torch.zeros(int(shp), dtype=f8, device=dev)
        yield f"{op} {str(dt)[6:]} a={al} o={other} out{shp} " + _act_res(
            lambda op=op, x=x, al=al, other=other, out=out: getattr(torch, op)(
                x(), x() if other == "t" else 3, alpha=al, out=out
            ),
            out,
        )


def test_alpha_convert_matches_stock_cuda(mojo_device):
    _compare(list(_alpha_convert_lines(mojo_device)), _F8_ITER["alpha_convert"])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="regenerating needs stock CUDA torch"
)
def test_alpha_convert_data_regenerates_on_stock_cuda():
    assert list(_alpha_convert_lines("cuda")) == _F8_ITER["alpha_convert"]
