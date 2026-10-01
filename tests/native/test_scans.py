"""The scans group of the native mojo backend (tmb/ops/scans.mojo): cumsum,
cumprod, logcumsumexp and cummax / cummin, through public torch APIs,
compared against CPU torch."""

import pytest
import torch

from tests.native.conftest import ran, skip_if_metal


def _round(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t.to(dtype).float()


def cuda_lowp_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """CUDA's half/bfloat16 cumsum, step for step (ScanUtils.cuh `scan_dim`):
    every addition rounds to the element dtype, in the order of the route
    CUDA takes -- one sequential line per element of the other dims (outer
    dim), Sklansky tiles with the carry folded into each tile's first element
    (innermost dim), cub's 128-thread x 30-item tiles (1-D; in-order
    look-back)."""
    dt = x.dtype
    dim = dim % x.dim()
    if x.numel() == x.size(dim):
        return _cub_1d(x.flatten(), dt).reshape(x.shape)
    if dim != x.dim() - 1:
        xm = x.movedim(dim, 0).float()
        out = torch.empty_like(xm)
        acc = torch.zeros_like(xm[0])
        for i in range(xm.shape[0]):
            acc = _round(acc + xm[i], dt)
            out[i] = acc
        return out.to(dt).movedim(0, dim)
    rows = x.numel() // x.size(-1)
    n = x.size(-1)
    lx = max(0, (n - 1).bit_length())
    ly = max(0, (rows - 1).bit_length())
    log_x = ((9 + lx - ly) % (1 << 32)) // 2  # uint32 arithmetic
    log_x = min(max(log_x, 4), 9)
    nx = 1 << log_x
    xr = x.reshape(rows, n).float()
    out = torch.empty_like(xr)
    total = torch.zeros(rows)
    for c0 in range(0, n, 2 * nx):
        buf = torch.zeros(rows, 2 * nx)
        w = min(2 * nx, n - c0)
        buf[:, :w] = xr[:, c0 : c0 + w]
        buf[:, 0] = _round(buf[:, 0] + total, dt)
        for m in range(log_x + 1):
            sz = 1 << m
            t = torch.arange(nx)
            a = ((t >> m) << (m + 1)) | sz
            ti = a + (t % sz)
            si = a - 1
            buf[:, ti] = _round(buf[:, ti] + buf[:, si], dt)
        out[:, c0 : c0 + w] = buf[:, :w]
        total = buf[:, 2 * nx - 1].clone()
    return out.to(dt).reshape(x.shape)


def _cub_1d(x: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    threads, items = 128, 30
    xf = x.float()
    n = xf.numel()
    out = torch.empty(n)
    prefix = None
    for t0 in range(0, n, threads * items):
        tile = torch.zeros(threads * items)
        w = min(threads * items, n - t0)
        tile[:w] = xf[t0 : t0 + w]
        cnt = torch.clamp(w - torch.arange(threads) * items, 0, items)
        tile = tile.reshape(threads, items)
        agg = tile[:, 0].clone()
        for k in range(1, items):
            agg = torch.where(k < cnt, _round(agg + tile[:, k], dt), agg)
        v = agg.clone()
        for off in (1, 2, 4, 8, 16):
            lane = torch.arange(threads) % 32
            shifted = torch.roll(v, off)
            v = torch.where(lane >= off, _round(shifted + v, dt), v)
        warp_tot = v.reshape(4, 32)[:, 31]
        excl = torch.roll(v, 1)
        has = (torch.arange(threads) % 32) > 0
        # wps[w]: the in-order total of the warps before w (wps[0] unused).
        wps = [warp_tot[0]]
        acc = warp_tot[0]
        for wi in range(1, 4):
            wps.append(acc)
            acc = _round(acc + warp_tot[wi], dt)
        block_tot = acc
        for tid in range(threads):
            wi = tid // 32
            if wi > 0:
                excl[tid] = _round(wps[wi] + excl[tid], dt) if has[tid] else wps[wi]
                has[tid] = True
        if prefix is not None:
            excl = torch.where(has, _round(prefix + excl, dt), prefix)
            has[:] = True
        run = excl.clone()
        res = torch.zeros(threads, items)
        for k in range(items):
            first = (k == 0) & ~has
            run = torch.where(first, tile[:, k], _round(run + tile[:, k], dt))
            res[:, k] = run
        out[t0 : t0 + w] = res.flatten()[:w]
        prefix = block_tot if prefix is None else _round(prefix + block_tot, dt)
    return out.to(dt)


SHAPES_DIMS = [
    ((7,), 0),
    ((5, 300), 1),
    ((5, 300), 0),
    ((3, 4, 5), 1),
    ((3, 4, 5), 0),
    ((2, 3, 70), -1),
    ((2, 1, 4), 1),
]


def _float_tol(dtype: torch.dtype) -> tuple[float | None, float | None]:
    if dtype in (torch.float16, torch.bfloat16):
        # CUDA (and this device) round the half running value after every
        # step; the reference rounds once.
        return 1.0, 5e-2
    if dtype == torch.float32:
        # The block scans add in a tree, CPU in sequence: a few ulp of the
        # running sum, which near a cancellation is large relative to it.
        return 1e-4, 1e-5
    return None, None


@pytest.mark.parametrize(("shape", "dim"), SHAPES_DIMS)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.int32]
)
def test_cumsum_every_dim_matches_cpu(mojo_gpu, shape, dim, dtype):
    x = (torch.randn(shape) * 4).to(dtype)
    with ran("aten::cumsum"):
        got = torch.cumsum(x.to(mojo_gpu), dim)
    if dtype in (torch.float16, torch.bfloat16):
        # Bit-exact against CUDA's rounding order.
        torch.testing.assert_close(got.cpu(), cuda_lowp_cumsum(x, dim), atol=0, rtol=0)
        return
    ref = torch.cumsum(x.float() if dtype.is_floating_point else x, dim).to(got.dtype)
    atol, rtol = _float_tol(dtype)
    torch.testing.assert_close(got.cpu(), ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize(("shape", "dim"), SHAPES_DIMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64])
def test_cumprod_matches_cpu(mojo_gpu, shape, dim, dtype):
    x = (torch.rand(shape) * 0.4 + 0.8).to(dtype)
    if dtype == torch.int64:
        x = torch.randint(-2, 3, shape)
    with ran("aten::cumprod"):
        got = torch.cumprod(x.to(mojo_gpu), dim)
    ref = torch.cumprod(x.float() if dtype.is_floating_point else x, dim).to(got.dtype)
    atol, rtol = _float_tol(dtype)
    torch.testing.assert_close(got.cpu(), ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("op", [torch.cumsum, torch.cumprod])
@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.int8, torch.int16])
def test_cum_integer_inputs_promote_to_int64(mojo_gpu, op, dtype):
    x = torch.randint(0, 2, (4, 6)).to(dtype)
    got = op(x.to(mojo_gpu), 1)
    assert got.dtype == torch.int64
    torch.testing.assert_close(got.cpu(), op(x, 1))


@pytest.mark.parametrize("op", [torch.cumsum, torch.cumprod])
def test_cum_dtype_kwarg_casts_first(mojo_gpu, op):
    x = torch.tensor([[1.7, 2.2, -0.6], [0.5, 1.5, 2.5]])
    got = op(x.to(mojo_gpu), 1, dtype=torch.int64)
    torch.testing.assert_close(got.cpu(), op(x, 1, dtype=torch.int64))
    got = op(x.to(mojo_gpu), 0, dtype=torch.float16)
    torch.testing.assert_close(got.cpu(), op(x, 0, dtype=torch.float16))


@pytest.mark.parametrize("op", [torch.cumsum, torch.cumprod])
def test_cum_out_and_inplace(mojo_gpu, op):
    x = torch.randn(4, 5)
    out = torch.empty(0, device=mojo_gpu)
    op(x.to(mojo_gpu), 0, out=out)
    torch.testing.assert_close(out.cpu(), op(x, 0))
    # A non-contiguous out of the right shape is written where it lives.
    base = torch.zeros(5, 4, device=mojo_gpu)
    op(x.to(mojo_gpu), 1, out=base.t())
    torch.testing.assert_close(base.t().cpu(), op(x, 1))
    # The out's dtype is the compute dtype when dtype= is absent.
    out_d = torch.empty(4, 5, dtype=torch.float64, device=mojo_gpu)
    skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    op(x.to(mojo_gpu), 1, out=out_d)
    torch.testing.assert_close(out_d.cpu(), op(x.double(), 1))


@pytest.mark.parametrize("op", ["cumsum_", "cumprod_"])
def test_cum_inplace_method(mojo_gpu, op):
    x = torch.randn(3, 6)
    y = x.to(mojo_gpu)
    r = getattr(y, op)(1)
    assert r.data_ptr() == y.data_ptr()
    torch.testing.assert_close(y.cpu(), getattr(x.clone(), op)(1))
    with pytest.raises(RuntimeError, match="Bad in-place call"):
        getattr(y, op)(1, dtype=torch.float16)


@pytest.mark.parametrize("op", [torch.cumsum, torch.cumprod])
def test_cum_inplace_refuses_internal_overlap(mojo_gpu, op):
    name = op.__name__ + "_"
    with pytest.raises(RuntimeError, match="single memory location"):
        getattr(torch.ones(1, device=mojo_gpu).expand(3), name)(0)


@pytest.mark.parametrize("op", [torch.cumsum, torch.cumprod])
def test_cum_zero_dim_and_empty(mojo_gpu, op):
    s = torch.tensor(3.5)
    torch.testing.assert_close(op(s.to(mojo_gpu), 0).cpu(), op(s, 0))
    torch.testing.assert_close(op(s.to(mojo_gpu), -1).cpu(), op(s, -1))
    e = torch.empty(0, 3)
    assert op(e.to(mojo_gpu), 1).shape == (0, 3)
    with pytest.raises(IndexError):
        op(torch.randn(2, 3, device=mojo_gpu), 2)


@pytest.mark.parametrize("op", [torch.cumsum, torch.cumprod])
def test_cum_noncontiguous_input(mojo_gpu, op):
    x = torch.randn(6, 5)
    xm = x.to(mojo_gpu)
    torch.testing.assert_close(op(xm.t(), 1).cpu(), op(x.t(), 1))
    torch.testing.assert_close(op(xm[:, ::2], 0).cpu(), op(x[:, ::2], 0))


def test_cum_nonfinite_propagates(mojo_gpu):
    x = torch.tensor([[1.0, float("nan"), 2.0], [float("inf"), -1.0, 3.0]])
    for op in (torch.cumsum, torch.cumprod):
        for d in (0, 1):
            torch.testing.assert_close(
                op(x.to(mojo_gpu), d).cpu(), op(x, d), equal_nan=True
            )


def test_cum_long_rows(mojo_gpu):
    """Rows longer than one 256-wide tile take the block-scan route."""
    x = torch.rand(3, 1000) * 0.02 + 0.995
    torch.testing.assert_close(
        torch.cumprod(x.to(mojo_gpu), 1).cpu(),
        torch.cumprod(x, 1),
        rtol=1e-4,
        atol=1e-5,
    )
    xi = torch.randint(-50, 50, (2, 3000))
    torch.testing.assert_close(
        torch.cumsum(xi.to(mojo_gpu), 1).cpu(), torch.cumsum(xi, 1)
    )


@pytest.mark.parametrize(("shape", "dim"), SHAPES_DIMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_logcumsumexp_matches_cpu(mojo_gpu, shape, dim, dtype):
    x = (torch.randn(shape) * 3).to(dtype)
    with ran("aten::_logcumsumexp"):
        got = torch.logcumsumexp(x.to(mojo_gpu), dim)
    ref = torch.logcumsumexp(x.double(), dim)
    atol, rtol = (1e-5, 1e-5) if dtype == torch.float32 else (3e-2, 2e-2)
    torch.testing.assert_close(got.cpu().double(), ref, atol=atol, rtol=rtol)


def test_logcumsumexp_nonfinite(mojo_gpu):
    inf = float("inf")
    x = torch.tensor(
        [
            [-inf, -inf, 1.0, -inf],
            [inf, inf, 1.0, -inf],
            [0.0, float("nan"), 1.0, 2.0],
            [inf, -inf, 3.0, inf],
        ]
    )
    torch.testing.assert_close(
        torch.logcumsumexp(x.to(mojo_gpu), 1).cpu(),
        torch.logcumsumexp(x, 1),
        equal_nan=True,
    )


def test_logcumsumexp_out_scalar_empty_and_errors(mojo_gpu):
    x = torch.randn(3, 4)
    out = torch.empty(0, device=mojo_gpu)
    torch.logcumsumexp(x.to(mojo_gpu), 1, out=out)
    torch.testing.assert_close(out.cpu(), torch.logcumsumexp(x, 1))
    s = torch.tensor(-2.0)
    torch.testing.assert_close(torch.logcumsumexp(s.to(mojo_gpu), 0).cpu(), s)
    assert torch.logcumsumexp(torch.empty(0, 2, device=mojo_gpu), 0).shape == (0, 2)
    with pytest.raises(RuntimeError, match="not implemented for 'Long'"):
        torch.logcumsumexp(torch.arange(4, device=mojo_gpu), 0)


@pytest.mark.parametrize("op", [torch.cummax, torch.cummin])
@pytest.mark.parametrize(("shape", "dim"), SHAPES_DIMS)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int64,
        torch.int32,
        torch.bool,
        torch.uint8,
        torch.int8,
    ],
)
def test_cummax_cummin_match_cpu(mojo_gpu, op, shape, dim, dtype):
    # Few distinct values, so ties (the LAST index wins) are common.
    x = torch.randint(-3, 4, shape).to(dtype)
    name = "aten::_cummax_helper" if op is torch.cummax else "aten::_cummin_helper"
    with ran(name):
        values, indices = op(x.to(mojo_gpu), dim)
    ref_v, ref_i = op(x, dim)
    torch.testing.assert_close(values.cpu(), ref_v)
    torch.testing.assert_close(indices.cpu(), ref_i)


@pytest.mark.parametrize("op", [torch.cummax, torch.cummin])
def test_cummax_cummin_nan_sticks(mojo_gpu, op):
    nan = float("nan")
    x = torch.tensor([[1.0, nan, 3.0, nan, -1.0], [2.0, 2.0, -5.0, 9.0, 9.0]])
    x = torch.cat([x, torch.randn(2, 300)], 1)
    values, indices = op(x.to(mojo_gpu), 1)
    ref_v, ref_i = op(x, 1)
    torch.testing.assert_close(values.cpu(), ref_v, equal_nan=True)
    torch.testing.assert_close(indices.cpu(), ref_i)


@pytest.mark.parametrize("op", [torch.cummax, torch.cummin])
def test_cummax_cummin_out_and_noncontiguous(mojo_gpu, op):
    x = torch.randn(5, 4)
    # Right-shaped outs: ATen's composite cummax_out resizes through
    # aten::resize_, which the mojo device does not implement.
    v = torch.empty(4, 5, device=mojo_gpu)
    i = torch.empty(5, 4, dtype=torch.int64, device=mojo_gpu).t()
    op(x.to(mojo_gpu).t(), 1, out=(v, i))
    ref_v, ref_i = op(x.t(), 1)
    torch.testing.assert_close(v.cpu(), ref_v)
    torch.testing.assert_close(i.cpu(), ref_i)
    s = torch.tensor(2.0)
    rv, ri = op(s.to(mojo_gpu), 0)
    assert rv.item() == 2.0 and ri.item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_cumprod_keeps_cudas_overflow(mojo_gpu, dtype):
    """CUDA's scan keeps the running product in the element dtype: once it
    overflows to inf it stays inf (a float32 running value would come back
    down to 32768)."""
    x = torch.tensor([[256.0, 256.0], [256.0, 256.0], [0.5, 0.5]], dtype=dtype)
    got = torch.cumprod(x.to(mojo_gpu), 0).cpu().float()
    if dtype == torch.float16:
        assert got[:, 0].tolist() == [256.0, float("inf"), float("inf")]
    # bfloat16 does not overflow there; it must match the dtype-rounded scan.
    ref = x.clone()
    for r in range(1, 3):
        ref[r] = ref[r - 1] * x[r]
    torch.testing.assert_close(got, ref.float())


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.int16])
@pytest.mark.parametrize("op", [torch.cumsum, torch.cumprod])
def test_cum_into_narrow_integer_dtype(mojo_gpu, op, dtype):
    x = torch.randint(-5, 50, (4, 9))
    got = op(x.to(mojo_gpu), 1, dtype=dtype)
    assert got.dtype == dtype
    torch.testing.assert_close(got.cpu(), op(x, 1, dtype=dtype))


def test_logcumsumexp_integer_scalar_and_empty(mojo_gpu):
    s = torch.tensor(5)
    got = torch.logcumsumexp(s.to(mojo_gpu), 0)
    assert got.dtype == torch.int64 and got.item() == 5
    e = torch.empty(0, 3, dtype=torch.int64)
    got = torch.logcumsumexp(e.to(mojo_gpu), 1)
    assert got.shape == (0, 3) and got.dtype == torch.int64


# Stock CUDA (torch 2.14, H100), recorded: max and sum of cumsum(ones).
_CUDA_ONES_CUMSUM = {
    ((2, 4096, 2), 1, torch.float16): (2048.0, 25169920.0),
    ((2, 4096, 2), 1, torch.bfloat16): (256.0, 4063744.0),
    ((3, 5000), 1, torch.float16): (4996.0, 37484376.0),
    ((3, 5000), 1, torch.bfloat16): (4960.0, 37298304.0),
    ((4096,), 0, torch.float16): (4080.0, 8361152.0),
    ((4096,), 0, torch.bfloat16): (4080.0, 8331240.0),
    ((1, 4096), 1, torch.float16): (4080.0, 8361152.0),
    ((1, 4096), 1, torch.bfloat16): (4080.0, 8331240.0),
    # A trailing size-1 dim: the scan dim is not the last one, so CUDA takes
    # its outer-dim (sequential) route.
    ((2, 4096, 1), 1, torch.float16): (2048.0, 12584960.0),
    ((2, 4096, 1), 1, torch.bfloat16): (256.0, 2031872.0),
}


@pytest.mark.parametrize(("shape", "dim", "dtype"), list(_CUDA_ONES_CUMSUM))
def test_lowp_cumsum_rounds_like_cuda(mojo_gpu, shape, dim, dtype):
    """CUDA's scan_dim<scalar_t> keeps the half running sum in half on all of
    its routes (outer dim, innermost dim, cub for 1-D): cumsum(ones) saturates
    where CUDA's does."""
    x = torch.ones(shape, dtype=dtype)
    got = torch.cumsum(x.to(mojo_gpu), dim).cpu().float()
    assert (got.max().item(), got.sum().item()) == _CUDA_ONES_CUMSUM[
        (shape, dim, dtype)
    ]
    torch.testing.assert_close(got, cuda_lowp_cumsum(x, dim).float(), atol=0, rtol=0)
