"""Native-backend batch-norm overloads and SyncBatchNorm's building blocks
(tmb/ops/batch_norm.mojo), through public torch APIs on the mojo device.

The overloads over `native_batch_norm` compare against CPU torch. The
SyncBatchNorm blocks have no CPU kernel in torch, so they compare against
the formulas of torch's CUDA kernels (Normalization.cuh) written out here
in float64.
"""

import pytest
import torch

from tests.native.conftest import ran, skip_if_metal
from torch_mojo_backend import register_mojo_devices

aten = torch.ops.aten
HALF = [torch.float16, torch.bfloat16]


@pytest.fixture(autouse=True)
def _registered():
    register_mojo_devices()


def _close(got, want, dtype=torch.float32):
    if dtype in HALF:
        atol = rtol = 1e-2
    elif dtype == torch.float32:
        atol = rtol = 2e-5
    else:
        atol = rtol = 1e-10
    torch.testing.assert_close(
        got.cpu().double(), want.double(), atol=atol, rtol=rtol, equal_nan=True
    )


def _bn_inputs(shape, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = shape[1]
    x = (torch.randn(shape, generator=g) * 3 + 1).to(dtype)
    w = (torch.rand(c, generator=g) + 0.5).to(dtype)
    b = torch.randn(c, generator=g).to(dtype)
    rm = torch.randn(c, generator=g).to(dtype)
    rv = (torch.rand(c, generator=g) + 0.5).to(dtype)
    return x, w, b, rm, rv


def _to(d, *ts):
    return [None if t is None else t.to(d) for t in ts]


# ---------------------------------------------------------------------------
# Overloads over native_batch_norm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4, 3), (2, 5, 7), (3, 4, 5, 6)])
@pytest.mark.parametrize("training", [True, False])
def test_native_batch_norm_legit_and_out(mojo_device, shape, training):
    x, w, b, rm, rv = _bn_inputs(shape, torch.float32)
    want = aten._native_batch_norm_legit(
        x, w, b, rm.clone(), rv.clone(), training, 0.1, 1e-5
    )
    xd, wd, bd = _to(mojo_device, x, w, b)
    rmd, rvd = _to(mojo_device, rm, rv)
    with ran("aten::_native_batch_norm_legit"):
        got = aten._native_batch_norm_legit(xd, wd, bd, rmd, rvd, training, 0.1, 1e-5)
    # CPU returns empty saved statistics in evaluation; CUDA (and this
    # backend) the running mean and rsqrt(running_var + eps).
    if not training:
        want = (want[0], rm, (rv + 1e-5).rsqrt())
    for g, w_ in zip(got, want):
        _close(g, w_)
    if training:
        rm_want, rv_want = rm.clone(), rv.clone()
        aten._native_batch_norm_legit(x, w, b, rm_want, rv_want, True, 0.1, 1e-5)
        _close(rmd, rm_want)
        _close(rvd, rv_want)
    for name in ["native_batch_norm", "_native_batch_norm_legit"]:
        out = torch.empty(0, device=mojo_device)
        save_mean = torch.empty(0, device=mojo_device)
        save_invstd = torch.empty(0, device=mojo_device)
        fn = getattr(aten, name).out
        with ran(f"aten::{name}.out"):
            r = fn(
                xd,
                wd,
                bd,
                rm.to(mojo_device),
                rv.to(mojo_device),
                training,
                0.1,
                1e-5,
                out=out,
                save_mean=save_mean,
                save_invstd=save_invstd,
            )
        assert r[0] is out and r[1] is save_mean and r[2] is save_invstd
        for g, w_ in zip(r, want):
            _close(g, w_)


@pytest.mark.parametrize("shape", [(4, 3), (2, 5, 7)])
def test_native_batch_norm_legit_no_stats(mojo_device, shape):
    x, w, b, _, _ = _bn_inputs(shape, torch.float32)
    want = aten._native_batch_norm_legit.no_stats(x, w, b, True, 0.1, 1e-5)
    xd, wd, bd = _to(mojo_device, x, w, b)
    with ran("aten::_native_batch_norm_legit.no_stats"):
        got = aten._native_batch_norm_legit.no_stats(xd, wd, bd, True, 0.1, 1e-5)
    for g, w_ in zip(got, want):
        _close(g, w_)
    outs = [torch.empty(0, device=mojo_device) for _ in range(3)]
    with ran("aten::_native_batch_norm_legit.no_stats_out"):
        aten._native_batch_norm_legit.no_stats_out(
            xd,
            wd,
            bd,
            True,
            0.1,
            1e-5,
            out=outs[0],
            save_mean=outs[1],
            save_invstd=outs[2],
        )
    for g, w_ in zip(outs, want):
        _close(g, w_)
    with pytest.raises(RuntimeError, match="Expected has_running_mean to be true"):
        aten._native_batch_norm_legit.no_stats(xd, wd, bd, False, 0.1, 1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_batch_norm_with_update_and_backward(mojo_device, dtype):
    x, w, b, rm, rv = _bn_inputs((4, 3, 5), dtype)
    xd, wd, bd = _to(mojo_device, x, w, b)
    rmd, rvd = _to(mojo_device, rm, rv)
    with ran("aten::_batch_norm_with_update"):
        out, save_mean, save_invstd, reserve = aten._batch_norm_with_update(
            xd, wd, bd, rmd, rvd, 0.1, 1e-5
        )
    # The float32 computation is the reference: CUDA (like this backend)
    # computes half inputs in float32 and rounds once.
    rm32, rv32 = rm.float(), rv.float()
    want = aten._batch_norm_with_update(
        x.float(), w.float(), b.float(), rm32, rv32, 0.1, 1e-5
    )
    assert reserve.dtype == torch.uint8 and reserve.numel() == 0
    _close(out, want[0].to(dtype), dtype)
    _close(save_mean, want[1])
    _close(save_invstd, want[2])
    _close(rmd, rm32.to(dtype), dtype)
    _close(rvd, rv32.to(dtype), dtype)
    grad = torch.randn(x.shape).to(dtype)
    want_g = aten.native_batch_norm_backward(
        grad.float(),
        x.float(),
        w.float(),
        rm32,
        rv32,
        want[1],
        want[2],
        True,
        1e-5,
        [True, True, True],
    )
    want_g = [t.to(dtype) for t in want_g]
    with ran("aten::batch_norm_backward"):
        got_g = aten.batch_norm_backward(
            grad.to(mojo_device),
            xd,
            wd,
            rmd,
            rvd,
            save_mean,
            save_invstd,
            True,
            1e-5,
            [True, True, True],
            reserve,
        )
    for g, w_ in zip(got_g, want_g):
        _close(g, w_, dtype)
    masked = aten.batch_norm_backward(
        grad.to(mojo_device),
        xd,
        wd,
        rmd,
        rvd,
        save_mean,
        save_invstd,
        True,
        1e-5,
        [False, True, False],
        reserve,
    )
    assert masked[0].numel() == 0 and masked[2].numel() == 0
    _close(masked[1], want_g[1], dtype)


def test_batch_norm_with_update_out(mojo_device):
    x, w, b, rm, rv = _bn_inputs((4, 3), torch.float32)
    want = aten._batch_norm_with_update(x, w, b, rm.clone(), rv.clone(), 0.2, 1e-5)
    outs = [torch.empty(0, device=mojo_device) for _ in range(3)]
    reserve = torch.empty(0, dtype=torch.uint8, device=mojo_device)
    with ran("aten::_batch_norm_with_update.out"):
        r = aten._batch_norm_with_update.out(
            *_to(mojo_device, x, w, b, rm, rv),
            0.2,
            1e-5,
            out=outs[0],
            save_mean=outs[1],
            save_invstd=outs[2],
            reserve=reserve,
        )
    assert r[3] is reserve
    for g, w_ in zip(outs, want):
        _close(g, w_)


def test_batch_norm_inference_without_affine(mojo_device):
    x, _, _, rm, rv = _bn_inputs((3, 4, 5), torch.float32)
    want = aten.native_batch_norm(x, None, None, rm, rv, False, 0.1, 1e-5)
    got = aten.native_batch_norm(
        *_to(mojo_device, x, None, None, rm, rv), False, 0.1, 1e-5
    )
    _close(got[0], want[0])
    _close(got[1], rm)
    _close(got[2], (rv + 1e-5).rsqrt())


@pytest.mark.parametrize("dtype", HALF)
def test_batch_norm_inference_half_running_stats(mojo_device, dtype):
    """batch_norm_cuda returns float32 saved statistics for any input."""
    x, w, b, rm, rv = _bn_inputs((3, 4, 5), dtype)
    got = aten.native_batch_norm(*_to(mojo_device, x, w, b, rm, rv), False, 0.1, 1e-5)
    assert got[1].dtype == torch.float32 and got[2].dtype == torch.float32
    _close(got[1], rm.float())
    _close(got[2], (rv.float() + 1e-5).rsqrt())


def test_batch_norm_training_single_value_per_channel(mojo_device):
    """One value per channel: CUDA's unbiased running variance is
    `0 * N / (N - 1)` = NaN, the output 0."""
    x, w, b, rm, rv = _bn_inputs((1, 3, 1, 1), torch.float32)
    rmd, rvd = _to(mojo_device, rm, rv)
    out, _, _ = aten.native_batch_norm(
        *_to(mojo_device, x, w, b), rmd, rvd, True, 0.1, 1e-5
    )
    _close(out, b.view(1, 3, 1, 1).expand(1, 3, 1, 1))
    assert torch.isnan(rvd.cpu()).all()


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("affine", [True, False])
def test_native_batch_norm_float64(mojo_device, training, affine):
    skip_if_metal(mojo_device, "Apple GPUs have no float64")
    x, w, b, rm, rv = _bn_inputs((4, 3, 6), torch.float64)
    if not affine:
        w = b = None
    rm_w, rv_w = rm.clone(), rv.clone()
    want = aten.native_batch_norm(x, w, b, rm_w, rv_w, training, 0.1, 1e-5)
    if not training:  # CPU leaves the saved statistics empty, CUDA does not
        want = (want[0], rm, (rv + 1e-5).rsqrt())
    rmd, rvd = _to(mojo_device, rm, rv)
    got = aten.native_batch_norm(
        *_to(mojo_device, x, w, b), rmd, rvd, training, 0.1, 1e-5
    )
    for g, w_ in zip(got, want):
        _close(g, w_, torch.float64)
    _close(rmd, rm_w, torch.float64)
    _close(rvd, rv_w, torch.float64)
    nt = aten._native_batch_norm_legit_no_training(
        *_to(mojo_device, x, w, b, rm, rv), 0.1, 1e-5
    )
    want_nt = aten._native_batch_norm_legit_no_training(x, w, b, rm, rv, 0.1, 1e-5)
    want_nt = (want_nt[0], rm, (rv + 1e-5).rsqrt())
    for g, w_ in zip(nt, want_nt):
        _close(g, w_, torch.float64)


# ---------------------------------------------------------------------------
# SyncBatchNorm building blocks (CUDA formulas, Normalization.cuh)
# ---------------------------------------------------------------------------


def _planes(x):
    return x.double().reshape(x.shape[0], x.shape[1], -1)


def _ref_stats(x):
    p = _planes(x)
    mean = p.mean(dim=(0, 2))
    var = ((p - mean.view(1, -1, 1)) ** 2).mean(dim=(0, 2))
    return mean, var


SHAPES = [(4, 3), (2, 5, 7), (3, 4, 5, 6), (8, 16, 33)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_batch_norm_stats(mojo_device, dtype, shape):
    x, *_ = _bn_inputs(shape, dtype)
    mean, var = _ref_stats(x)
    with ran("aten::batch_norm_stats"):
        got_mean, got_invstd = aten.batch_norm_stats(x.to(mojo_device), 1e-5)
    assert got_mean.dtype == torch.float32
    _close(got_mean, mean)
    _close(got_invstd, 1 / (var + 1e-5).sqrt())
    # InvStd: a zero variance with a zero eps gives 0, not inf.
    const = torch.ones(3, 2, dtype=dtype)
    _, invstd = aten.batch_norm_stats(const.to(mojo_device), 0.0)
    assert (invstd.cpu() == 0).all()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("running", [True, False])
def test_batch_norm_update_stats(mojo_device, shape, running):
    x, _, _, rm, rv = _bn_inputs(shape, torch.float32)
    mean, var = _ref_stats(x)
    n = x.numel() // x.shape[1]
    rmd, rvd = _to(mojo_device, rm, rv) if running else (None, None)
    got_mean, got_var = aten.batch_norm_update_stats(x.to(mojo_device), rmd, rvd, 0.1)
    _close(got_mean, mean)
    _close(got_var, var)
    if running:
        _close(rmd, mean * 0.1 + 0.9 * rm.double())
        _close(rvd, var * n / (n - 1) * 0.1 + 0.9 * rv.double())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("affine", [True, False])
def test_batch_norm_elemt(mojo_device, dtype, shape, affine):
    x, w, b, _, _ = _bn_inputs(shape, dtype)
    c = shape[1]
    mean = torch.randn(c)
    invstd = torch.rand(c) + 0.5
    bshape = [1, c] + [1] * (len(shape) - 2)
    want = (x.double() - mean.double().view(bshape)) * invstd.double().view(bshape)
    if affine:
        want = w.double().view(bshape) * want + b.double().view(bshape)
    else:
        w = b = None
    with ran("aten::batch_norm_elemt"):
        got = aten.batch_norm_elemt(*_to(mojo_device, x, w, b, mean, invstd), 1e-5)
    _close(got, want.to(dtype), dtype)
    out = torch.empty(0, dtype=dtype, device=mojo_device)
    with ran("aten::batch_norm_elemt.out"):
        aten.batch_norm_elemt.out(
            *_to(mojo_device, x, w, b, mean, invstd), 1e-5, out=out
        )
    _close(out, want.to(dtype), dtype)


def test_batch_norm_elemt_type_checks(mojo_device):
    d = mojo_device
    x = torch.randn(2, 3, 4, dtype=torch.half, device=d)
    m = torch.zeros(3, dtype=torch.half, device=d)
    with pytest.raises(
        RuntimeError, match="Expected mean to have type Float but got Half"
    ):
        aten.batch_norm_elemt(x, None, None, m, m, 1e-5)


def _ref_gather(mean, invstd, counts, eps):
    avg = torch.zeros(mean.shape[1], dtype=torch.float64)
    var_n = torch.zeros_like(avg)
    n = 0
    for j in range(mean.shape[0]):
        cnt = float(counts[j])
        m = mean[j].double()
        v = (1 / invstd[j].double()) ** 2 - eps
        v = v * cnt
        factor = 1 / (n + cnt)
        var_n = var_n + v + (avg - m) ** 2 * n * cnt * factor
        avg = n * factor * avg + cnt * factor * m
        n = int(n + cnt)
    return avg, 1 / (var_n / n + eps).sqrt(), var_n / (n - 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_batch_norm_gather_stats(mojo_device, dtype):
    c = 5
    x = torch.randn(4, c).to(dtype)
    mean = torch.randn(3, c)
    invstd = torch.rand(3, c) + 0.5
    counts = torch.tensor([4.0, 6.0, 5.0]).to(dtype)
    rm = torch.randn(c).to(dtype)
    rv = (torch.rand(c) + 0.5).to(dtype)
    avg, inv, unbiased = _ref_gather(mean, invstd, counts, 1e-5)
    rmd, rvd = _to(mojo_device, rm, rv)
    with ran("aten::batch_norm_gather_stats_with_counts"):
        got = aten.batch_norm_gather_stats_with_counts(
            *_to(mojo_device, x, mean, invstd),
            rmd,
            rvd,
            0.1,
            1e-5,
            counts.to(mojo_device),
        )
    _close(got[0], avg)
    _close(got[1], inv)
    _close(rmd, (0.9 * rm.double() + 0.1 * avg).to(dtype), dtype)
    _close(rvd, (0.9 * rv.double() + 0.1 * unbiased).to(dtype), dtype)
    avg7, inv7, _ = _ref_gather(mean, invstd, torch.full((3,), 7.0), 1e-3)
    with ran("aten::batch_norm_gather_stats"):
        got7 = aten.batch_norm_gather_stats(
            *_to(mojo_device, x, mean, invstd), None, None, 0.1, 1e-3, 7
        )
    _close(got7[0], avg7)
    _close(got7[1], inv7)
    with pytest.raises(RuntimeError, match="expected mean to be 2-dimensional"):
        aten.batch_norm_gather_stats_with_counts(
            *_to(mojo_device, x, mean[0], invstd[0]),
            None,
            None,
            0.1,
            1e-5,
            counts.to(mojo_device),
        )


def _ref_backward_reduce(g, x, mean, invstd):
    gp, xp = _planes(g), _planes(x)
    sum_dy = gp.sum(dim=(0, 2))
    dot = (gp * (xp - mean.double().view(1, -1, 1))).sum(dim=(0, 2))
    return sum_dy, dot, dot * invstd.double(), sum_dy


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_batch_norm_backward_reduce(mojo_device, dtype, shape):
    x, w, _, _, _ = _bn_inputs(shape, dtype)
    g = torch.randn(shape).to(dtype)
    c = shape[1]
    mean = torch.randn(c)
    invstd = torch.rand(c) + 0.5
    want = _ref_backward_reduce(g, x, mean, invstd)
    with ran("aten::batch_norm_backward_reduce"):
        got = aten.batch_norm_backward_reduce(
            *_to(mojo_device, g, x, mean, invstd, w), True, True, True
        )
    # Half sums are rounded through the dtype on CUDA's route: compare at
    # the dtype's precision, scaled by the magnitude of the sum.
    for k in range(4):
        tol = 2e-5 if dtype == torch.float32 else 3e-2
        scale = 1 + _planes(g).abs().sum(dim=(0, 2)) * (1 if k % 2 == 0 else 4)
        assert ((got[k].cpu().double() - want[k]).abs() <= tol * scale).all()


def test_batch_norm_backward_reduce_routes(mojo_device):
    """CUDA's channels-last route (here: a 2-D input) computes all four
    results; the other route leaves the unrequested ones undefined."""
    d = mojo_device
    m = torch.zeros(3, device=d)
    s = torch.ones(3, device=d)
    w = torch.ones(3, device=d)
    x2, g2 = torch.randn(4, 3, device=d), torch.randn(4, 3, device=d)
    r = aten.batch_norm_backward_reduce(g2, x2, m, s, w, True, False, False)
    assert all(t is not None and t.shape == (3,) for t in r)
    r = aten.batch_norm_backward_reduce(g2, x2, m, s, None, True, True, True)
    assert r[2].shape == (0,) and r[3].shape == (0,)
    x3, g3 = torch.randn(4, 3, 5, device=d), torch.randn(4, 3, 5, device=d)
    r = aten.batch_norm_backward_reduce(g3, x3, m, s, w, True, False, False)
    assert r[0] is not None and r[2] is None and r[3] is None
    r = aten.batch_norm_backward_reduce(g3, x3, m, s, w, False, True, True)
    assert r[0] is None and r[1] is None and r[2] is not None


def _cuda_warp_reduce(vals, dtype):
    """block_reduce.cuh WarpReduce over a Float2 whose shuffled operands are
    rebuilt through the half dtype (float32 accumulation)."""
    v = [torch.tensor(x, dtype=torch.float32) for x in vals]
    off = 16
    while off > 0:
        old = list(v)
        for lane in range(32):
            src = lane + off if lane + off < 32 else lane
            v[lane] = v[lane] + old[src].to(dtype).float()
        off //= 2
    return v[0]


def test_batch_norm_backward_reduce_half_follows_cuda_rounding(mojo_device):
    """(2, 5, 7) float16: CUDA's block is 32 x 2 threads per channel; the
    emulated tree must land on exactly the same float32 sums."""
    dtype = torch.float16
    gen = torch.Generator().manual_seed(12)
    x = torch.randn(2, 5, 7, generator=gen).to(dtype)
    g = torch.randn(2, 5, 7, generator=gen).to(dtype)
    mean = torch.randn(5, generator=gen)
    invstd = torch.rand(5, generator=gen) + 0.5
    got = aten.batch_norm_backward_reduce(
        *_to(mojo_device, g, x, mean, invstd, None), True, False, False
    )
    bx, by = 32, 2
    for c in range(5):
        thread = [[0.0, 0.0] for _ in range(bx * by)]
        for tid in range(bx * by):
            tx, ty = tid % bx, tid // bx
            s1 = torch.tensor(0.0)
            s2 = torch.tensor(0.0)
            for n in range(ty, 2, by):
                for xi in range(tx, 7, bx):
                    gv = g[n, c, xi].float()
                    cv = x[n, c, xi].float() - mean[c]
                    s1 = s1 + gv.to(dtype).float()
                    s2 = s2 + (gv * cv).to(dtype).float()
            thread[tid] = [float(s1), float(s2)]
        warps = []
        for w0 in range(0, bx * by, 32):
            lanes = thread[w0 : w0 + 32]
            warps.append(
                [_cuda_warp_reduce([t[k] for t in lanes], dtype) for k in range(2)]
            )
        final = []
        for k in range(2):
            vals = [float(warps[i][k]) if i < len(warps) else 0.0 for i in range(32)]
            final.append(_cuda_warp_reduce(vals, dtype))
        assert float(got[0][c]) == float(final[0])
        assert float(got[1][c]) == float(final[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("affine", [True, False])
def test_batch_norm_backward_elemt(mojo_device, dtype, shape, affine):
    x, w, _, _, _ = _bn_inputs(shape, dtype)
    if dtype != torch.float32:
        w = w.float()  # CUDA's mixed route: float32 weight with float32 stats
    g = torch.randn(shape).to(dtype)
    c = shape[1]
    mean = torch.randn(c)
    invstd = torch.rand(c) + 0.5
    sum_dy = torch.randn(c)
    sum_dy_xmu = torch.randn(c)
    count = torch.tensor([5, 7, 3], dtype=torch.int32)
    norm = 1 / 15
    bshape = [1, c] + [1] * (len(shape) - 2)
    wv = w.double().view(bshape) if affine else 1.0
    iv = invstd.double().view(bshape)
    want = (
        g.double()
        - (sum_dy.double() * norm).view(bshape)
        - (x.double() - mean.double().view(bshape))
        * (iv * iv * sum_dy_xmu.double().view(bshape) * norm)
    ) * (wv * iv)
    with ran("aten::batch_norm_backward_elemt"):
        got = aten.batch_norm_backward_elemt(
            *_to(
                mojo_device,
                g,
                x,
                mean,
                invstd,
                w if affine else None,
                sum_dy,
                sum_dy_xmu,
                count,
            )
        )
    _close(got, want.to(dtype), dtype)
    with pytest.raises(RuntimeError, match="expected scalar type Int but found Long"):
        aten.batch_norm_backward_elemt(
            *_to(
                mojo_device, g, x, mean, invstd, None, sum_dy, sum_dy_xmu, count.long()
            )
        )
