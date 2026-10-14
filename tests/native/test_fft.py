"""Tests for the FFTs of the mojo device (aten::_fft_c2c / _fft_r2c /
_fft_c2r, torch_mojo_backend/mojo/tmb/ops/spectral.mojo), through the public
torch.fft API, against CPU torch.

Sizes cover 1, 2, primes (Bluestein: 97, 1009), powers of two, mixed radix
2*3*5*7, a large 2**16, batched / non-contiguous / multi-dim inputs, every
norm mode, float32 and float64 (float64 is not on Apple GPUs), and out=.

Tolerances: an FFT's rounding error scales with the magnitude of its
outputs, not with each output (a near-zero bin of a large transform keeps an
absolute error of eps * max|X|), so the bound is relative to the largest
output -- what both cuFFT and pocketfft achieve (~2e-7 for float32).
"""

import os
import subprocess
import sys
import textwrap
import threading

import pytest
import torch

from tests.native.conftest import skip_if_metal
from torch_mojo_backend import native

SIZES = [1, 2, 3, 8, 12, 97, 210, 1009, 4096]
NORMS = [None, "ortho", "forward", "backward"]


def _skip64(device: str, dtype: torch.dtype):
    if dtype in (torch.float64, torch.complex128):
        skip_if_metal(device, "float64 is not supported on Apple GPUs")


def _check(actual: torch.Tensor, expected: torch.Tensor):
    actual = actual.cpu()
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    if expected.numel() == 0:
        return
    eps = 2e-6 if expected.dtype in (torch.float32, torch.complex64) else 1e-13
    scale = max(expected.abs().max().item(), 1.0)
    torch.testing.assert_close(actual, expected, rtol=eps * 10, atol=eps * scale)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("n", SIZES)
def test_fft_ifft(mojo_gpu: str, dtype: torch.dtype, n: int):
    _skip64(mojo_gpu, dtype)
    x = torch.randn(3, n, dtype=dtype)
    d = x.to(mojo_gpu)
    for norm in NORMS:
        _check(torch.fft.fft(d, norm=norm), torch.fft.fft(x, norm=norm))
        _check(torch.fft.ifft(d, norm=norm), torch.fft.ifft(x, norm=norm))


def test_fft_large(mojo_gpu: str):
    x = torch.randn(2, 1 << 16, dtype=torch.complex64)
    _check(torch.fft.fft(x.to(mojo_gpu)), torch.fft.fft(x))
    y = torch.randn(1, 65537, dtype=torch.complex64)  # a prime past 2**16
    _check(
        torch.fft.ifft(y.to(mojo_gpu), norm="ortho"), torch.fft.ifft(y, norm="ortho")
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n", [1, 2, 5, 16, 97, 100, 210])
def test_real_transforms(mojo_gpu: str, dtype: torch.dtype, n: int):
    _skip64(mojo_gpu, dtype)
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    r = torch.randn(4, n, dtype=dtype)
    c = torch.randn(4, n, dtype=cdtype)
    for norm in NORMS:
        _check(torch.fft.rfft(r.to(mojo_gpu), norm=norm), torch.fft.rfft(r, norm=norm))
        _check(
            torch.fft.ihfft(r.to(mojo_gpu), norm=norm), torch.fft.ihfft(r, norm=norm)
        )
        spec = torch.fft.rfft(r)
        _check(
            torch.fft.irfft(spec.to(mojo_gpu), n=n, norm=norm),
            torch.fft.irfft(spec, n=n, norm=norm),
        )
        if n > 1:
            # a non-Hermitian input: the imaginary parts of the zero and
            # Nyquist bins are ignored, as a c2r transform does
            _check(
                torch.fft.irfft(c.to(mojo_gpu), norm=norm),
                torch.fft.irfft(c, norm=norm),
            )
            _check(
                torch.fft.hfft(c.to(mojo_gpu), norm=norm), torch.fft.hfft(c, norm=norm)
            )


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_multi_dim(mojo_gpu: str, dtype: torch.dtype):
    _skip64(mojo_gpu, dtype)
    rdtype = torch.float32 if dtype == torch.complex64 else torch.float64
    x = torch.randn(4, 6, 10, dtype=dtype)
    d = x.to(mojo_gpu)
    _check(torch.fft.fft2(d), torch.fft.fft2(x))
    _check(torch.fft.fftn(d), torch.fft.fftn(x))
    _check(
        torch.fft.ifftn(d, dim=(0, 2), norm="ortho"),
        torch.fft.ifftn(x, dim=(0, 2), norm="ortho"),
    )
    _check(torch.fft.fft(d, dim=0), torch.fft.fft(x, dim=0))
    r = torch.randn(5, 6, 7, dtype=rdtype)
    _check(torch.fft.rfftn(r.to(mojo_gpu)), torch.fft.rfftn(r))
    _check(
        torch.fft.rfft2(r.to(mojo_gpu), norm="forward"),
        torch.fft.rfft2(r, norm="forward"),
    )
    c = torch.randn(5, 6, 4, dtype=dtype)
    _check(torch.fft.irfftn(c.to(mojo_gpu)), torch.fft.irfftn(c))
    _check(torch.fft.irfft2(c.to(mojo_gpu), s=(6, 9)), torch.fft.irfft2(c, s=(6, 9)))
    _check(torch.fft.hfft2(c.to(mojo_gpu)), torch.fft.hfft2(c))
    _check(torch.fft.ihfftn(r.to(mojo_gpu)), torch.fft.ihfftn(r))


def test_non_contiguous_and_padding(mojo_gpu: str):
    x = torch.randn(6, 8, 10, dtype=torch.complex64)
    d = x.to(mojo_gpu)
    _check(
        torch.fft.fft(d.transpose(0, 2), dim=1), torch.fft.fft(x.transpose(0, 2), dim=1)
    )
    _check(torch.fft.fft(d[:, ::2], dim=1), torch.fft.fft(x[:, ::2], dim=1))
    _check(torch.fft.fft(d, n=13), torch.fft.fft(x, n=13))
    _check(torch.fft.fft(d, n=4), torch.fft.fft(x, n=4))
    r = torch.randn(10, 7)
    _check(torch.fft.rfft(r.to(mojo_gpu).t(), n=12), torch.fft.rfft(r.t(), n=12))
    _check(torch.fft.fftshift(torch.fft.fft(d)), torch.fft.fftshift(torch.fft.fft(x)))
    _check(torch.fft.ifftshift(d, dim=(0, 1)), torch.fft.ifftshift(x, dim=(0, 1)))
    # integer / bool inputs promote to float in torch.fft
    i = torch.randint(-3, 3, (4, 5))
    _check(torch.fft.fft(i.to(mojo_gpu)), torch.fft.fft(i))


def test_out(mojo_gpu: str):
    x = torch.randn(3, 10, dtype=torch.complex64)
    out = torch.empty(0, dtype=torch.complex64, device=mojo_gpu)
    torch.fft.fft(x.to(mojo_gpu), out=out)
    _check(out, torch.fft.fft(x))
    r = torch.randn(3, 10)
    out = torch.empty(0, dtype=torch.complex64, device=mojo_gpu)
    torch.fft.rfft(r.to(mojo_gpu), out=out)
    _check(out, torch.fft.rfft(r))
    out = torch.empty(0, device=mojo_gpu)
    torch.fft.irfft(x.to(mojo_gpu), out=out)
    _check(out, torch.fft.irfft(x))


def test_runs_natively(mojo_gpu: str):
    native.op_counting(True)
    native.op_counts_reset()
    x = torch.randn(8, dtype=torch.complex64, device=mojo_gpu)
    torch.fft.fft(x)
    torch.fft.irfft(x)
    torch.fft.rfft(x.real.contiguous())
    assert native.op_count("aten::_fft_c2c") == 1
    assert native.op_count("aten::_fft_c2r") == 1
    assert native.op_count("aten::_fft_r2c") == 1


def test_empty_batch(mojo_gpu: str):
    # An empty result, deliberately: CUDA raises CUFFT_INVALID_SIZE here,
    # a cuFFT plan limitation rather than torch semantics. (CPU torch's MKL
    # path itself refuses an empty batch.)
    x = torch.randn(0, 8, dtype=torch.complex64)
    got = torch.fft.fft(x.to(mojo_gpu))
    assert got.shape == (0, 8) and got.dtype == torch.complex64


def test_half_declines(mojo_gpu: str):
    with pytest.raises(NotImplementedError):
        torch.ops.aten._fft_r2c(torch.randn(8).half().to(mojo_gpu), [0], 0, True)


def test_c2r_input_sizes(mojo_gpu: str):
    """The one-sided half (last_dim_size // 2 + 1 entries) is read in place;
    a shorter input is zero-padded, a longer one's extra entries ignored --
    no size makes it read out of bounds. CUDA accepts these; for a longer
    input its result is not a defined transform (it matches no reading of
    the input), so only the exact and shorter sizes are compared."""
    c = torch.randn(4, 9, dtype=torch.complex64)
    want = torch.ops.aten._fft_c2r(c, [0, 1], 2, 16)
    _check(torch.ops.aten._fft_c2r(c.to(mojo_gpu), [0, 1], 2, 16), want)
    ones = torch.ones(4, dtype=torch.complex64)
    got = torch.ops.aten._fft_c2r(ones.to(mojo_gpu), [0], 0, 4).cpu()
    assert torch.equal(got, torch.tensor([4.0, 0.0, 0.0, 0.0]))  # stock CUDA's
    got = torch.ops.aten._fft_c2r(c.to(mojo_gpu), [0, 1], 2, 4)
    assert got.shape == (4, 4)


@pytest.mark.parametrize("n", [4, 8, 16, 64, 17, 97, 127])
def test_fft_of_an_infinity(mojo_gpu: str, n: int):
    """A single infinite input at index 0: every bin is inf * w^0 = inf +
    0i. The unit-twiddle and quarter-turn terms are a copy and a sign rather
    than a multiply, and primes up to 127 are direct DFT passes, so no NaN
    appears. (CPU agrees for 4..64 and 17 but goes through Bluestein, all
    NaN, at 97 and 127; stock CUDA agrees at 17..127 but has NaNs at the
    powers of two: 6 of 8 bins at n = 8, inf + NaN i at bins 1-3 and 5-7.
    That is an artifact of cuFFT's private kernels -- a multiply by a unit
    twiddle -- not something its results let us model in general, so the
    exact value is kept here; see test_fft_of_an_infinity_quarter_turns.)"""
    x = torch.zeros(n, dtype=torch.complex64)
    x[0] = float("inf")
    got = torch.fft.fft(x.to(mojo_gpu)).cpu()
    assert torch.equal(got, torch.full((n,), complex(float("inf"), 0.0)))


_INF = float("inf")
_NAN = float("nan")
# torch.fft.fft of zeros(16) with x[3] = inf on stock CUDA (torch 2.11+cu130,
# H100); CPU torch gives the same
_CUDA_INF3_N16 = [
    (_INF, 0.0), (_INF, -_INF), (-_INF, -_INF), (-_INF, _INF),
    (0.0, _INF), (_INF, _INF), (_INF, -_INF), (-_INF, -_INF),
    (-_INF, 0.0), (-_INF, _INF), (_INF, _INF), (_INF, -_INF),
    (0.0, -_INF), (-_INF, -_INF), (-_INF, _INF), (_INF, _INF),
]  # fmt: skip


def test_fft_of_an_infinity_quarter_turns(mojo_gpu: str):
    """A twiddle between passes that is a quarter turn (-i, -1, +i) is a
    sign and a swap, not a multiply by a table entry with a zero part, so an
    infinity stays an infinity. n = 16 (passes of 8 then 2), inf at index 4:
    every twiddle on its path is a quarter turn, the result is exact (CPU
    torch agrees; stock CUDA has NaNs in 12 of the 16 bins). Inf at index 3:
    the even bins match stock CUDA bin for bin (bin 4 was NaN + inf i before
    quarter turns were special-cased); the odd bins carry the inf - inf of
    two successive non-quarter rotations of an infinite complex value, which
    a two-pass transform cannot avoid -- cuFFT does n = 16 in one pass."""
    x = torch.zeros(16, dtype=torch.complex64)
    x[4] = _INF
    got = torch.fft.fft(x.to(mojo_gpu)).cpu()
    assert torch.equal(got, torch.fft.fft(x))
    x = torch.zeros(16, dtype=torch.complex64)
    x[3] = _INF
    got = torch.fft.fft(x.to(mojo_gpu)).cpu()
    cuda = torch.tensor([complex(*v) for v in _CUDA_INF3_N16])
    assert torch.equal(got[0::2], cuda[0::2])
    assert torch.isinf(got[1::2]).all()


@pytest.mark.parametrize("n", [8, 9, 130, 131, 262, 255])
def test_irfft_drops_the_imaginary_dc_and_nyquist(mojo_gpu: str, n: int):
    """The zero (and, for an even n, the Nyquist) entry of a Hermitian input
    is its own mirror: a c2r transform reads only its real part, as
    CPU/pocketfft does (cuFFT drops it only up to rounding). A huge imaginary part there must not leak through rounding (it did
    on the Bluestein path: n = 131 gave ~1e11 instead of 1/131)."""
    m = n // 2 + 1
    x = torch.zeros(m, dtype=torch.complex64)
    x[0] = torch.tensor(complex(1.0, 1e20))
    if n % 2 == 0:
        x[m - 1] = torch.tensor(complex(0.0, 1e20))
    got = torch.fft.irfft(x.to(mojo_gpu), n=n).cpu()
    torch.testing.assert_close(got, torch.full((n,), 1.0 / n), rtol=1e-5, atol=1e-7)


def test_out_validation(mojo_gpu: str):
    """The out= overloads follow cuda/SpectralOps.cpp: the generated
    wrapper's device check first, then at::mul_out into the out, a
    TensorIterator: the out's internal overlap is checked as given (before
    any resize), and the result must be castable to its dtype."""
    x = torch.arange(8.0)
    xd = x.to(mojo_gpu)
    c = x.to(torch.complex64)
    for fn, arg, out in (
        (torch.fft.fft, c, torch.empty(8, dtype=torch.complex64)),
        (torch.fft.rfft, x, torch.empty(5, dtype=torch.complex64)),
        (torch.fft.irfft, c, torch.empty(14)),
    ):
        with pytest.raises(RuntimeError, match="same device"):
            fn(arg.to(mojo_gpu), out=out)
    with pytest.raises(RuntimeError, match="same device"):
        torch.ops.aten._fft_c2c.out(
            c, [0], 0, True, out=torch.empty(8, dtype=torch.complex64, device=mojo_gpu)
        )
    for size in (5, 6):  # same shape (no resize) and another shape
        out = torch.empty(1, dtype=torch.complex64, device=mojo_gpu).expand(size)
        with pytest.raises(RuntimeError, match="unsupported operation"):
            torch.fft.rfft(xd, out=out)
    out = torch.empty(1, dtype=torch.complex64, device=mojo_gpu).expand(8)
    with pytest.raises(RuntimeError, match="unsupported operation"):
        torch.fft.fft(c.to(mojo_gpu), out=out)
    with pytest.raises(RuntimeError, match="can't be cast"):
        torch.ops.aten._fft_c2c.out(
            c.to(mojo_gpu), [0], 0, True, out=torch.empty(8, device=mojo_gpu)
        )
    # a wider out, and an out that is the input itself, are fine
    out = torch.empty(8, dtype=torch.complex128, device=mojo_gpu)
    torch.fft.fft(c.to(mojo_gpu), out=out)
    torch.testing.assert_close(
        out.cpu(), torch.fft.fft(c).to(torch.complex128), rtol=1e-6, atol=1e-5
    )
    z = c.to(mojo_gpu)
    torch.fft.fft(z, out=z)
    _check(z, torch.fft.fft(c))


def test_table_cache_byte_bound(mojo_gpu: str):
    """Bluestein lengths whose chirp tables together pass the cache's byte
    bound: the ones past it are uploaded per call, still correct."""
    for n in (131 * 8000, 131 * 8001, 131 * 8002, 131 * 8003, 131 * 8000):
        x = torch.randn(n, dtype=torch.complex64)
        _check(torch.fft.fft(x.to(mojo_gpu)), torch.fft.fft(x))


def test_fft_nan_stays_local(mojo_gpu: str):
    x = torch.randn(4, 16, dtype=torch.complex64)
    x[1, 3] = float("nan")
    got = torch.fft.fft(x.to(mojo_gpu)).cpu()
    assert torch.isnan(got[1]).all()
    _check(got[[0, 2, 3]], torch.fft.fft(x[[0, 2, 3]]))


def test_concurrent_ffts(mojo_gpu: str):
    """FFTs of distinct lengths (each with its own cached twiddle table) from
    several threads at once: the shim serializes the op bodies."""
    sizes = [97, 210, 256, 1009, 12, 30, 64, 101]
    inputs = [torch.randn(3, n, dtype=torch.complex64) for n in sizes]
    results: dict[int, torch.Tensor] = {}
    errors: list[BaseException] = []

    def work(i: int):
        try:
            for _ in range(5):
                results[i] = torch.fft.fft(inputs[i].to(mojo_gpu)).cpu()
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(sizes))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    for i, x in enumerate(inputs):
        _check(results[i], torch.fft.fft(x))


@pytest.mark.parametrize("n", [131, 262, 1009])
def test_fft_of_an_infinity_bluestein(mojo_gpu: str, n: int):
    """Past cuFFT's own Bluestein threshold (a prime factor above 127) the
    chirp convolution spreads inf * 0 = NaN through the row, here as on
    stock CUDA (checked bin for bin against torch 2.11+cu130: every bin
    NaN). Pinned so a change of this behaviour is a decision."""
    x = torch.zeros(n, dtype=torch.complex64)
    x[0] = float("inf")
    got = torch.fft.fft(x.to(mojo_gpu)).cpu()
    assert torch.isnan(got).all()


@pytest.mark.parametrize("n", [131, 262, 1009])
def test_bluestein_keeps_huge_inputs_finite(mojo_gpu: str, n: int):
    """The chirp convolution is normalized inside (1/m folded into the
    chirp's spectrum): an input near float32's max stays finite, as on
    CUDA. The impulse's transform is 1e36 in every bin."""
    x = torch.zeros(n, dtype=torch.complex64)
    x[0] = 1e36
    got = torch.fft.fft(x.to(mojo_gpu)).cpu()
    assert torch.isfinite(got).all()
    torch.testing.assert_close(
        got, torch.full((n,), 1e36, dtype=torch.complex64), rtol=1e-5, atol=0
    )
    y = torch.randn(n, dtype=torch.complex64) * 1e35
    torch.testing.assert_close(
        torch.fft.fft(y.to(mojo_gpu)).cpu(), torch.fft.fft(y), rtol=1e-4, atol=1e33
    )


_LEAK_SCRIPT = textwrap.dedent("""
    import torch
    import torch_mojo_backend
    torch_mojo_backend.register_mojo_devices()
    dev = torch.accelerator.current_accelerator()
    # after 1,049,310 fills most of the table budget, 1,048,131's chirp
    # still fits but its padded chirp does not: a partial hit every call
    big = torch.ones(1_049_310, dtype=torch.complex64, device=dev)
    torch.fft.fft(big)
    del big
    x = torch.ones(1_048_131, dtype=torch.complex64, device=dev)
    torch.fft.fft(x)
    torch.accelerator.synchronize()
    before = torch.accelerator.memory_allocated()
    for _ in range(5):
        torch.fft.fft(x)
    torch.accelerator.synchronize()
    after = torch.accelerator.memory_allocated()
    assert after == before, (before, after)
    print("no growth", flush=True)
""")


def test_bluestein_tables_do_not_leak(mojo_gpu: str):
    """A partial cache hit (the chirp cached, the padded chirp not, or the
    reverse) must not leak the cached reference it re-uploads over. Run in
    a fresh process: the table cache is process-wide, and the partial hit
    needs it empty at the start (any earlier test may have filled it)."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONHOME", "PYTHONEXECUTABLE")
    }
    proc = subprocess.run(
        [sys.executable, "-c", _LEAK_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "no growth" in proc.stdout


_STREAM_SCRIPT = textwrap.dedent("""
    import torch
    import torch_mojo_backend
    torch_mojo_backend.register_mojo_devices()
    dev = torch.accelerator.current_accelerator()
    n = 1_000_003  # prime: Bluestein (chirp tables) and Stockham twiddles
    x = torch.randn(n, dtype=torch.complex64)
    want = torch.fft.fft(x)
    d = x.to(dev)
    s1 = torch.Stream(device=dev)
    s2 = torch.Stream(device=dev)
    busy = torch.randn(2048, 2048, device=dev)
    torch.accelerator.synchronize()
    with s1:
        # work queued ahead on s1, so a copy ordered on s1 lands late; this
        # is the process's first transform, so it uploads every table
        for _ in range(20):
            busy = busy @ busy
            busy = busy / busy.abs().amax()
        a = torch.fft.fft(d)
    with s2:
        # the tables just cached, read from another stream
        b = torch.fft.fft(d)
    torch.accelerator.synchronize()
    scale = max(want.abs().max().item(), 1.0)
    for got in (a, b):
        torch.testing.assert_close(
            got.cpu(), want, rtol=2e-5, atol=2e-6 * scale
        )
    print("both complete", flush=True)
""")


def test_cached_tables_are_complete_for_another_stream(mojo_gpu: str):
    """A table uploaded on one stream is cached; a transform on another
    stream right after must read it complete (the upload finishes before
    the table is published). Run in a fresh process: the table cache is
    process-wide, and the first transform must be the one that uploads
    (an earlier test, e.g. the byte-bound one, may already have filled it)."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONHOME", "PYTHONEXECUTABLE")
    }
    proc = subprocess.run(
        [sys.executable, "-c", _STREAM_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "both complete" in proc.stdout
