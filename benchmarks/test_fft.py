"""FFTs: the kernels of tmb/kernels/fft (mixed-radix Stockham, direct
passes for primes up to 127, Bluestein beyond), against stock PyTorch's
(cuFFT on CUDA).

Shapes: a batch of power-of-two rows (the Stockham regime), a mixed-radix
size, a prime (Bluestein) and one large single transform; the 2-D case
transforms two dims of an image-like batch.
"""

from __future__ import annotations

import pytest
import torch
from bench_lib.cases import both
from bench_lib.check import Bench
from bench_lib.hw import Hardware

SHAPES: dict[str, tuple[int, ...]] = {
    "B_256x1024": (256, 1024),
    "M_128x210": (128, 210),
    "P_64x1009": (64, 1009),
    "L_1x65536": (1, 65536),
}
SHAPES_2D: dict[str, tuple[int, ...]] = {
    "I_16x128x128": (16, 128, 128),
    "A_8x60x97": (8, 60, 97),
}

COVERS: dict[str, str] = {
    "aten::_fft_c2c": "test_fft",
    "aten::_fft_r2c": "test_rfft",
    "aten::_fft_c2r": "test_irfft",
}

_OUT = "out= plumbing (resize, copy_) over the functional op this module measures"
SKIPPED: dict[str, str] = {
    "aten::_fft_c2c.out": _OUT,
    "aten::_fft_r2c.out": _OUT,
    "aten::_fft_c2r.out": _OUT,
}


def _flops(shape: tuple[int, ...], dims: int = 1) -> float:
    """The conventional 5 n log2(n) per complex transform of length n."""
    n = 1
    for d in shape[-dims:]:
        n *= d
    batch = 1
    for d in shape[:-dims]:
        batch *= d
    return 5.0 * batch * n * max(n.bit_length() - 1, 1)


@pytest.mark.parametrize("dtype_id", ("c64",))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("_fft_c2c")
def test_fft(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = SHAPES[shape_id]
    x_ref, x_our = both(torch.randn(shape, dtype=torch.complex64), hw, mojo_device)
    bench.run(
        lambda: torch.fft.fft(x_ref), lambda: torch.fft.fft(x_our), flops=_flops(shape)
    )


@pytest.mark.parametrize("dtype_id", ("c64",))
@pytest.mark.parametrize("shape_id", SHAPES_2D)
@pytest.mark.bench_op("_fft_c2c")
def test_fft2(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = SHAPES_2D[shape_id]
    x_ref, x_our = both(torch.randn(shape, dtype=torch.complex64), hw, mojo_device)
    bench.run(
        lambda: torch.fft.fft2(x_ref),
        lambda: torch.fft.fft2(x_our),
        flops=_flops(shape, 2),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("_fft_r2c")
def test_rfft(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = SHAPES[shape_id]
    x_ref, x_our = both(torch.randn(shape), hw, mojo_device)
    bench.run(
        lambda: torch.fft.rfft(x_ref),
        lambda: torch.fft.rfft(x_our),
        flops=_flops(shape) / 2,
    )


@pytest.mark.parametrize("dtype_id", ("c64",))
@pytest.mark.parametrize("shape_id", SHAPES)
@pytest.mark.bench_op("_fft_c2r")
def test_irfft(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = SHAPES[shape_id]
    half = (*shape[:-1], shape[-1] // 2 + 1)
    x_ref, x_our = both(torch.randn(half, dtype=torch.complex64), hw, mojo_device)
    n = shape[-1]
    bench.run(
        lambda: torch.fft.irfft(x_ref, n=n),
        lambda: torch.fft.irfft(x_our, n=n),
        flops=_flops(shape) / 2,
    )
