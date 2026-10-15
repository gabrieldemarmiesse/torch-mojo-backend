"""The complex constructors: torch.complex / torch.polar on the complex
kernel family (tmb/kernels/complex), against stock PyTorch's.
"""

from __future__ import annotations

import pytest
import torch
from bench_lib.cases import both
from bench_lib.check import Bench
from bench_lib.hw import Hardware

SHAPES: dict[str, tuple[int, ...]] = {"B_256x1024": (256, 1024)}

COVERS: dict[str, str] = {"aten::complex": "test_complex", "aten::polar": "test_polar"}

_OUT = "out= plumbing (resize, copy_) over the functional op this module measures"
SKIPPED: dict[str, str] = {"aten::complex.out": _OUT, "aten::polar.out": _OUT}


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", ("B_256x1024",))
@pytest.mark.bench_op("complex")
def test_complex(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = SHAPES[shape_id]
    re_ref, re_our = both(torch.randn(shape), hw, mojo_device)
    im_ref, im_our = both(torch.randn(shape), hw, mojo_device)
    bench.run(
        lambda: torch.complex(re_ref, im_ref),
        lambda: torch.complex(re_our, im_our),
        flops=2.0 * re_ref.numel(),
    )


@pytest.mark.parametrize("dtype_id", ("f32",))
@pytest.mark.parametrize("shape_id", ("B_256x1024",))
@pytest.mark.bench_op("polar")
def test_polar(
    shape_id: str, dtype_id: str, bench: Bench, hw: Hardware, mojo_device: torch.device
):
    shape = SHAPES[shape_id]
    a_ref, a_our = both(torch.rand(shape), hw, mojo_device)
    t_ref, t_our = both(torch.randn(shape), hw, mojo_device)
    bench.run(
        lambda: torch.polar(a_ref, t_ref),
        lambda: torch.polar(a_our, t_our),
        flops=20.0 * a_ref.numel(),
    )
