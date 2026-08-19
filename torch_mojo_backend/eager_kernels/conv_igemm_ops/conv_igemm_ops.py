from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class ConvIgemmExtension(MojoFileExtension):
    """sm_90a implicit-GEMM conv2d forward (TMA im2col, no col buffer).

    A separate extension from ``ConvExtension`` on purpose: this one pulls in
    the 16-bit tensor-core GEMM family's sources, which a source checkout or
    wheel may not ship, and the materialized-im2col route must keep working
    (including on CPU) when they are absent.
    """

    MOJO_FILE: ClassVar[Path] = Path("conv_igemm_ops/conv_igemm_ops.mojo")
