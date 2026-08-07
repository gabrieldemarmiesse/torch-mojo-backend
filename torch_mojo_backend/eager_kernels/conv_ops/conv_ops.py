from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class ConvExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path("conv_ops/conv_ops.mojo")
