from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class ElementwiseExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path("elementwise_ops/elementwise_ops.mojo")
