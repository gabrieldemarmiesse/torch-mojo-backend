from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class ReductionExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path("reduction_ops/reduction_ops.mojo")
