from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class OptimizerExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path("optimizer_ops/optimizer_ops.mojo")
