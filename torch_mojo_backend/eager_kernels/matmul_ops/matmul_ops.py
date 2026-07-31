from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class MatmulExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path("matmul_ops/matmul_ops.mojo")
