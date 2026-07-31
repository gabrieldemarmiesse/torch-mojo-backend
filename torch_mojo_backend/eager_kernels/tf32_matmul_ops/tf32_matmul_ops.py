from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class TF32MatmulExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path("tf32_matmul_ops/tf32_matmul_ops.mojo")
