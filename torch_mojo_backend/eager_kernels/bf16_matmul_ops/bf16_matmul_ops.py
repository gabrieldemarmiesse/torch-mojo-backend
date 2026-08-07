from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class BF16MatmulExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path("bf16_matmul_ops/bf16_matmul_ops.mojo")
