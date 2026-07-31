from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class DropoutExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path("dropout_ops/dropout_ops.mojo")
