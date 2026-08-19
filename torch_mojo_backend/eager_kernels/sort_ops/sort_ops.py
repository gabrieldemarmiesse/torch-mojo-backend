from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class SortExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path("sort_ops/sort_ops.mojo")
