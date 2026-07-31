from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class DataMovementExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path("data_movement_ops/data_movement_ops.mojo")
