from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class LogicExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path("logic_ops/logic_ops.mojo")
