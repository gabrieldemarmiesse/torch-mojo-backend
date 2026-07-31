from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class NNExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path("nn_ops/nn_ops.mojo")
