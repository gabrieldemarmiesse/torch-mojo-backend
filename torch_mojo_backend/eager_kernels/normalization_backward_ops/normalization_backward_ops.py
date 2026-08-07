from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class NormalizationBackwardExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path(
        "normalization_backward_ops/normalization_backward_ops.mojo"
    )
