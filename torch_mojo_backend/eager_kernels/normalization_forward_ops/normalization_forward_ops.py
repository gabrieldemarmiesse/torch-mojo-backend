from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class NormalizationForwardExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path(
        "normalization_forward_ops/normalization_forward_ops.mojo"
    )
