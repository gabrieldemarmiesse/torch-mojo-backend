from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class ActivationBackwardExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path(
        "activation_backward_ops/activation_backward_ops.mojo"
    )
