from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class ActivationForwardExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path(
        "activation_forward_ops/activation_forward_ops.mojo"
    )
