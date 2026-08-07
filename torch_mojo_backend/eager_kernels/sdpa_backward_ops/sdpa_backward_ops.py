from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class SDPABackwardExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path("sdpa_backward_ops/sdpa_backward_ops.mojo")
