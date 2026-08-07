from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class SoftmaxBackwardExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path("softmax_backward_ops/softmax_backward_ops.mojo")
