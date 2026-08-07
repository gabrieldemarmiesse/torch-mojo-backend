from pathlib import Path
from typing import ClassVar

from torch_mojo_backend.eager_kernels import MojoFileExtension


class FlashAttentionExtension(MojoFileExtension):
    MOJO_FILE: ClassVar[Path] = Path("flash_attention_ops/flash_attention_ops.mojo")
