import torch_mojo_backend._mojo_import_path  # noqa: F401 -- extends MAX's Mojo import path with the Mojo source root before `max` reads it
import torch_mojo_backend._ptxas  # noqa: F401 -- sets MAX's ptxas default before `max` is imported below
from torch_mojo_backend.custom_torch_ops_in_mojo.torch_custom_ops import (
    make_torch_op_from_mojo,
)
from torch_mojo_backend.mojo_device.log_aten_calls import log_aten_calls
from torch_mojo_backend.mojo_device.register import register_mojo_devices
from torch_mojo_backend.monkeypatching import preserve_dynamo_stream_contexts
from torch_mojo_backend.torch_compile_backend.compiler import (
    MAPPING_TORCH_ATEN_TO_MOJO,
    MojoCompilerError,
    get_accelerators,
    mojo_backend,
)

preserve_dynamo_stream_contexts()

__all__ = [
    "mojo_backend",
    "get_accelerators",
    "MAPPING_TORCH_ATEN_TO_MOJO",
    "MojoCompilerError",
    "register_mojo_devices",
    "make_torch_op_from_mojo",
    "log_aten_calls",
]
