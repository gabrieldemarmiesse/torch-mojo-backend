import os

# Off by default: beartype's wrapper frames cost real per-call CPU time in
# eager hot paths. The test suite opts in via tests/conftest.py.
if os.environ.get("TORCH_MOJO_BACKEND_BEARTYPE", "0") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()


# Before anything that can import max.nn: pin the mojo runtime by loading a
# mojo extension first when one is cached (see preload_tensor_holder_if_cached
# and upstream_issues/modular-5-max-nn-import-breaks-mojo-extension-load.md).
from torch_mojo_backend.eager_kernels import preload_tensor_holder_if_cached

preload_tensor_holder_if_cached()

from torch_mojo_backend.custom_torch_ops_in_mojo.torch_custom_ops import (
    make_torch_op_from_mojo,
)
from torch_mojo_backend.mojo_device.log_aten_calls import log_aten_calls
from torch_mojo_backend.mojo_device.register import register_mojo_devices
from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor
from torch_mojo_backend.torch_compile_backend.compiler import (
    MAPPING_TORCH_ATEN_TO_MOJO,
    MojoCompilerError,
    get_accelerators,
    mojo_backend,
)

__all__ = [
    "mojo_backend",
    "get_accelerators",
    "MAPPING_TORCH_ATEN_TO_MOJO",
    "MojoCompilerError",
    "register_mojo_devices",
    "make_torch_op_from_mojo",
    "TorchMojoTensor",
    "log_aten_calls",
]
