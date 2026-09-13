"""Registration of the `mojo` device (see docs/native_backend.md)."""

import threading

import torch

from torch_mojo_backend import native
from torch_mojo_backend.distributed import register_distributed_backend
from torch_mojo_backend.mojo_device.hip_peer import warn_if_gpu_torch_on_hip
from torch_mojo_backend.monkeypatching import (
    fix_batch_isend_irecv_for_python_process_groups,
    fix_privateuse1_dlpack_device_type,
)
from torch_mojo_backend.native import device_module
from torch_mojo_backend.triton_driver import install_triton_hook

_lock = threading.Lock()
_torch_registered = False  # torch's own registrations: not repeatable
_registered = False  # everything, so a failed attempt can be retried


def register_mojo_devices():
    """Enable the mojo device globally: build/load the two shims once and
    register the backend, the `torch.mojo` module and the `.mojo()` helpers.
    Idempotent; a failure (a build that did not go through) leaves it
    retryable."""
    global _registered, _torch_registered
    if _registered:
        return
    with _lock:
        if _registered:
            return
        if not _torch_registered:
            # Module._apply otherwise replaces a shared CPU Parameter
            # independently in each child module; swapping preserves tied
            # weights (GPT-2's token embedding and lm_head) as one Parameter
            # and one allocation.
            torch.__future__.set_swap_module_params_on_conversion(True)
            torch.utils.rename_privateuse1_backend("mojo")
            torch._register_device_module("mojo", device_module)
            torch.utils.generate_methods_for_privateuse1_backend()
            _torch_registered = True
        # each of these is idempotent, so a retry may run them again
        fix_privateuse1_dlpack_device_type()
        fix_batch_isend_irecv_for_python_process_groups()
        native.register()
        install_triton_hook()
        warn_if_gpu_torch_on_hip()
        register_distributed_backend()
        _registered = True
