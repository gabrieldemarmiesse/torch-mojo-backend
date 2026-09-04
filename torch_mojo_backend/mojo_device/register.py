from collections.abc import Callable
from functools import wraps

import torch
from torch.utils.backend_registration import _setup_privateuseone_for_python_backend

from torch_mojo_backend.distributed import register_distributed_backend
from torch_mojo_backend.mojo_device import (
    comm_fence,
    deferred_compile,
    torch_mojo_device_module,
)
from torch_mojo_backend.mojo_device.hip_peer import warn_if_gpu_torch_on_hip
from torch_mojo_backend.mojo_device.mojo_device_aten_ops import _aten_ops_registry
from torch_mojo_backend.mojo_device.mojo_device_autocast import register_autocast_ops
from torch_mojo_backend.mojo_device.mojo_device_autograd import register_autograd_ops
from torch_mojo_backend.monkeypatching import apply_torch_monkeypatches

_registered = False


def _fence_pending_collectives(func: Callable[..., object]) -> Callable[..., object]:
    """Order this op after any collective still in flight on the comm stream.

    Wrapped around every eager op at registration time — the one choke point
    where a default-stream consumer of a collective's buffer can be caught
    (see mojo_device/comm_fence.py). While nothing is pending, which is every
    op of a non-distributed run, the cost is this frame plus a dict
    truthiness test.
    """
    pending = comm_fence.PENDING  # by reference: comm_fence mutates in place
    fence_pending_args = comm_fence.fence_pending_args

    @wraps(func)
    def with_comm_fence(*args: object, **kwargs: object) -> object:
        if pending:
            fence_pending_args(args, kwargs)
        return func(*args, **kwargs)

    return with_comm_fence


def _resolve_overload(op_name: str) -> torch._ops.OpOverload | None:
    """``"aten::add.Tensor"`` -> ``torch.ops.aten.add.Tensor``; a name with no
    overload suffix takes ``.default``. ``None`` when this torch build has no
    such operator, in which case only the library registration is installed
    and the op keeps redispatching through C++.
    """
    namespace, _, rest = op_name.partition("::")
    packet_name, _, overload_name = rest.partition(".")
    try:
        packet = getattr(getattr(torch.ops, namespace), packet_name)
        return getattr(packet, overload_name or "default")
    except AttributeError:
        return None


def register_mojo_devices():
    """Enable the mojo device globally and register all aten ops"""
    global _registered
    if _registered:
        # Already registered
        return

    # Module._apply otherwise replaces a shared CPU Parameter independently in
    # each child module when its converted tensor has a different subclass.
    # Swapping preserves tied weights (for example GPT-2's token embedding and
    # lm_head) as one Parameter and one Mojo allocation.
    torch.__future__.set_swap_module_params_on_conversion(True)

    _setup_privateuseone_for_python_backend(
        "mojo", backend_module=torch_mojo_device_module
    )
    # Everything torch lacks an extension point for lives in one file.
    apply_torch_monkeypatches(torch_mojo_device_module)

    # Register all collected aten operations. The library registration is what
    # a call reaching the backend key from C++ finds -- factories such as
    # `torch.empty(device="mojo")` have no wrapper argument and never pass
    # through `__torch_dispatch__`. The table beside it is the shortcut for
    # calls that did (see deferred_compile.DIRECT_IMPLS).
    for op_name, func in _aten_ops_registry:
        wrapped = _fence_pending_collectives(func)
        torch.library.impl(op_name, "privateuseone")(wrapped)
        overload = _resolve_overload(op_name)
        if overload is not None:
            deferred_compile.DIRECT_IMPLS[overload] = wrapped

    register_autograd_ops()
    register_autocast_ops()
    warn_if_gpu_torch_on_hip()
    register_distributed_backend()

    _registered = True
