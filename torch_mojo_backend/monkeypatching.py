"""Every monkeypatch this project applies, in one place.

A monkeypatch here means replacing or mutating, at runtime, an attribute of
a module or class we do not own -- mostly PyTorch internals that have no
extension point yet. Each patch is one function with a docstring saying what
upstream lacks, so that the patch can be turned into an upstream PR and
deleted from here. Nothing else in the package may patch a third-party
module; keep new patches in this file
(``tests/test_monkeypatching_is_centralized.py`` enforces that for
``torch``-rooted assignments).

Every patch installer is called by ``register.register_mojo_devices``, never
at import time. Official registration APIs (``torch.library.impl``, the
PrivateUse1 backend module, ``torch.__future__`` toggles) are not
monkeypatches and stay in ``mojo_device/register.py``.
"""

import functools
import os
import sys
from functools import wraps
from typing import TYPE_CHECKING

import torch
import torch.distributed.distributed_c10d as c10d
import torch.utils._triton

if TYPE_CHECKING:
    from triton.backends.driver import DriverBase


def fix_privateuse1_dlpack_device_type():
    """`Tensor.__dlpack_device__` doesn't recognize a *renamed* PrivateUse1
    backend.

    ``torch/_tensor.py``'s ``Tensor.__dlpack_device__`` maps a PrivateUse1
    tensor to DLPack's ``kDLExtDev`` by comparing ``self.device.type``
    against the string literal ``"privateuse1"`` -- so after
    ``torch.utils.rename_privateuse1_backend("mojo")`` it never matches, and
    every mojo tensor's ``__dlpack_device__()`` raises ``ValueError("Unknown
    device type mojo for Dlpack")``. Two other call sites in that very same
    file (the ``__cuda_array_interface__`` gate) correctly compare against
    ``torch._C._get_privateuse1_backend_name()`` instead of the literal;
    this one method just didn't get the memo.

    ``Tensor.__dlpack__`` itself (the capsule export) is unaffected --
    ATen's C++ DLConvertor keys off the ``DeviceType`` enum, not the
    Python-visible name -- so only the device-query half needs patching.
    ``torch_compile_backend/compiler.py``'s ``fast_from_dlpack`` routes
    around this bug for its own zero-copy exchange (it never calls
    ``__dlpack_device__``), but plain ``torch.utils.dlpack`` /
    ``max.driver.Buffer.from_dlpack(t)`` usage elsewhere (user code,
    ``test_compile_mojo_device.py``) goes through the single-arg DLPack
    protocol, which calls ``__dlpack_device__()`` first and needs this fix.
    """
    original = torch.Tensor.__dlpack_device__
    if getattr(original, "_torch_mojo_backend", False):
        return

    from torch.utils.dlpack import (  # noqa: PLC0415 -- mirrors the private import inside the method being patched
        DLDeviceType,
    )

    @wraps(original)
    def __dlpack_device__(self: torch.Tensor) -> tuple[int, int]:
        if self.device.type == torch._C._get_privateuse1_backend_name():
            index = self.device.index if self.device.index is not None else 0
            return (DLDeviceType.kDLExtDev, index)
        return original(self)

    __dlpack_device__._torch_mojo_backend = True  # ty: ignore[unresolved-attribute]
    torch.Tensor.__dlpack_device__ = (  # ty: ignore[invalid-assignment]
        __dlpack_device__
    )


def fix_batch_isend_irecv_for_python_process_groups():
    """`batch_isend_irecv` never coalesces for a Python `ProcessGroup`
    subclass, and a bidirectional exchange then deadlocks.

    ``torch/distributed/distributed_c10d.py``'s ``batch_isend_irecv`` gates
    its NCCL-style coalescing on ``type(group) is ProcessGroup``. The mojo
    backend IS a Python subclass of ``ProcessGroup`` (it has to be: the
    C++ ``Backend`` cannot be implemented in Python), so the check is False
    however capable the backend is, and every operation in the list goes out
    as its own NCCL group. A two-rank exchange then deadlocks on the device:
    each rank's comm stream holds ``[send(->peer), recv(<-peer)]``, and the
    send cannot retire until the peer posts its recv, which sits behind that
    peer's own send. The very next line asks the backend whether it
    ``supports_coalescing``, which is the real question; ``isinstance`` is
    what the type check means. Everything else is torch's own code path,
    called unchanged.
    """
    original = c10d.batch_isend_irecv
    if getattr(original, "_torch_mojo_backend", False):
        return

    @wraps(original)
    def batch_isend_irecv(p2p_op_list: list[c10d.P2POp]) -> list[c10d.Work]:
        c10d._check_p2p_op_list(p2p_op_list)
        group = p2p_op_list[0].group or c10d._get_default_group()
        device = p2p_op_list[0].tensor.device
        coalesces = (
            type(group) is not torch.distributed.ProcessGroup
            and isinstance(group, torch.distributed.ProcessGroup)
            and group._get_backend(device).supports_coalescing
        )
        if not coalesces:
            return original(p2p_op_list)
        with c10d._coalescing_manager(group, device, async_ops=True) as manager:
            for op in p2p_op_list:
                peer = "group_dst" if op.op is c10d.isend else "group_src"
                op.op(op.tensor, group=op.group, tag=op.tag, **{peer: op.group_peer})
        return manager.works

    batch_isend_irecv._torch_mojo_backend = True  # ty: ignore[unresolved-attribute]
    c10d.batch_isend_irecv = batch_isend_irecv  # ty: ignore[invalid-assignment]
    torch.distributed.batch_isend_irecv = batch_isend_irecv  # ty: ignore[invalid-assignment]


def add_mojo_to_the_inductor_gpu_types():
    """Inductor decides "is this an accelerator?" against a hardcoded list.

    ``torch/_inductor/utils.py``'s ``GPU_TYPES = ["cuda", "mps", "xpu",
    "mtia"]`` feeds ``is_gpu()``, and there is no registry for it the way
    there is for the device interface and the codegen backend -- so a
    PrivateUse1 backend reads as a CPU however completely it registers.
    What that costs: no device guard around the generated kernels
    (``device_need_guard``), CPU loop-merging heuristics in the scheduler,
    no alignment-driven input cloning, and the "which device does this
    pointwise op belong to" scan in ``lowering.py`` never picking ours.

    Appending is safe here because ``is_gpu`` reads the module global and
    the two modules that ``from ... import GPU_TYPES`` bind the same list
    object.

    The list is process-wide and ``get_gpu_type()`` asserts at most one of
    its entries is available, so "mojo" and a working ``torch.cuda`` cannot
    both be in it: with both, that assert fires in the autotuning
    subprocess setup and the profiler benchmarking, for CUDA workloads as
    much as for ours. There is no per-graph answer to give it, so this
    patch stands aside on a torch with a working CUDA/ROCm build --
    ``inductor.enable_inductor()`` refuses outright there, and says why.
    """
    if torch.cuda.is_available():
        return
    from torch._inductor import (  # noqa: PLC0415 -- 0.87 s of sympy, measured; every register_mojo_devices() would otherwise pay it
        utils as inductor_utils,
    )

    if "mojo" not in inductor_utils.GPU_TYPES:
        inductor_utils.GPU_TYPES.append("mojo")


def let_has_triton_see_the_mojo_device():
    """`torch.utils._triton.has_triton` asks a hardcoded device dict.

    Its ``triton_supported_devices = {"cuda", "xpu", "cpu", "mtia"}`` is a
    local of the function, so a PrivateUse1 backend cannot add itself: with
    a CPU torch wheel every entry answers "not available", ``has_triton()``
    is False, and Inductor then refuses to build a Triton scheduling for
    our device (``scheduler.py``'s ``create_backend`` raises
    ``TritonMissing``). The fix upstream is one more entry in that dict,
    fed by the registered device interfaces.

    The rebinding loop is the price of ``from torch.utils._triton import
    has_triton``: a dozen torch modules hold the original by value.
    """
    from torch_mojo_backend.native import (  # noqa: PLC0415 -- `torch.mojo`, imported here so registering the devices does not pull in the native backend
        device_module,
    )

    original = torch.utils._triton.has_triton
    if getattr(original, "_torch_mojo_backend", False):
        return

    @functools.cache
    def has_triton() -> bool:
        return original() or (
            torch.utils._triton.has_triton_package() and device_module.is_available()
        )

    has_triton._torch_mojo_backend = True  # ty: ignore[unresolved-attribute]
    torch.utils._triton.has_triton = has_triton
    for module in list(sys.modules.values()):
        if getattr(module, "has_triton", None) is original:
            module.has_triton = has_triton  # ty: ignore[unresolved-attribute] -- setting an attribute torch bound by value


def register_the_mojo_triton_target(driver: "type[DriverBase]"):
    """Triton picks a compiler backend by the name inside the target, and
    Inductor puts the torch device type there.

    ``torch/_inductor/runtime/triton_heuristics.py`` builds
    ``GPUTarget(compile_meta["device_type"], ...)`` from
    ``DeviceProperties.type``, which is the torch device type -- "mojo".
    ``triton.compiler.compiler.make_backend`` then asks every registered
    backend ``supports_target(target)``, and NVIDIA's answers only to
    "cuda", so compilation dies with "0 compatible backends". Registering
    an alias of the NVIDIA backend under our name is what an out-of-tree
    Triton backend would do through the ``triton.backends`` entry point;
    only ``supports_target`` differs, the target's ``arch`` (the compute
    capability) is what the backend actually compiles against.

    The entry also carries `driver` (`triton_driver.driver_class()`), because
    the same dict is what `torch/_inductor/runtime/triton_helpers.py`'s
    ``set_driver_to_gpu`` -- run at the import of every generated kernel
    module -- scans for a backend whose driver ``is_active()``. NVIDIA's says
    no (it asks ``torch.cuda.is_available()``), so without an entry of ours
    the import of the generated module raises "Could not find an active GPU
    backend".
    """
    import triton.backends  # noqa: PLC0415 -- triton is optional
    from triton.backends.nvidia.compiler import (  # noqa: PLC0415 -- triton is optional
        CUDABackend,
    )

    if "mojo" in triton.backends.backends:
        return

    class MojoBackend(CUDABackend):
        @staticmethod
        def supports_target(target: object) -> bool:
            return getattr(target, "backend", None) == "mojo"

    triton.backends.backends["mojo"] = triton.backends.Backend(
        compiler=MojoBackend, driver=driver
    )


def compile_inductor_kernels_in_process():
    """An Inductor compile worker is a fresh interpreter that imports only
    torch, and nothing can teach it about an out-of-tree device.

    ``torch/_inductor/compile_worker/__main__.py`` (started as a subprocess
    by default) imports torch and triton and nothing else, so our Triton
    backend registration is absent there and importing a generated kernel
    module dies inside ``torch/_inductor/runtime/triton_helpers.py``'s
    ``set_driver_to_gpu`` with "Could not find an active GPU backend".
    Upstream has no worker-startup import hook, so the compiles run in this
    process instead; on the graphs measured here that is also the faster of
    the two, the pool costing more to start than these compiles take.

    ``TORCHINDUCTOR_WORKER_START=fork`` is the other way out -- a forked
    worker inherits the registration and works -- and is left to the user,
    forking a process that already holds a GPU context being its own risk.
    An explicit ``TORCHINDUCTOR_COMPILE_THREADS`` also stands.
    """
    from torch._inductor import (  # noqa: PLC0415 -- see add_mojo_to_the_inductor_gpu_types
        config as inductor_config,
    )

    if (
        inductor_config.worker_start_method != "fork"
        and "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ
    ):
        inductor_config.compile_threads = 1
