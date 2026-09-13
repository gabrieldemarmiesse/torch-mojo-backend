"""Run Triton kernels on the mojo device.

Triton compiles and launches through its own GPU backend (bundled ptxas,
libcuda / libamdhip64 from the display driver), so it needs no CUDA or ROCm
build of torch; the one thing it asks torch for is "which device and stream
is current", through a driver object. The drivers here answer with the mojo
device: launches land on the mojo current stream's vendor handle, so they are
ordered with our kernels. The vendor device ordinal is the mojo device index
(MAX enumerates accelerators in vendor order, and the *_VISIBLE_DEVICES
variables apply to both).

Triton has one active driver per process and reads the device and the stream
from it *before* it looks at a launch's arguments (`triton/runtime/jit.py`'s
`run`), so whichever driver is installed answers for every Triton launch in
the process -- nothing in the protocol can tell a launch over mojo tensors
from one over CUDA tensors. `register_mojo_devices()` therefore installs this
driver automatically only on a torch with no working CUDA/ROCm build
(`install_triton_hook`), which is the case the feature exists for; with a
vendor build of torch the user asks for it by hand with `enable_triton()`,
and takes the whole process with them. TORCH_MOJO_BACKEND_TRITON=0 turns the
automatic hook off.
"""

from __future__ import annotations

import ctypes
import functools
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from typing import TYPE_CHECKING

import torch

from torch_mojo_backend.native import device_module

if TYPE_CHECKING:
    from types import ModuleType

    from triton.backends.driver import DriverBase, GPUDriver
    from triton.backends.nvidia.driver import CudaUtils

_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
_DRIVER_MODULE = "triton.runtime.driver"


@functools.cache
def accelerator_api() -> str:
    """ "cuda", "hip", "metal" or "cpu": what MAX drives on this machine."""
    from torch_mojo_backend.torch_compile_backend.utils import (  # noqa: PLC0415 -- imports max.driver; keep it off the import path
        get_accelerators,
    )

    for d in get_accelerators():
        api = getattr(d, "api", "")
        if api != "cpu":
            return str(api)
    return "cpu"


@functools.cache
def _libcuda() -> ctypes.CDLL:
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit(0)
    return cuda


@functools.cache
def _device_capability(device: int) -> tuple[int, int]:
    """(major, minor) straight from the CUDA driver API, like Triton's own
    torch-free path; independent of the torch build and Triton version."""
    cuda = _libcuda()
    handle = ctypes.c_int()
    if cuda.cuDeviceGet(ctypes.byref(handle), device) != 0:
        raise RuntimeError(f"cuDeviceGet({device}) failed")
    major, minor = ctypes.c_int(), ctypes.c_int()
    cuda.cuDeviceGetAttribute(
        ctypes.byref(major), _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, handle
    )
    cuda.cuDeviceGetAttribute(
        ctypes.byref(minor), _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, handle
    )
    return (major.value, minor.value)


@functools.cache
def _stream_context(stream: int) -> ctypes.c_void_p:
    """The driver context that owns a stream: MAX streams live in MAX's own
    context (one per device), so the stream itself is what says which.

    Cached on the handle -- a mojo stream keeps its vendor handle for life,
    the assumption `cuda_interop._external_stream` already makes."""
    ctx = ctypes.c_void_p()
    if _libcuda().cuStreamGetCtx(ctypes.c_void_p(stream), ctypes.byref(ctx)) != 0:
        raise RuntimeError("cuStreamGetCtx failed for the mojo stream")
    return ctx


def _push_stream_context(stream: int) -> ctypes.c_void_p | None:
    """Make the context that owns `stream` current, returning the one to put
    back -- None when nothing had to change, which is the common case.

    HIP streams are not bound to a context, so this is CUDA only."""
    if stream == 0 or accelerator_api() != "cuda":
        return None
    cuda = _libcuda()
    wanted = _stream_context(stream)
    previous = ctypes.c_void_p()
    if cuda.cuCtxGetCurrent(ctypes.byref(previous)) != 0:
        raise RuntimeError("cuCtxGetCurrent failed")
    if previous.value == wanted.value:
        return None
    if cuda.cuCtxSetCurrent(wanted) != 0:
        raise RuntimeError("cuCtxSetCurrent failed for the mojo stream's context")
    return previous


def _push_device_context(device: int) -> ctypes.c_void_p | None:
    """The same for a device, through its mojo current stream."""
    return _push_stream_context(
        device_module.stream_native_handle(device_module.current_stream(device))
    )


def _pop_context(previous: ctypes.c_void_p | None):
    """Undo a `_push_*_context`. A NULL `previous` is the "no context current"
    state, and is restored as faithfully as any other."""
    if previous is not None:
        _libcuda().cuCtxSetCurrent(previous)


def _current_stream_handle(device: int) -> int:
    """Triton asks this right before every launch, for the current device."""
    return device_module.stream_native_handle(device_module.current_stream(device))


class _DeviceInterface:
    """What triton.testing (do_bench, the autotuner's benchmarker) asks of
    the `torch.cuda` module."""

    @staticmethod
    def Event(enable_timing: bool = False) -> torch.Event:  # noqa: N802 -- torch.cuda's spelling
        return torch.Event(device="mojo", enable_timing=enable_timing)

    synchronize = staticmethod(device_module.synchronize)
    current_device = staticmethod(device_module.current_device)
    set_device = staticmethod(device_module.set_device)

    @staticmethod
    def empty_cache():
        pass


def _set_current_device(device: torch.device | str | int | None):
    device_module.set_device(
        device_module.current_device() if device is None else device
    )


def _bind_mojo(driver: GPUDriver):
    """The callables Triton's GPUDriver base takes from torch.cuda."""
    driver.get_current_device = device_module.current_device
    driver.set_current_device = _set_current_device
    driver.get_current_stream = _current_stream_handle


class _MojoCudaUtils:
    """Triton's `CudaUtils` with `load_binary` under the right context.

    `loadBinary` (`triton/backends/nvidia/driver.c`) calls `cuModuleLoadData`
    in whatever context is current and retains `device`'s primary context only
    when there is none -- it never checks that an already-current context
    belongs to `device`. Inductor loads under `DeviceGuard(MojoInterface, i)`,
    which moves only mojo's TLS device, so compiling for `mojo:1` with device
    0's context current would put the module in context 0 and then launch it
    on device 1's stream.

    Delegation rather than wrapping the method in place: `CudaUtils` is a
    process-wide singleton whose `__init__` re-binds `load_binary` on every
    `CudaUtils()`, so an in-place wrapper would silently come undone.
    """

    def __init__(self, utils: CudaUtils):
        self._utils = utils

    def load_binary(
        self, name: str, data: bytes, shared: int, device: int
    ) -> tuple[int, int, int, int, int]:
        previous = _push_device_context(device)
        try:
            return self._utils.load_binary(name, data, shared, device)
        finally:
            _pop_context(previous)

    def __getattr__(self, name: str) -> object:
        return getattr(self._utils, name)


def _cuda_driver_class() -> type[DriverBase]:
    from triton.backends.nvidia.driver import (  # noqa: PLC0415 -- triton is optional
        CudaDriver,
        CudaLauncher,
        CudaUtils,
    )

    class MojoCudaLauncher(CudaLauncher):
        """Every launch under the context that owns its stream.

        Triton's generated `launch` installs a context only when none is
        current, and then device 0's primary one whatever device the stream
        belongs to (`ensureCudaContext`). Triton's own JIT and Inductor's
        generated launchers both call this object, so it is the one place
        both paths pass through.
        """

        def __call__(
            self,
            grid_x: int,
            grid_y: int,
            grid_z: int,
            stream: int,
            function: int,
            *args: object,
        ):
            previous = _push_stream_context(stream)
            try:
                super().__call__(grid_x, grid_y, grid_z, stream, function, *args)
            finally:
                _pop_context(previous)

    class MojoCudaDriver(CudaDriver):
        def __init__(self):
            # not the base constructors: they bind torch.cuda
            self.utils = _MojoCudaUtils(CudaUtils())
            self.launcher_cls = MojoCudaLauncher
            self.get_device_capability = _device_capability
            _bind_mojo(self)

        def get_active_torch_device(self) -> torch.device:
            return torch.device("mojo", self.get_current_device())

        def get_device_interface(self) -> type[_DeviceInterface]:
            return _DeviceInterface

        @staticmethod
        def is_active() -> bool:
            return accelerator_api() == "cuda"

        def get_empty_cache_for_benchmark(self) -> torch.Tensor:
            return torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="mojo")

    return MojoCudaDriver


def _hip_driver_class() -> type[DriverBase]:
    """The AMD counterpart; its target comes from the HIP driver API
    (`utils.get_device_properties`), so nothing else changes -- HIP streams
    are not bound to a context, so the context guards above have no
    counterpart here. Untested: written from Triton's AMD driver, no AMD GPU
    was available."""
    from triton.backends.amd.driver import (  # noqa: PLC0415 -- triton is optional
        HIPDriver,
        HIPLauncher,
        HIPUtils,
    )

    class MojoHipDriver(HIPDriver):
        def __init__(self):
            self.utils = HIPUtils()
            self.launcher_cls = HIPLauncher
            _bind_mojo(self)

        def get_active_torch_device(self) -> torch.device:
            return torch.device("mojo", self.get_current_device())

        def get_device_interface(self) -> type[_DeviceInterface]:
            return _DeviceInterface

        @staticmethod
        def is_active() -> bool:
            return accelerator_api() == "hip"

        def get_empty_cache_for_benchmark(self) -> torch.Tensor:
            return torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="mojo")

    return MojoHipDriver


@functools.cache
def driver_class() -> type[DriverBase]:
    """The mojo Triton driver class for the vendor MAX drives. Cached: it is
    also the class of the registered Triton backend
    (`monkeypatching.register_the_mojo_triton_target`), which Triton compares
    the active driver against with `isinstance`."""
    api = accelerator_api()
    if api == "cuda":
        return _cuda_driver_class()
    if api == "hip":
        return _hip_driver_class()
    raise RuntimeError(
        f"Triton has no backend for the mojo device's {api!r} accelerator"
    )


def make_driver() -> DriverBase:
    return driver_class()()


def enable_triton():
    """Make Triton launch on the mojo device, for the whole process
    (idempotent).

    Triton keeps one active driver and asks it for the device and the stream
    before it inspects a launch's arguments, so from here on **every** Triton
    launch in this process goes to the mojo current device and the mojo
    current stream, whatever its tensors are. On a torch with a working CUDA
    build that means CUDA tensors must not be launched through Triton in this
    process: a tensor produced on a `torch.cuda` stream would then be consumed
    on an unordered mojo stream, and on a different GPU whenever the two
    current devices differ. `register_mojo_devices()` installs the driver by
    itself only where that cannot happen -- a torch with no working CUDA/ROCm
    build (see `install_triton_hook`).
    """
    from triton.runtime import driver  # noqa: PLC0415 -- triton is optional

    driver.set_active(make_driver())


class _AfterTritonDriverImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Installs the mojo driver the moment `triton.runtime.driver` finishes
    importing: a finder that hands back the module's real spec with a loader
    wrapper, so importing triton stays as lazy as the user's code makes it."""

    def find_spec(
        self, name: str, path: object, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if name != _DRIVER_MODULE:
            return None
        sys.meta_path.remove(self)  # one shot; the real finders take over below
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        self._loader = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType):
        self._loader.exec_module(module)
        if accelerator_api() in ("cuda", "hip"):  # never break an unrelated import
            module.driver.set_active(make_driver())


def install_triton_hook():
    """Called by register_mojo_devices(): drive Triton from the mojo device
    once its runtime exists, now or on import -- but only on a torch with no
    working CUDA/ROCm build.

    The active Triton driver is process-wide and answers before a launch's
    arguments are looked at (see `enable_triton`), so installing it beside a
    working `torch.cuda` would redirect that process's *CUDA* Triton launches
    to the mojo current device and stream as well. Nothing in the driver
    protocol distinguishes the two, so there is no automatic answer that is
    right for both: the hook serves the case the feature exists for -- the CPU
    torch wheel, where Triton has no other device to run on -- and a vendor
    build of torch gets `enable_triton()` on request instead.
    """
    if os.environ.get("TORCH_MOJO_BACKEND_TRITON", "1") == "0":
        return
    if torch.cuda.is_available():
        return
    if importlib.util.find_spec("triton") is None:
        return
    if _DRIVER_MODULE in sys.modules:
        if accelerator_api() in ("cuda", "hip"):
            enable_triton()
    elif not any(isinstance(f, _AfterTritonDriverImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _AfterTritonDriverImport())
