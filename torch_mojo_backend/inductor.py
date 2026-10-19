"""TorchInductor on the mojo device: `torch.compile(fn, backend="inductor")`.

Inductor is device-agnostic through two registries, the ones Intel's XPU
backend and torch_npu plug into:

* `torch._dynamo.device_interface.register_interface_for_device` -- the
  runtime side: which device is current, its streams, events and properties;
* `torch._inductor.codegen.common.register_backend_for_device` -- the codegen
  side: which Scheduling emits kernels and which wrapper emits the host code,
  plus a `DeviceOpOverrides` giving the few device-specific lines the wrapper
  writes (`get_raw_stream`, `set_device`, the device guard).

With those in place Inductor generates ordinary Triton kernels; they compile
and launch through `triton_driver`, so a compiled graph runs on the mojo
current stream and needs no CUDA build of torch. What is left is what no registry
covers -- torch's hardcoded device lists and its compile workers -- one
monkeypatch each: `add_mojo_to_the_inductor_gpu_types`,
`let_has_triton_see_the_mojo_device`, `register_the_mojo_triton_target` and
`compile_inductor_kernels_in_process`.

Device properties and the Triton target key off `triton_driver.accelerator_api()`
("cuda" or "hip", what MAX drives on this machine): CUDA reads them from
`libcuda.so.1` the way Triton's own NVIDIA driver does; HIP has no such
driver library to probe with ctypes, so it reuses the HIP Triton driver's
own `HIPUtils().get_device_properties()`, and its Triton target aliases
`triton.backends.amd.compiler.HIPBackend` instead of the NVIDIA one. Both
still need `add_mojo_to_the_inductor_gpu_types`,
`let_has_triton_see_the_mojo_device`, `register_the_mojo_triton_target` and
`compile_inductor_kernels_in_process` -- none of torch's hardcoded device
lists or its compile workers know about a PrivateUse1 device either way.

For a torch with no working CUDA/ROCm build only: `GPU_TYPES` is one
process-wide list and `get_gpu_type()` asserts at most one of its entries is
available, so "mojo" and "cuda" cannot both be in it (`enable_inductor`
refuses rather than break torch's own CUDA path).
"""

from __future__ import annotations

import ctypes
import functools
from dataclasses import dataclass

import torch
import torch._dynamo.exc
import torch.utils._triton
from torch._dynamo.device_interface import (
    DeviceInterface,
    register_interface_for_device,
)
from torch._inductor.codegen.common import (
    DeviceOpOverrides,
    register_backend_for_device,
    register_device_op_overrides,
)
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.codegen.wrapper import PythonWrapperCodegen

from torch_mojo_backend import _ptxas, monkeypatching
from torch_mojo_backend.native import device_module
from torch_mojo_backend.triton_driver import (
    accelerator_api,
    driver_class,
    enable_triton,
)

# CUdevice_attribute
_ATTR_MAX_THREADS_PER_BLOCK = 1
_ATTR_WARP_SIZE = 10
_ATTR_MULTIPROCESSOR_COUNT = 16
_ATTR_MAX_THREADS_PER_MULTIPROCESSOR = 39
_ATTR_COMPUTE_CAPABILITY_MAJOR = 75
_ATTR_COMPUTE_CAPABILITY_MINOR = 76
_ATTR_MAX_REGISTERS_PER_MULTIPROCESSOR = 82

# AMD GPUs have no sub-generational compute-capability split like CUDA's
# major/minor: a bare gfx string (e.g. "gfx942") is both the arch Triton's
# HIPBackend compiles against and what Inductor's DeviceProperties.cc wants
# (see torch's own CudaInterface.get_compute_capability under torch.version.hip).
_HIP_MAX_THREADS_PER_BLOCK = 1024

_worker_device: dict[str, int] = {}


@dataclass(frozen=True)
class MojoDeviceProperties:
    """The `torch.cuda.get_device_properties` fields Inductor reads
    (`runtime/hints.py`'s `DeviceProperties.create`), from the vendor driver
    rather than from torch, which here has no usable CUDA/ROCm build.

    `cc` is `get_compute_capability`'s return value: `major * 10 + minor` for
    CUDA, the bare gfx string (e.g. "gfx942") for HIP. `major` is CUDA-only
    (`is_triton_capable`'s pre-Volta gate); HIP has no such generation split."""

    # native.device_module.MojoDeviceProperties is the public MAX view;
    # this separate field set is the Triton autotuner's contract.
    name: str
    cc: int | str
    multi_processor_count: int
    max_threads_per_multi_processor: int
    regs_per_multiprocessor: int
    max_threads_per_block: int
    warp_size: int
    major: int | None = None


def _cuda_device_properties(index: int) -> MojoDeviceProperties:
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit(0)
    handle = ctypes.c_int()
    if cuda.cuDeviceGet(ctypes.byref(handle), index) != 0:
        raise RuntimeError(f"cuDeviceGet({index}) failed")

    def attr(which: int) -> int:
        out = ctypes.c_int()
        if cuda.cuDeviceGetAttribute(ctypes.byref(out), which, handle) != 0:
            raise RuntimeError(f"cuDeviceGetAttribute({which}) failed")
        return out.value

    name = ctypes.create_string_buffer(256)
    cuda.cuDeviceGetName(name, ctypes.c_int(256), handle)
    major = attr(_ATTR_COMPUTE_CAPABILITY_MAJOR)
    minor = attr(_ATTR_COMPUTE_CAPABILITY_MINOR)
    return MojoDeviceProperties(
        name=name.value.decode(errors="replace"),
        cc=major * 10 + minor,
        major=major,
        multi_processor_count=attr(_ATTR_MULTIPROCESSOR_COUNT),
        max_threads_per_multi_processor=attr(_ATTR_MAX_THREADS_PER_MULTIPROCESSOR),
        regs_per_multiprocessor=attr(_ATTR_MAX_REGISTERS_PER_MULTIPROCESSOR),
        max_threads_per_block=attr(_ATTR_MAX_THREADS_PER_BLOCK),
        warp_size=attr(_ATTR_WARP_SIZE),
    )


def _hip_device_properties(index: int) -> MojoDeviceProperties:
    """No libamdhip64 probing of our own: `HIPUtils` is the extension
    `triton.backends.amd.driver` already builds and dlopens for the HIP
    Triton driver (`triton_driver._hip_driver_class`), so reusing it needs no
    new native code. Its dict's `arch` carries ROCm feature suffixes
    (e.g. "gfx942:sramecc+:xnack-"); `HIPDriver.get_current_target` strips
    them the same way before handing the bare gfx string to Triton."""
    # Optional Triton dependency: its wheels are unavailable on macOS.
    from triton.backends.amd.driver import (  # noqa: PLC0415 -- triton is optional
        HIPUtils,
    )

    props = HIPUtils().get_device_properties(index)
    arch = str(props["arch"]).split(":")[0]
    return MojoDeviceProperties(
        name=arch,
        cc=arch,
        multi_processor_count=props["multiprocessor_count"],
        max_threads_per_multi_processor=props["max_threads_per_sm"],
        regs_per_multiprocessor=props["max_num_regs"],
        max_threads_per_block=_HIP_MAX_THREADS_PER_BLOCK,
        warp_size=props["warpSize"],
    )


@functools.cache
def _device_properties(index: int) -> MojoDeviceProperties:
    if accelerator_api() == "hip":
        return _hip_device_properties(index)
    return _cuda_device_properties(index)


def get_raw_stream(device_index: int) -> int:
    """The vendor handle of the mojo current stream, in the shape Inductor's
    generated wrapper wants (`get_raw_stream(0)` then `kernel.run(..., stream=)`).
    Imported by name into every generated module: see
    `MojoDeviceOpOverrides.import_get_raw_stream_as`."""
    return device_module.stream_native_handle(
        device_module.current_stream(device_index)
    )


def _index(device: torch.types.Device) -> int:
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        device = device.index
    if device is None:
        return device_module.current_device()
    return device


class MojoInterface(DeviceInterface):
    """The `torch.mojo` half of `torch.cuda` that Dynamo and Inductor use."""

    device = device_module.device
    Event = torch.Event
    Stream = torch.Stream

    class Worker:
        """What a compile worker, which cannot touch the GPU, may ask: the
        properties are read once here and answered from the cache after.
        (The equivalent of torch's own `caching_worker_current_devices`,
        private to us: nothing in torch reads that dict for another
        device type.)"""

        @staticmethod
        def set_device(device: int):
            _worker_device["index"] = device

        @staticmethod
        def current_device() -> int:
            return _worker_device.get("index", device_module.current_device())

        @staticmethod
        def get_device_properties(
            device: torch.types.Device = None,
        ) -> MojoDeviceProperties:
            if device is None:
                return _device_properties(MojoInterface.Worker.current_device())
            return _device_properties(_index(device))

    current_device = staticmethod(device_module.current_device)
    set_device = staticmethod(device_module.set_device)
    device_count = staticmethod(device_module.device_count)
    synchronize = staticmethod(device_module.synchronize)
    is_available = staticmethod(device_module.is_available)
    stream = staticmethod(device_module.stream)
    current_stream = staticmethod(device_module.current_stream)
    set_stream = staticmethod(device_module.set_stream)
    get_raw_stream = staticmethod(get_raw_stream)
    is_bf16_supported = staticmethod(device_module.is_bf16_supported)

    @staticmethod
    def _set_stream_by_id(stream_id: int, device_index: int, device_type: int):
        device_module.set_stream(
            torch.Stream(
                stream_id=stream_id, device_index=device_index, device_type=device_type
            )
        )

    @staticmethod
    def exchange_device(device: int) -> int:
        previous = device_module.current_device()
        device_module.set_device(device)
        return previous

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        if device < 0:
            return device_module.current_device()
        return MojoInterface.exchange_device(device)

    @staticmethod
    def memory_allocated(device: torch.types.Device = None) -> int:
        return device_module.memory_allocated(device)

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> int | str:
        return _device_properties(_index(device)).cc

    @staticmethod
    def is_triton_capable(device: torch.types.Device = None) -> bool:
        if accelerator_api() == "hip":
            return True  # every gfx9/gfx10/gfx11 target MAX drives has a Triton backend
        major = _device_properties(_index(device)).major
        return major is not None and major >= 7

    @classmethod
    def raise_if_triton_unavailable(cls, device: torch.types.Device = None):
        import triton.backends  # noqa: PLC0415 -- triton is optional

        if not cls.is_triton_capable(device):
            raise RuntimeError("the mojo device is too old for Triton (pre-Volta)")
        vendor_backend = "amd" if accelerator_api() == "hip" else "nvidia"
        if vendor_backend not in triton.backends.backends:
            # torch >= 2.14's has_triton() reads only this subclass as "no".
            unavailable = getattr(
                torch._dynamo.exc, "TritonUnavailableError", RuntimeError
            )
            raise unavailable(f"triton not built with the {vendor_backend!r} backend")


class MojoDeviceOpOverrides(DeviceOpOverrides):
    """The device-specific lines of the generated Python wrapper. The C++
    (AOTInductor) half of this interface is left unimplemented."""

    def import_get_raw_stream_as(self, name: str) -> str:
        return f"from torch_mojo_backend.inductor import get_raw_stream as {name}"

    def set_device(self, device_idx: int) -> str:
        return f"torch.mojo.set_device({device_idx})"

    def synchronize(self) -> str:
        return "torch.mojo.synchronize()"

    def device_guard(self, device_idx: int) -> str:
        return f"torch.mojo.device({device_idx})"


def enable_inductor():
    """Make `torch.compile(backend="inductor")` codegen for the mojo device
    (call after `register_mojo_devices`).

    Needs a torch with no working CUDA/ROCm build. Inductor decides "is this
    an accelerator?" from one process-wide list, `torch._inductor.utils`'s
    `GPU_TYPES`, and `get_gpu_type()` asserts at most one of its entries is
    available; with mojo appended next to a working `torch.cuda` that assert
    fires -- in autotuning's subprocess setup and in the profiler
    benchmarking -- for that process's CUDA workloads as much as for ours.
    Nothing in that API is per-graph, so the choice is one backend or the
    other, and this raises rather than silently break torch's own.

    It also takes Triton for the whole process (see `enable_triton`).
    """
    if torch.cuda.is_available():
        raise RuntimeError(
            "Inductor on the mojo device needs a torch without a working "
            "CUDA/ROCm build: torch._inductor's GPU_TYPES is process-wide and "
            "get_gpu_type() asserts at most one of its entries is available, so "
            "'mojo' cannot be added beside 'cuda' without breaking Inductor for "
            "CUDA workloads too. Use the CPU torch wheel -- the mojo device "
            "reaches the GPU through MAX, not through torch."
        )
    enable_triton()
    api = accelerator_api()
    if api != "hip":
        _ptxas.apply_triton_default()  # HIP has no ptxas to point anywhere
    monkeypatching.add_mojo_to_the_inductor_gpu_types()
    monkeypatching.let_has_triton_see_the_mojo_device()
    monkeypatching.register_the_mojo_triton_target(driver_class(), api)
    monkeypatching.compile_inductor_kernels_in_process()
    register_interface_for_device("mojo", MojoInterface)
    for index in range(device_module.device_count()):
        register_interface_for_device(f"mojo:{index}", MojoInterface)
    register_device_op_overrides("mojo", MojoDeviceOpOverrides())
    register_backend_for_device("mojo", TritonScheduling, PythonWrapperCodegen)
    # torch >= 2.14's has_triton() asks the interfaces just registered; drop
    # an answer cached before them.
    torch.utils._triton.has_triton.cache_clear()
