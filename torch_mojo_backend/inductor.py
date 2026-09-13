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

NVIDIA only, like `triton_driver`, and for a torch with no working CUDA/ROCm
build only: `GPU_TYPES` is one process-wide list and `get_gpu_type()` asserts
at most one of its entries is available, so "mojo" and "cuda" cannot both be
in it (`enable_inductor` refuses rather than break torch's own CUDA path).
"""

from __future__ import annotations

import ctypes
import functools
from dataclasses import dataclass

import torch
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
from torch_mojo_backend.triton_driver import driver_class, enable_triton

# CUdevice_attribute
_ATTR_MAX_THREADS_PER_BLOCK = 1
_ATTR_WARP_SIZE = 10
_ATTR_MULTIPROCESSOR_COUNT = 16
_ATTR_MAX_THREADS_PER_MULTIPROCESSOR = 39
_ATTR_COMPUTE_CAPABILITY_MAJOR = 75
_ATTR_COMPUTE_CAPABILITY_MINOR = 76
_ATTR_MAX_REGISTERS_PER_MULTIPROCESSOR = 82

_worker_device: dict[str, int] = {}


@dataclass(frozen=True)
class MojoDeviceProperties:
    """The `torch.cuda.get_device_properties` fields Inductor reads
    (`runtime/hints.py`'s `DeviceProperties.create`), from the CUDA driver
    rather than from torch, which here has no usable CUDA build."""

    name: str
    major: int
    minor: int
    multi_processor_count: int
    max_threads_per_multi_processor: int
    regs_per_multiprocessor: int
    max_threads_per_block: int
    warp_size: int


@functools.cache
def _device_properties(index: int) -> MojoDeviceProperties:
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
    return MojoDeviceProperties(
        name=name.value.decode(errors="replace"),
        major=attr(_ATTR_COMPUTE_CAPABILITY_MAJOR),
        minor=attr(_ATTR_COMPUTE_CAPABILITY_MINOR),
        multi_processor_count=attr(_ATTR_MULTIPROCESSOR_COUNT),
        max_threads_per_multi_processor=attr(_ATTR_MAX_THREADS_PER_MULTIPROCESSOR),
        regs_per_multiprocessor=attr(_ATTR_MAX_REGISTERS_PER_MULTIPROCESSOR),
        max_threads_per_block=attr(_ATTR_MAX_THREADS_PER_BLOCK),
        warp_size=attr(_ATTR_WARP_SIZE),
    )


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
        return 0

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> int:
        props = _device_properties(_index(device))
        return props.major * 10 + props.minor

    @staticmethod
    def is_triton_capable(device: torch.types.Device = None) -> bool:
        return _device_properties(_index(device)).major >= 7

    @classmethod
    def raise_if_triton_unavailable(cls, device: torch.types.Device = None):
        import triton.backends  # noqa: PLC0415 -- triton is optional

        if not cls.is_triton_capable(device):
            raise RuntimeError("the mojo device is too old for Triton (pre-Volta)")
        if "nvidia" not in triton.backends.backends:
            raise RuntimeError("triton not built with the 'nvidia' backend")


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
    _ptxas.apply_triton_default()
    monkeypatching.add_mojo_to_the_inductor_gpu_types()
    monkeypatching.let_has_triton_see_the_mojo_device()
    monkeypatching.register_the_mojo_triton_target(driver_class())
    monkeypatching.compile_inductor_kernels_in_process()
    register_interface_for_device("mojo", MojoInterface)
    for index in range(device_module.device_count()):
        register_interface_for_device(f"mojo:{index}", MojoInterface)
    register_device_op_overrides("mojo", MojoDeviceOpOverrides())
    register_backend_for_device("mojo", TritonScheduling, PythonWrapperCodegen)
