"""HIP-runtime facts about mojo tensors on AMD GPUs: which device owns a
pointer, and making a device current on the calling thread.

The AMD counterpart of `cuda_peer.py`, with the same posture: everything is
asked of the runtime that MAX itself drives, so the answers are facts about an
allocation rather than assumptions about how two runtimes enumerate. MAX
dlopens `libamdhip64.so` from the ROCm install (``$ROCM_PATH`` or
``/opt/rocm``); this module reuses THAT copy — found through the process's own
memory map, preferring the one under that install when a ROCm torch wheel
has mapped its own — instead of resolving the soname a second time, which on
a box with several ROCm versions could map a different runtime with its own
device table. RCCL links against the same soname, so the ordinals here,
MAX's and RCCL's are one numbering.

Only the driver-style `hipPointerGetAttribute` (singular) is used: it writes
a plain int, where `hipPointerGetAttributes` fills a struct whose layout
changed in ROCm 6.0 — the trap cuda_peer.py describes for the CUDA runtime
API.
"""

from __future__ import annotations

import ctypes
import os
import warnings
from pathlib import Path

import torch

from torch_mojo_backend.torch_compile_backend import utils as _compile_utils

# hip/driver_types.h: hipPointer_attribute (CUDA's CUpointer_attribute values).
_ATTRIBUTE_MEMORY_TYPE = 2
_ATTRIBUTE_DEVICE_ORDINAL = 9
# hip/hip_runtime_api.h: hipMemoryType, ROCm >= 6.0 numbering.
_MEMORYTYPE_DEVICE = 2
_MEMORYTYPE_MANAGED = 3
_MEMORYTYPE_UNIFIED = 11

_SONAMES = ("libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so")
HIP_SUCCESS = 0


def _rocm_roots() -> list[Path]:
    """Where MAX looks for its HIP runtime, in its order: $ROCM_PATH, then
    /opt/rocm ($HIP_PATH is the older spelling of the first)."""
    roots = []
    for root in (os.environ.get("ROCM_PATH"), os.environ.get("HIP_PATH"), "/opt/rocm"):
        if root and Path(root) not in roots:
            roots.append(Path(root))
    return roots


def _mapped_runtime_paths() -> list[Path]:
    """Every distinct libamdhip64 mapped in this process, in map order."""
    found: list[Path] = []
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                path = line.rstrip("\n").partition(" /")[2]
                if not path:
                    continue
                # A replaced file keeps its mapping under "<path> (deleted)".
                path = "/" + path.removesuffix(" (deleted)")
                if (
                    Path(path).name.startswith("libamdhip64.so")
                    and Path(path) not in found
                ):
                    found.append(Path(path))
    except OSError:
        pass
    return found


def _mapped_runtime_path() -> Path | None:
    """The libamdhip64 MAX loaded, if any is mapped yet.

    A ROCm torch wheel maps its own bundled copy too. /proc/self/maps is
    ordered by address, not by load order, so prefer the copy under the
    ROCm install MAX resolves its runtime from over any other.
    """
    mapped = _mapped_runtime_paths()
    for root in _rocm_roots():
        for path in mapped:
            if root.resolve() in path.resolve().parents:
                return path
    return mapped[0] if mapped else None


def _candidate_runtime_paths() -> list[str]:
    candidates: list[str] = []
    mapped = _mapped_runtime_path()
    if mapped is not None:
        candidates.append(str(mapped))
    # The install MAX will use comes before whatever LD_LIBRARY_PATH resolves
    # a bare soname to, so a pre-MAX call cannot bind a different runtime.
    for root in _rocm_roots():
        candidates.extend(str(root / "lib" / name) for name in _SONAMES)
    candidates.extend(_SONAMES)
    return candidates


_RUNTIME: list[ctypes.CDLL] = []


def _runtime() -> ctypes.CDLL | None:
    """libamdhip64, or None where this is not a ROCm stack.

    Cached once a load succeeds; a miss is not cached, so a call made before
    MAX has loaded HIP does not pin the answer for the process.
    """
    if _RUNTIME:
        return _RUNTIME[0]
    for path in _candidate_runtime_paths():
        try:
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        lib.hipPointerGetAttribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.hipPointerGetAttribute.restype = ctypes.c_int
        lib.hipSetDevice.argtypes = [ctypes.c_int]
        lib.hipSetDevice.restype = ctypes.c_int
        lib.hipGetErrorString.argtypes = [ctypes.c_int]
        lib.hipGetErrorString.restype = ctypes.c_char_p
        _RUNTIME.append(lib)
        return lib
    return None


def available() -> bool:
    return _runtime() is not None


def runtime_dir() -> Path | None:
    """Directory of the HIP runtime in use — where its ROCm's librccl lives too."""
    mapped = _mapped_runtime_path()
    return mapped.parent if mapped is not None else None


def _pointer_attribute(ptr: int, attribute: int) -> int | None:
    lib = _runtime()
    if lib is None or not ptr:
        return None
    value = ctypes.c_int(-1)
    status = lib.hipPointerGetAttribute(
        ctypes.byref(value), ctypes.c_int(attribute), ctypes.c_void_p(ptr)
    )
    return value.value if status == HIP_SUCCESS else None


def device_ordinal(ptr: int) -> int | None:
    """The HIP ordinal owning `ptr`, or None if it is not GPU memory.

    Device, managed and unified allocations all count: on an APU (MI300A)
    the HBM pool is shared with the host, and the runtime classifies a
    `hipMalloc` there the same way it does on a discrete card, but the
    ordinal is the useful fact for every kind the GPU can address natively.
    """
    kind = _pointer_attribute(ptr, _ATTRIBUTE_MEMORY_TYPE)
    if kind not in (_MEMORYTYPE_DEVICE, _MEMORYTYPE_MANAGED, _MEMORYTYPE_UNIFIED):
        return None
    return _pointer_attribute(ptr, _ATTRIBUTE_DEVICE_ORDINAL)


def set_device(ordinal: int):
    """Make `ordinal` this thread's current HIP device (thread-local in HIP).

    `cudaSetDevice`'s counterpart: RCCL resolves the GPU a communicator binds
    to from it at `ncclCommInitRank`.
    """
    lib = _runtime()
    if lib is None:
        raise RuntimeError("libamdhip64 is not loadable; is this a ROCm system?")
    status = lib.hipSetDevice(ordinal)
    if status != HIP_SUCCESS:
        detail = lib.hipGetErrorString(status).decode()
        raise RuntimeError(
            f"hipSetDevice({ordinal}) failed: {detail} (hipError_t={status})"
        )


def _amd_gpu_present() -> bool:
    """The kernel's AMD GPU compute interface; cheaper than any runtime call."""
    return Path("/dev/kfd").exists()


def warn_if_gpu_torch_on_hip():
    """One-time hint at registration: a GPU torch wheel on an AMD box hurts.

    Runs only where an AMD GPU is present (``/dev/kfd``), so it never touches
    MAX's device enumeration on NVIDIA or CPU-only hosts; and it can only
    warn, never fail registration.

    - A CUDA wheel: the HIP runtime walks every shared object mapped in the
      process on each kernel load (``dl_iterate_phdr`` from libhsa-runtime64,
      hunting embedded code objects), and the CUDA wheel maps ~3 GB of NVIDIA
      libraries it never uses here. Measured on 4x MI300A, nanoGPT 124M under
      DDP: the first training step took 14.7 s with torch 2.11+cu130 against
      1.0 s with torch 2.11+cpu, and steady state ran 6% slower as well.
    - A ROCm wheel: torch loads its own bundled libamdhip64 at import, next
      to the one MAX loads from the ROCm install, so the process runs two HIP
      runtimes. RCCL and the pointer-ownership query bind to one of them and
      MAX's buffers belong to the other; that combination is untested and the
      distributed backend refuses to guess about it.

    The CPU wheel is what this backend needs anyway.
    """
    if not _amd_gpu_present():
        return
    cuda_build = getattr(torch.version, "cuda", None) is not None
    hip_build = getattr(torch.version, "hip", None) is not None
    if not (cuda_build or hip_build):
        return
    try:
        # Through the module, not a bound name: tests substitute the function.
        if not any(device.api == "hip" for device in _compile_utils.get_accelerators()):
            return
    except Exception:  # a hint must never break registration
        return
    if hip_build:
        detail = (
            "a ROCm build, which loads its own HIP runtime next to the one MAX "
            "uses; the RCCL process group and the pointer-ownership query cannot "
            "serve buffers from two runtimes"
        )
    else:
        detail = (
            "a CUDA build; the HIP runtime rescans every mapped library at each "
            "kernel load, and the CUDA wheel maps gigabytes of unused NVIDIA "
            "libraries, so first-use kernel loads are ~10x slower"
        )
    warnings.warn(
        f"torch-mojo-backend: this torch (torch {torch.__version__}) is {detail}. "
        "Install the CPU wheel instead: "
        "uv pip install torch --index-url https://download.pytorch.org/whl/cpu",
        RuntimeWarning,
        stacklevel=2,
    )
