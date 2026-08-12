"""Does a foreign CUDA tensor live on the same physical GPU as a mojo one?

If it does, a transfer between them never has to touch the host: both pointers
are ordinary device addresses in one process, so the copy stays on HBM instead
of crossing PCIe twice.  Measured on an H100 PCIe, 537 MB: 0.75 ms on device
against 386 ms through the host.

Answering the question is the hard part.  MAX's `Device` exposes only an
ordinal, an api string and a model name -- nothing that identifies a physical
card -- so matching MAX's `gpu:0` to torch's `cuda:0` by ordinal alone would be
an assumption about two runtimes enumerating identically, and being wrong means
reading another GPU's memory.  Instead we ask CUDA which device each POINTER
belongs to, which is a fact about the allocation rather than a guess about
enumeration, and works for any multi-GPU arrangement.
"""

from __future__ import annotations

import ctypes
import functools

import torch

# From cuda.h.  The driver call writes a plain int for both of these.
_ATTRIBUTE_MEMORY_TYPE = 2
_ATTRIBUTE_DEVICE_ORDINAL = 9
_MEMORYTYPE_DEVICE = 2


@functools.cache
def _driver() -> ctypes.CDLL | None:
    """libcuda, or None where this is not an NVIDIA stack.

    A ROCm torch reports its devices as `cuda` too -- `torch.version.hip` is
    the only thing that separates them -- so a device-type check is not a
    vendor check.  Everything here speaks the CUDA driver API, so anything
    else returns None and the caller bounces through the host, which is always
    correct and merely slower.

    The DRIVER library on purpose, not the runtime: `cuPointerGetAttribute`
    writes an int, while the runtime's `cudaPointerGetAttributes` writes a
    struct whose layout has grown across CUDA versions.  Declaring that struct
    a field short overruns the buffer and corrupts the heap, which then
    crashes somewhere unrelated -- a trap this file exists partly to avoid.
    libcuda is also always present when CUDA works, whereas libcudart moves
    between the torch wheel, the nvidia wheels and the system.
    """
    if getattr(torch.version, "hip", None) is not None:
        return None
    try:
        return ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None


def _pointer_attribute(ptr: int, attribute: int) -> int | None:
    lib = _driver()
    if lib is None or not ptr:
        return None
    value = ctypes.c_int(-1)
    status = lib.cuPointerGetAttribute(
        ctypes.byref(value), ctypes.c_int(attribute), ctypes.c_ulonglong(ptr)
    )
    return value.value if status == 0 else None


def device_ordinal(ptr: int) -> int | None:
    """The CUDA ordinal owning `ptr`, or None if it is not CUDA device memory.

    Returns an ordinal for MAX-allocated memory too: MAX allocates through the
    same driver, so its buffers answer this query exactly as torch's do.
    """
    if _pointer_attribute(ptr, _ATTRIBUTE_MEMORY_TYPE) != _MEMORYTYPE_DEVICE:
        return None
    return _pointer_attribute(ptr, _ATTRIBUTE_DEVICE_ORDINAL)


def same_physical_device(one: int, other: int) -> bool:
    """True when both pointers are device memory on the same physical GPU."""
    first = device_ordinal(one)
    return first is not None and first == device_ordinal(other)
