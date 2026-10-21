"""The CUDA driver API through ctypes, torch-free (libcuda.so.1 ships with
the driver, whatever torch build is installed)."""

import ctypes
import functools

# CUdevice_attribute
MAX_THREADS_PER_BLOCK = 1
WARP_SIZE = 10
MULTIPROCESSOR_COUNT = 16
MAX_THREADS_PER_MULTIPROCESSOR = 39
COMPUTE_CAPABILITY_MAJOR = 75
COMPUTE_CAPABILITY_MINOR = 76
MAX_REGISTERS_PER_MULTIPROCESSOR = 82


@functools.cache
def libcuda() -> ctypes.CDLL:
    """libcuda with `cuInit` done. Raises (and caches nothing) when there is
    no driver or it fails to initialize."""
    cuda = ctypes.CDLL("libcuda.so.1")
    if cuda.cuInit(0) != 0:
        raise RuntimeError("cuInit failed")
    return cuda


def device(index: int) -> ctypes.c_int:
    handle = ctypes.c_int()
    if libcuda().cuDeviceGet(ctypes.byref(handle), index) != 0:
        raise RuntimeError(f"cuDeviceGet({index}) failed")
    return handle


def attribute(handle: ctypes.c_int, which: int) -> int:
    out = ctypes.c_int()
    if libcuda().cuDeviceGetAttribute(ctypes.byref(out), which, handle) != 0:
        raise RuntimeError(f"cuDeviceGetAttribute({which}) failed")
    return out.value


def name(handle: ctypes.c_int, lib: ctypes.CDLL | None = None) -> str:
    buf = ctypes.create_string_buffer(256)
    (lib or libcuda()).cuDeviceGetName(buf, 256, handle)
    return buf.value.decode(errors="replace")
