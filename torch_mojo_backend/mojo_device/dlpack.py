"""Hand-built DLPack capsules over a raw device allocation.

`torch_compile_backend/compiler.py` adopts a compiled MAX graph's output
buffers as `mojo` tensors zero-copy with `make_capsule_privateuse1`; torch
cannot do that itself, since its importer keys off the DLPack device-type
code and MAX tags its buffers with the vendor one. `make_capsule` is the
vendor-tagged variant, for consumers like `max.driver.Buffer.from_dlpack`.

Only contiguous allocations are exported (callers materialize first), so the
capsule advertises compact row-major layout (strides=NULL). The capsule
pins the caller-supplied `holder` -- any object whose refcount keeps the
memory alive -- until the consumer's deleter runs.
"""

# ctypes._CData / ctypes._Pointer are typeshed-only names (not real runtime
# attributes of the ctypes module); deferred evaluation keeps annotations
# that reference them from crashing at import time.
from __future__ import annotations

import ctypes
from collections.abc import Callable, Sequence

import max.driver
from max.dtype import DType


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))

_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLManagedTensorDeleter),
]

# DLPack type codes (see ATen/dlpack.h): kDLInt=0, kDLUInt=1, kDLFloat=2,
# kDLBfloat=4, kDLBool=6.
_DLPACK_CODE_OF: dict[DType, tuple[int, int]] = {
    DType.bool: (6, 8),
    DType.int8: (0, 8),
    DType.int16: (0, 16),
    DType.int32: (0, 32),
    DType.int64: (0, 64),
    DType.uint8: (1, 8),
    DType.uint16: (1, 16),
    DType.uint32: (1, 32),
    DType.uint64: (1, 64),
    DType.float16: (2, 16),
    DType.float32: (2, 32),
    DType.float64: (2, 64),
    DType.bfloat16: (4, 16),
}

# kDLCPU=1, kDLCUDA=2, kDLMetal=8, kDLROCM=10.
_DLPACK_DEVICE_TYPE_OF = {"cpu": 1, "cuda": 2, "metal": 8, "hip": 10}

_CAPSULE_NAME = b"dltensor"

_pyapi = ctypes.pythonapi
_PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_pyapi.PyCapsule_New.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    _PyCapsule_Destructor,
]
_pyapi.PyCapsule_New.restype = ctypes.py_object
_pyapi.PyCapsule_IsValid.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_pyapi.PyCapsule_IsValid.restype = ctypes.c_int
_pyapi.PyCapsule_GetPointer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_pyapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
_pyapi.Py_IncRef.argtypes = [ctypes.py_object]
_pyapi.Py_IncRef.restype = None
_pyapi.Py_DecRef.argtypes = [ctypes.py_object]
_pyapi.Py_DecRef.restype = None

# A second handle on the same symbols, taking the capsule as a Python object
# rather than as a raw address: ctypes caches one function object per name per
# library, so `argtypes` cannot be both at once and the two callers above and
# below need different ones.
_pyapi_obj = ctypes.PyDLL(None)
_pyapi_obj.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_pyapi_obj.PyCapsule_GetPointer.restype = ctypes.c_void_p


class _ExportState:
    """Python objects that must outlive one exported DLManagedTensor.

    ``manager_ctx`` owns one manual Python reference to this state. That makes
    the DLPack consumer, rather than the module-global diagnostics dictionary,
    the lifetime root. Keeping both CFUNCTYPE objects here is equally
    important: a PyCapsule and a C++ DLPack consumer retain their callback
    addresses, but those raw C function pointers do not retain ctypes' Python
    callback trampolines by themselves.
    """

    __slots__ = (
        "managed",
        "shape_arr",
        "holder",
        "registry",
        "managed_deleter",
        "capsule_destructor",
        "released",
    )

    def __init__(
        self,
        managed: _DLManagedTensor,
        shape_arr: ctypes.Array[ctypes.c_int64],
        holder: object,
        registry: dict[int, _ExportState],
    ):
        self.managed = managed
        self.shape_arr = shape_arr
        self.holder = holder
        self.registry = registry
        self.managed_deleter = None
        self.capsule_destructor = None
        self.released = False


# Every live export, keyed by the DLManagedTensor struct address. The entry
# makes live exports observable for diagnostics/tests. It is not the ownership
# root: reloading or tearing down this module can replace the dictionary while
# a DLPack consumer still holds the raw DLManagedTensor pointer.
_live_exports: dict[int, _ExportState] = {}


def _release_export(
    handle: ctypes._Pointer[_DLManagedTensor],
    *,
    addressof: Callable[[ctypes._CData], int] = ctypes.addressof,
    cast: Callable[[int, type[ctypes.py_object]], ctypes.py_object] = ctypes.cast,
    py_object: type[ctypes.py_object] = ctypes.py_object,
    py_decref: Callable[[object], None] = _pyapi.Py_DecRef,
):
    """Release the producer reference owned by ``manager_ctx`` exactly once.

    All helpers needed during release are captured as defaults. An old ctypes
    callback can therefore finish safely after ``importlib.reload`` replaces
    this module's globals.
    """
    if not handle:
        return
    manager_ctx = handle.contents.manager_ctx
    if not manager_ctx:
        return

    # Reading a py_object from its address creates a normal local reference;
    # that keeps the state (and therefore this DLManagedTensor) alive until the
    # callback returns, even after the manual manager_ctx reference is dropped.
    state = cast(manager_ctx, py_object).value
    handle.contents.manager_ctx = None
    if state.released:
        return
    state.released = True
    state.registry.pop(addressof(handle.contents), None)
    py_decref(state)


def _deleter_impl(
    handle: ctypes._Pointer[_DLManagedTensor],
    release_export: Callable[
        [ctypes._Pointer[_DLManagedTensor]], None
    ] = _release_export,
):
    release_export(handle)


_managed_deleter = _DLManagedTensorDeleter(_deleter_impl)


def _capsule_destructor_impl(
    capsule_ptr: object,
    *,
    capsule_name: bytes = _CAPSULE_NAME,
    capsule_is_valid: Callable[[object, bytes], bool] = _pyapi.PyCapsule_IsValid,
    capsule_get_pointer: Callable[[object, bytes], int] = _pyapi.PyCapsule_GetPointer,
    cast: Callable[..., object] = ctypes.cast,
    managed_pointer: object = ctypes.POINTER(_DLManagedTensor),
    release_export: Callable[..., None] = _release_export,
):
    # A consumer that adopted the memory renames the capsule to
    # "used_dltensor" and becomes responsible for calling the deleter; if
    # the capsule dies still named "dltensor" it was never consumed and the
    # export is released here.
    if capsule_is_valid(capsule_ptr, capsule_name):
        addr = capsule_get_pointer(capsule_ptr, capsule_name)
        release_export(cast(addr, managed_pointer))


_capsule_destructor = _PyCapsule_Destructor(_capsule_destructor_impl)


def dlpack_device(device: max.driver.Device) -> tuple[int, int]:
    """The DLPack (device_type, device_id) pair for a max.driver.Device."""
    if device.label == "cpu":
        return (_DLPACK_DEVICE_TYPE_OF["cpu"], 0)
    device_type = _DLPACK_DEVICE_TYPE_OF.get(device.api)
    if device_type is None:
        raise BufferError(f"Cannot export device {device} via DLPack")
    return (device_type, device.id)


# torch's C++ DLPack importer maps this device-type code straight to
# `at::Device(DeviceType::PrivateUse1, index)` (aten/src/ATen/DLConvertor.cpp),
# independent of a *renamed* PrivateUse1 backend's Python-visible name (this
# project renames it to "mojo"). See `make_capsule_privateuse1` below and
# `torch_compile_backend/compiler.py`, which imports MAX output buffers this
# way.
_KDL_EXT_DEV = 12


def _build_capsule(
    holder: object,
    data_ptr: int,
    shape: Sequence[int],
    dtype: DType,
    device_type: int,
    device_id: int,
) -> object:
    """Shared "dltensor" PyCapsule builder for a contiguous device allocation.

    `holder` is any Python object whose refcount keeps the allocation
    alive; it is pinned until the consumer's deleter runs. `device_type` is
    a raw DLPack device-type code (see `make_capsule` and
    `make_capsule_privateuse1` for the two ways callers pick one).
    """
    code_bits = _DLPACK_CODE_OF.get(dtype)
    if code_bits is None:
        raise BufferError(f"dtype {dtype} is not exportable via DLPack")
    ndim = len(shape)
    shape_arr = (ctypes.c_int64 * ndim)(*shape)
    managed = _DLManagedTensor()
    managed.dl_tensor.data = data_ptr
    managed.dl_tensor.device = _DLDevice(device_type, device_id)
    managed.dl_tensor.ndim = ndim
    managed.dl_tensor.dtype = _DLDataType(code_bits[0], code_bits[1], 1)
    managed.dl_tensor.shape = shape_arr
    managed.dl_tensor.strides = None  # compact row-major
    managed.dl_tensor.byte_offset = 0
    managed.deleter = _managed_deleter
    addr = ctypes.addressof(managed)
    state = _ExportState(managed, shape_arr, holder, _live_exports)
    state.managed_deleter = _managed_deleter
    state.capsule_destructor = _capsule_destructor
    managed.manager_ctx = id(state)
    _live_exports[addr] = state

    # DLPack transfers this producer reference to either the unconsumed
    # capsule destructor or the consumer-provided storage deleter. It cannot be
    # represented solely by a Python container: module reload/teardown may
    # destroy that container while the consumer still owns the raw pointer.
    _pyapi.Py_IncRef(state)
    try:
        return _pyapi.PyCapsule_New(addr, _CAPSULE_NAME, _capsule_destructor)
    except Exception:
        _release_export(ctypes.pointer(managed))
        raise


def make_capsule(
    holder: object,
    data_ptr: int,
    shape: Sequence[int],
    dtype: DType,
    device: max.driver.Device,
) -> object:
    """A "dltensor" PyCapsule for a contiguous device allocation.

    `holder` is any Python object whose refcount keeps the allocation
    alive; it is pinned until the consumer's deleter runs.
    """
    return _build_capsule(holder, data_ptr, shape, dtype, *dlpack_device(device))


def make_capsule_privateuse1(
    holder: object, data_ptr: int, shape: Sequence[int], dtype: DType, device_index: int
) -> object:
    """A "dltensor" PyCapsule tagged for import as a `mojo` (renamed
    PrivateUse1) torch tensor at index `device_index`.

    Unlike `make_capsule` (which tags the real vendor device type so MAX
    recognizes the producer), this tags DLPack's ``kDLExtDev`` code:
    torch's C++ DLPack importer maps that straight to
    ``at::Device(DeviceType::PrivateUse1, device_index)`` regardless of the
    renamed backend's Python-visible name, so `torch.from_dlpack` on this
    capsule yields a `mojo:<device_index>` tensor sharing this memory
    zero-copy. Used for MAX graph outputs, whose buffers are otherwise
    tagged with MAX's own vendor device type (see compiler.py).
    """
    return _build_capsule(holder, data_ptr, shape, dtype, _KDL_EXT_DEV, device_index)


def retag_capsule(capsule: object, device_type: int, device_id: int) -> object:
    """Rewrite the device recorded in an unconsumed "dltensor" capsule.

    The producer of a capsule decides which DLPack device code it carries,
    and that code is the only thing an importer looks at to pick the torch
    device -- but the *memory* is the same either way when the two devices
    are two names for one piece of hardware. That is exactly the mojo/CUDA
    pair: a mojo tensor exports ``kDLExtDev``, a CUDA tensor exports
    ``kDLCUDA``, and both are a pointer into the same device's address
    space. Retagging is therefore how `cuda_interop` builds an alias --
    torch's own exporter fills in shape, strides, offset and dtype, and its
    deleter keeps the source tensor alive, which a hand-built capsule
    (`make_capsule*` above) would have to redo.

    The capsule is mutated in place and returned. Only a capsule the
    consumer has not adopted yet ("dltensor", not "used_dltensor") can be
    retagged: `PyCapsule_GetPointer` raises `ValueError` for anything else,
    which `PyDLL` turns back into a Python exception here.
    """
    managed = ctypes.cast(
        _pyapi_obj.PyCapsule_GetPointer(capsule, _CAPSULE_NAME),
        ctypes.POINTER(_DLManagedTensor),
    )
    managed.contents.dl_tensor.device = _DLDevice(device_type, device_id)
    return capsule
