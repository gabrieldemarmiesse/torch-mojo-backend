"""ctypes binding for the NCCL API — NCCL on NVIDIA, RCCL on AMD.

Why ctypes and not torch's own NCCL: this backend must work with a CPU-only
torch install (see "Rules about the eager mode" in AGENTS.md), so we cannot
rely on torch.distributed.ProcessGroupNCCL or torch.cuda being functional.

Two libraries implement one C API (`nccl.h` / `rccl.h`: same function names,
same enum values, same 128-byte unique id), so one binding serves both; only
where the library comes from and how the target GPU is selected differ:

- NVIDIA: `libnccl.so.2` from the nvidia-nccl-cu12 wheel. It statically links
  the CUDA runtime and dlopens libcuda.so.1 by itself, so this needs nothing
  beyond the wheel and a driver — the same trick mojo_device/cuda_peer.py
  uses. The target GPU is the CUDA context current on the calling thread;
  MAX binds the per-device *primary* context, so memory allocated by MAX is
  directly valid for NCCL, and `set_current_device` performs the minimal
  driver-API dance (cuInit -> cuDevicePrimaryCtxRetain -> cuCtxSetCurrent)
  that torch's `cudaSetDevice` would have done, without needing libcudart.
- AMD: `librccl.so.1` from the ROCm install whose HIP runtime MAX already
  loaded (mojo_device/hip_peer.py finds it next to that libamdhip64) — MAX
  itself needs a ROCm install on AMD, so there is nothing extra to ship, and
  taking RCCL from the same ROCm is what keeps one HIP runtime in the process
  (RCCL's `libamdhip64.so.N` dependency resolves to the loaded copy by
  soname). The target GPU is the thread's current HIP device, `hipSetDevice`.

Enum values are pinned to nccl.h from NCCL 2.27+ (verified against the
2.31.2 header) and rccl.h from ROCm 6.4 (RCCL 2.22); both keep them
ABI-stable across 2.x.
"""

import ctypes
import functools
import os
from pathlib import Path

from torch_mojo_backend.mojo_device import cuda_peer, hip_peer

# nccl.h: ncclResult_t
NCCL_SUCCESS = 0
NCCL_IN_PROGRESS = 7

# nccl.h: ncclRedOp_t
NCCL_SUM = 0
NCCL_PROD = 1
NCCL_MAX = 2
NCCL_MIN = 3
NCCL_AVG = 4

# nccl.h: ncclDataType_t
NCCL_INT8 = 0
NCCL_UINT8 = 1
NCCL_INT32 = 2
NCCL_UINT32 = 3
NCCL_INT64 = 4
NCCL_UINT64 = 5
NCCL_FLOAT16 = 6
NCCL_FLOAT32 = 7
NCCL_FLOAT64 = 8
NCCL_BFLOAT16 = 9

NCCL_UNIQUE_ID_BYTES = 128

_NCCL_LIB_ENV = "TORCH_MOJO_BACKEND_NCCL_LIB"
_RCCL_LIB_ENV = "TORCH_MOJO_BACKEND_RCCL_LIB"

# MAX's `Device.api` string -> the library implementing the NCCL API there.
_LIBRARY_NAME_OF = {"cuda": "NCCL", "hip": "RCCL"}


class NcclUniqueId(ctypes.Structure):
    """nccl.h: typedef struct { char internal[128]; } ncclUniqueId."""

    _fields_ = [("internal", ctypes.c_char * NCCL_UNIQUE_ID_BYTES)]


class NcclError(RuntimeError):
    """An NCCL/RCCL call returned a non-success ncclResult_t."""

    def __init__(self, func_name: str, result: int, detail: str):
        super().__init__(f"{func_name} failed: {detail} (ncclResult_t={result})")
        self.result = result


def _candidate_libnccl_paths() -> list[str]:
    override = os.environ.get(_NCCL_LIB_ENV)
    if override:
        return [override]
    candidates = []
    try:
        import nvidia.nccl  # noqa: PLC0415 -- optional: the wheel may not be installed

        # nvidia.nccl is a namespace package: no __file__, only __path__.
        for package_dir in nvidia.nccl.__path__:
            candidates.append(str(Path(package_dir) / "lib" / "libnccl.so.2"))
    except ImportError:
        pass
    # System fallbacks, same spirit as MAX's comm/vendor/ccl.mojo search list.
    candidates += ["libnccl.so.2", "libnccl.so"]
    return candidates


def _candidate_librccl_paths() -> list[str]:
    override = os.environ.get(_RCCL_LIB_ENV)
    if override:
        return [override]
    candidates = []
    # The ROCm whose HIP runtime is already in the process, first: see the
    # module docstring for why the two must come from the same install.
    runtime_dir = hip_peer.runtime_dir()
    if runtime_dir is not None:
        candidates.append(str(runtime_dir / "librccl.so.1"))
    for root in (os.environ.get("ROCM_PATH"), "/opt/rocm"):
        if root:
            candidates.append(str(Path(root) / "lib" / "librccl.so.1"))
    candidates += ["librccl.so.1", "librccl.so"]
    return candidates


def _install_help(api: str) -> str:
    if api == "hip":
        return (
            "could not load librccl.so.1 — it ships with every ROCm install "
            "(the one MAX loads libamdhip64 from); set ROCM_PATH to that "
            f"install or {_RCCL_LIB_ENV} to the library path"
        )
    return (
        "could not load libnccl.so.2 — install the nvidia-nccl-cu12 wheel or "
        f"set {_NCCL_LIB_ENV} to the library path"
    )


def _declare(lib: ctypes.CDLL):
    """Pin the argument types of every entry point used, once per library."""
    lib.ncclGetErrorString.restype = ctypes.c_char_p
    lib.ncclGetErrorString.argtypes = [ctypes.c_int]
    lib.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(NcclUniqueId)]
    # ncclUniqueId is passed BY VALUE — it must be a ctypes.Structure here: a
    # bare (c_char * 128) array argtype decays to a pointer like a C array,
    # shifting every following argument (NCCL then sees a garbage rank).
    lib.ncclCommInitRank.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        NcclUniqueId,
        ctypes.c_int,
    ]
    lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
    lib.ncclCommAbort.argtypes = [ctypes.c_void_p]
    lib.ncclCommGetAsyncError.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.ncclCommUserRank.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.ncclCommCount.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    for name, extra in [
        ("ncclAllReduce", [ctypes.c_int, ctypes.c_int]),  # datatype, op
        ("ncclReduceScatter", [ctypes.c_int, ctypes.c_int]),
        ("ncclAllGather", [ctypes.c_int]),  # datatype
    ]:
        fn = getattr(lib, name)
        fn.argtypes = (
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            + extra
            + [ctypes.c_void_p, ctypes.c_void_p]  # comm, stream
        )
    lib.ncclBroadcast.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,  # datatype
        ctypes.c_int,  # root
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.ncclReduce.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,  # datatype
        ctypes.c_int,  # op
        ctypes.c_int,  # root
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    for name in ["ncclSend", "ncclRecv"]:
        fn = getattr(lib, name)
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,  # datatype
            ctypes.c_int,  # peer
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
    lib.ncclGroupStart.argtypes = []
    lib.ncclGroupEnd.argtypes = []


class CclLibrary:
    """One loaded implementation of the NCCL API (NCCL or RCCL)."""

    def __init__(self, lib: ctypes.CDLL, path: str, name: str):
        self._lib = lib
        self.path = path
        self.name = name  # "NCCL" or "RCCL", for messages

    def _check(self, func_name: str, result: int):
        if result != NCCL_SUCCESS:
            detail = self._lib.ncclGetErrorString(result).decode()
            raise NcclError(func_name, result, detail)

    def version(self) -> int:
        """The runtime library's version code, e.g. 23102 for 2.31.2."""
        version = ctypes.c_int(0)
        self._check("ncclGetVersion", self._lib.ncclGetVersion(ctypes.byref(version)))
        return version.value

    def get_unique_id(self) -> bytes:
        """Generate the 128-byte communicator id (rank 0 only; share via the store)."""
        uid = NcclUniqueId()
        self._check("ncclGetUniqueId", self._lib.ncclGetUniqueId(ctypes.byref(uid)))
        # Not uid.internal: ctypes truncates c_char-array fields at the first NUL.
        return ctypes.string_at(ctypes.byref(uid), NCCL_UNIQUE_ID_BYTES)

    def init_rank(self, nranks: int, unique_id: bytes, rank: int) -> "NcclComm":
        """Collective, blocking: every rank of the clique must call concurrently.

        Binds the communicator to the device current on this thread — see
        `set_current_device`.
        """
        if len(unique_id) != NCCL_UNIQUE_ID_BYTES:
            raise ValueError(f"unique_id must be {NCCL_UNIQUE_ID_BYTES} bytes")
        uid = NcclUniqueId.from_buffer_copy(unique_id)
        handle = ctypes.c_void_p(0)
        self._check(
            "ncclCommInitRank",
            self._lib.ncclCommInitRank(ctypes.byref(handle), nranks, uid, rank),
        )
        assert handle.value is not None  # a checked init never leaves it null
        return NcclComm(self, handle.value)

    def group_start(self):
        self._check("ncclGroupStart", self._lib.ncclGroupStart())

    def group_end(self):
        self._check("ncclGroupEnd", self._lib.ncclGroupEnd())


@functools.cache
def load(api: str) -> CclLibrary:
    """The NCCL-API library for MAX's device api string ("cuda" or "hip")."""
    try:
        name = _LIBRARY_NAME_OF[api]
    except KeyError:
        raise RuntimeError(
            f"no NCCL-API collective library for the {api!r} device api; the "
            "mojo distributed backend supports NVIDIA (NCCL) and AMD (RCCL) GPUs"
        ) from None
    paths = _candidate_librccl_paths() if api == "hip" else _candidate_libnccl_paths()
    errors = []
    for path in paths:
        try:
            # RTLD_GLOBAL so a later dlopen of the same soname (for example by
            # MAX's optional vendor-CCL bridge) resolves to this exact library
            # instead of a mismatched system copy.
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            errors.append(f"{path}: {e}")
            continue
        _declare(lib)
        return CclLibrary(lib, path, name)
    raise RuntimeError(_install_help(api) + ". Tried:\n  " + "\n  ".join(errors))


class NcclComm:
    """One communicator, bound to the device that was current at init."""

    def __init__(self, ccl: CclLibrary, handle: int):
        self._ccl = ccl
        self._handle = handle
        self._aborted = False

    def destroy(self):
        if self._handle and not self._aborted:
            self._ccl._lib.ncclCommDestroy(ctypes.c_void_p(self._handle))
            self._handle = 0

    def abort(self):
        if self._handle:
            self._ccl._lib.ncclCommAbort(ctypes.c_void_p(self._handle))
            self._aborted = True
            self._handle = 0

    def async_error(self) -> int:
        err = ctypes.c_int(0)
        self._ccl._check(
            "ncclCommGetAsyncError",
            self._ccl._lib.ncclCommGetAsyncError(
                ctypes.c_void_p(self._handle), ctypes.byref(err)
            ),
        )
        return err.value

    def all_reduce(
        self, send_ptr: int, recv_ptr: int, count: int, dtype: int, op: int, stream: int
    ):
        self._ccl._check(
            "ncclAllReduce",
            self._ccl._lib.ncclAllReduce(
                send_ptr, recv_ptr, count, dtype, op, self._handle, stream
            ),
        )

    def broadcast(
        self,
        send_ptr: int,
        recv_ptr: int,
        count: int,
        dtype: int,
        root: int,
        stream: int,
    ):
        self._ccl._check(
            "ncclBroadcast",
            self._ccl._lib.ncclBroadcast(
                send_ptr, recv_ptr, count, dtype, root, self._handle, stream
            ),
        )

    def reduce(
        self,
        send_ptr: int,
        recv_ptr: int,
        count: int,
        dtype: int,
        op: int,
        root: int,
        stream: int,
    ):
        self._ccl._check(
            "ncclReduce",
            self._ccl._lib.ncclReduce(
                send_ptr, recv_ptr, count, dtype, op, root, self._handle, stream
            ),
        )

    def all_gather(
        self, send_ptr: int, recv_ptr: int, send_count: int, dtype: int, stream: int
    ):
        self._ccl._check(
            "ncclAllGather",
            self._ccl._lib.ncclAllGather(
                send_ptr, recv_ptr, send_count, dtype, self._handle, stream
            ),
        )

    def reduce_scatter(
        self,
        send_ptr: int,
        recv_ptr: int,
        recv_count: int,
        dtype: int,
        op: int,
        stream: int,
    ):
        self._ccl._check(
            "ncclReduceScatter",
            self._ccl._lib.ncclReduceScatter(
                send_ptr, recv_ptr, recv_count, dtype, op, self._handle, stream
            ),
        )

    def send(self, ptr: int, count: int, dtype: int, peer: int, stream: int):
        self._ccl._check(
            "ncclSend",
            self._ccl._lib.ncclSend(ptr, count, dtype, peer, self._handle, stream),
        )

    def recv(self, ptr: int, count: int, dtype: int, peer: int, stream: int):
        self._ccl._check(
            "ncclRecv",
            self._ccl._lib.ncclRecv(ptr, count, dtype, peer, self._handle, stream),
        )

    # Group semantics are library-global, not per communicator; exposed here
    # so a caller holding a communicator needs nothing else.
    def group_start(self):
        self._ccl.group_start()

    def group_end(self):
        self._ccl.group_end()


# --- Device selection ---------------------------------------------------------
# Both libraries pick the GPU a communicator binds to from per-thread runtime
# state. On HIP that is the current device (hipSetDevice). On CUDA it is the
# thread's current context; cuda_peer.py already dlopens libcuda for pointer
# queries, and here three more driver calls reproduce what the runtime API's
# cudaSetDevice() does.

CUDA_SUCCESS = 0


@functools.cache
def _libcuda() -> ctypes.CDLL:
    lib = ctypes.CDLL("libcuda.so.1")
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.cuDevicePrimaryCtxRetain.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
    ]
    lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    return lib


def _check_cu(func_name: str, result: int):
    if result != CUDA_SUCCESS:
        raise RuntimeError(f"{func_name} failed (CUresult={result})")


def set_current_cuda_device(ordinal: int):
    """Make `ordinal`'s primary context current on this thread (for NCCL init)."""
    lib = _libcuda()
    _check_cu("cuInit", lib.cuInit(0))
    device = ctypes.c_int(0)
    _check_cu("cuDeviceGet", lib.cuDeviceGet(ctypes.byref(device), ordinal))
    context = ctypes.c_void_p(0)
    _check_cu(
        "cuDevicePrimaryCtxRetain",
        lib.cuDevicePrimaryCtxRetain(ctypes.byref(context), device),
    )
    _check_cu("cuCtxSetCurrent", lib.cuCtxSetCurrent(context))


def set_current_device(api: str, ordinal: int):
    """Make GPU `ordinal` the calling thread's current device for `api`."""
    if api == "hip":
        hip_peer.set_device(ordinal)
    elif api == "cuda":
        set_current_cuda_device(ordinal)
    else:
        raise RuntimeError(f"no device selection for the {api!r} device api")


def device_ordinal(api: str, ptr: int) -> int | None:
    """The `api` ordinal of the GPU owning `ptr`, or None if it is not one."""
    if api == "hip":
        return hip_peer.device_ordinal(ptr)
    if api == "cuda":
        return cuda_peer.device_ordinal(ptr)
    return None
