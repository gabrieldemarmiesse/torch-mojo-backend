"""ctypes binding for the NCCL API — NCCL on NVIDIA, RCCL on AMD.

Why ctypes and not torch's own NCCL: this backend must work with a CPU-only
torch install (see "Rules about the eager mode" in AGENTS.md), so we cannot
rely on torch.distributed.ProcessGroupNCCL or torch.cuda being functional.

Two libraries implement one C API (`nccl.h` / `rccl.h`: same function names,
same enum values, same 128-byte unique id), so one binding serves both; only
where the library comes from and how the target GPU is selected differ:

- NVIDIA: `libnccl.so.2` from the nvidia-nccl-cu12 wheel. It statically links
  the CUDA runtime and dlopens libcuda.so.1 by itself, so this needs nothing
  beyond the wheel and a driver. The target GPU is the CUDA context current
  on the calling thread;
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
import os
from pathlib import Path

from torch_mojo_backend.mojo_device import hip_peer

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
# "mojo": build (first use only, cached) and use libmojoccl.so -- the
# in-repo NCCL-API implementation (torch_mojo_backend/distributed/mojoccl) --
# instead of vendor NCCL/RCCL. Same C ABI, same `_declare()` argtypes below.
_CCL_ENV = "TORCH_MOJO_BACKEND_CCL"

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


def uses_mojoccl() -> bool:
    return os.environ.get(_CCL_ENV, "").lower() == "mojo"


def vendor_name() -> str:
    """ "rccl" when MAX's accelerators are AMD (HIP api), else "nccl"."""
    from torch_mojo_backend.torch_compile_backend.utils import (  # noqa: PLC0415 -- imports max.driver; keep it off the import path
        get_accelerators,
    )

    return (
        "rccl"
        if any(getattr(d, "api", "") == "hip" for d in get_accelerators())
        else "nccl"
    )


def library_path() -> str:
    """Path of the collectives library: mojoccl (built from Mojo on first use,
    TORCH_MOJO_BACKEND_CCL=mojo) or the vendor NCCL / RCCL."""
    if uses_mojoccl():
        from torch_mojo_backend.distributed.mojoccl_build import (  # noqa: PLC0415 -- builds a library; keep it lazy
            ensure_built,
        )

        return ensure_built()
    candidates = (
        _candidate_librccl_paths()
        if vendor_name() == "rccl"
        else _candidate_libnccl_paths()
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    raise RuntimeError(
        "no NCCL/RCCL library found (looked at "
        + ", ".join(candidates)
        + "); install nvidia-nccl-cu12 "
        "or point TORCH_MOJO_BACKEND_NCCL_LIB / TORCH_MOJO_BACKEND_RCCL_LIB at one, or set "
        f"{_CCL_ENV}=mojo for the in-repo Mojo collectives"
    )
