"""Fast eager-mode kernels for mojo_device, compiled as CPython extensions.

The `.mojo` modules in this package are imported through `mojo.importer`
(the official Mojo import hook): the first import compiles each module with
`mojo build --emit shared-lib` and caches the resulting extension under
`__mojocache__/` (content addressed — recompiles only when a `.mojo` file
in this directory changes).

An op call here is one CPython extension call that receives raw data
pointers (from `TorchMojoTensor._ptr`) plus sizes/dtypes as plain ints, and
enqueues a kernel on MAX's own DeviceContext, so it stays correctly ordered
with every other MAX driver operation on that device.

`tensor_holder` is the ownership module: the Mojo `TensorHolder` type owns
each device allocation (stream-ordered alloc/free) and provides the host
transfer + strided copy/fill primitives. The kernel modules are grouped by
category (elementwise, nn, data movement, matmul, conv) and each is
imported lazily on the first call of an op in that category, so a given
workload only compiles the categories it uses. `matmul_ops` and `conv_ops`
contain pure-Mojo GPU kernels plus correctness-grade CPU paths for the MAX
CPU device.
"""

import fcntl
import hashlib
import importlib
import sys
from pathlib import Path

import mojo.importer  # noqa: F401  — installs the .mojo meta-path importer
from max import driver

_PACKAGE_DIR = Path(__file__).parent
_CACHE_DIR = _PACKAGE_DIR / "__mojocache__"

_MOJO_MODULES = (
    "tensor_holder",
    "activation_backward_ops",
    "activation_forward_ops",
    "dropout_ops",
    "embedding_backward_ops",
    "elementwise_ops",
    "nn_ops",
    "data_movement_ops",
    "logic_ops",
    "matmul_ops",
    "bf16_matmul_ops",
    "tf32_matmul_ops",
    "conv_ops",
    "reduction_ops",
    "loss_ops",
    "softmax_backward_ops",
    "normalization_forward_ops",
    "normalization_backward_ops",
    "optimizer_ops",
    "sdpa_backward_ops",
    "flash_attention_ops",
)


def _mojo_sources_hash() -> str:
    """Content hash of the package's .mojo sources.

    Must match `mojo.importer._calculate_mojo_source_hash` so we can
    predict the cache filename the importer will look for.
    """
    hasher = hashlib.sha256()
    for file_path in sorted(_PACKAGE_DIR.rglob("*.mojo")):
        hasher.update(str(file_path.relative_to(_PACKAGE_DIR)).encode())
        hasher.update(file_path.read_bytes())
    return hasher.hexdigest()[:16]


def _import_mojo_module(name: str):
    """Import (compiling on first use) one .mojo extension module.

    `mojo.importer` has no cross-process lock: concurrent cold-cache
    imports (e.g. pytest-xdist workers) would each run `mojo build`
    writing to the same cache file. Serialize the compile with a file
    lock; once the cache file for the current source hash exists, the
    import is a plain dlopen and needs no lock.
    """
    cache_file = _CACHE_DIR / f"{name}.hash-{_mojo_sources_hash()}.so"
    if cache_file.is_file():
        return importlib.import_module(f"torch_mojo_backend.eager_kernels.{name}")

    _CACHE_DIR.mkdir(exist_ok=True)
    with open(_CACHE_DIR / ".compile.lock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if not cache_file.is_file():
            print(
                f"torch-mojo-backend: compiling eager-mode Mojo kernels ({name}) "
                "(first use only, takes ~30s, cached afterwards)...",
                file=sys.stderr,
            )
        return importlib.import_module(f"torch_mojo_backend.eager_kernels.{name}")


def __getattr__(name: str):
    if name in _MOJO_MODULES:
        # `tensor_holder` registers the process-wide `TensorHolder` /
        # `TensorSpec` Python type objects that every other module's spec ops
        # take and return. Those modules only *use* the types (never
        # re-register them — a duplicate `add_type` aborts the process), so
        # `tensor_holder` must be imported and finalized before any of them,
        # whatever op the workload happens to touch first. Loading via `.to()`
        # (which hits `tensor_holder.alloc_from_host`) usually gets this right
        # by luck, but a factory-first workload (e.g. `torch.ones(..., FillSpec)`)
        # would otherwise import `elementwise_ops` first and fail with
        # "No Python type object registered for ... TensorSpec".
        if name != "tensor_holder" and "tensor_holder" not in globals():
            globals()["tensor_holder"] = _import_mojo_module("tensor_holder")
        module = _import_mojo_module(name)
        globals()[name] = module  # later lookups skip __getattr__
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_CTX_PTR_CACHE: dict[driver.Device, int] = {}


def _ctx_ptr(device: driver.Device) -> int:
    ptr = _CTX_PTR_CACHE.get(device)
    if ptr is None:
        ptr = device._device_context_ptr()
        _CTX_PTR_CACHE[device] = ptr
    return ptr
