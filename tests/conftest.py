import fcntl
import importlib.metadata
import math
import os
from collections.abc import Callable
from pathlib import Path

os.environ["MODULAR_TELEMETRY_ENABLED"] = "0"
os.environ["MAX_USE_EAGER_INTERPRETER"] = "1"
os.environ["TORCH_MOJO_BACKEND_TESTING"] = "1"
# Type-check the whole package under tests; production imports leave it off.
os.environ.setdefault("TORCH_MOJO_BACKEND_BEARTYPE", "1")
import pytest

# must be called before importing torch_mojo_backend
pytest.register_assert_rewrite("torch_mojo_backend.testing")


import torch
from max.driver import Device
from max.dtype import DType
from mojo.paths import _build_mojo_source_package

from torch_mojo_backend import get_accelerators, register_mojo_devices
from torch_mojo_backend.testing import CallChecker, Conf
from torch_mojo_backend.torch_compile_backend import compiler

os.environ["TORCH_MOJO_BACKEND_VERBOSE"] = "1"


# TODO: remove this when
# https://github.com/modular/modular/issues/5495 is fixed
def _mojo_source_package_stamp(source: Path) -> str:
    """Identity of one precompiled source package: toolchain + sources."""
    import hashlib

    try:
        toolchain = importlib.metadata.version("mojo-compiler")
    except importlib.metadata.PackageNotFoundError:
        toolchain = "unknown"
    digest = hashlib.sha256(toolchain.encode())
    for path in sorted(source.rglob("*.mojo")):
        info = path.stat()
        digest.update(
            f"{path.relative_to(source)}:{info.st_mtime_ns}:{info.st_size}".encode()
        )
    return digest.hexdigest()


def _build_mojo_source_package_once(source: Path) -> Path:
    """`_build_mojo_source_package` with the concurrency it is missing.

    Upstream precompiles to one fixed path derived from the SOURCE PATH
    hash only — no content or toolchain identity, no lock, no atomic
    rename — and rebuilds unconditionally. Under pytest-xdist every worker
    rewrites the same .mojoc while other workers' engine sessions import
    it, which surfaces as "MAXG_addKernelPackage: failed to import
    kernels" on a torn read; after a nightly bump, a node's stale package
    is equally torn between toolchains. Serialize builders with an flock
    and rebuild only when the sidecar stamp (toolchain + source mtimes)
    does not match, so exactly one worker builds and everyone else reuses.
    """
    stamp = _mojo_source_package_stamp(source)
    lock_path = (
        Path(os.environ.get("TMPDIR", "/tmp")) / f".modular_{os.getuid()}_mojo_pkg.lock"
    )
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        import hashlib
        import tempfile

        path_hash = hashlib.md5(str(source.absolute()).encode()).hexdigest()
        package = (
            Path(tempfile.gettempdir())
            / f".modular_{os.getuid()}"
            / "mojo_pkg"
            / f"mojo_pkg_{path_hash}.mojoc"
        )
        sidecar = Path(str(package) + ".stamp")
        try:
            if package.exists() and sidecar.read_text() == stamp:
                return package
        except OSError:
            pass
        built = _build_mojo_source_package(source)
        sidecar.write_text(stamp)
        return built


compiler.paths_to_mojo_kernels[0] = _build_mojo_source_package_once(
    compiler.paths_to_mojo_kernels[0]
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request, cuda_available: bool):
    device_name = request.param
    if not cuda_available and device_name == "cuda":
        pytest.skip("CUDA not available")
    return device_name


@pytest.fixture(
    params=[
        # Enable when pytorch supports it
        # Conf("mojo:cpu", True),
        # Conf("mojo:gpu", True),
        Conf("mojo:cpu", False)
        # Conf("mojo:gpu", False),
        # Conf("cpu", True),
        # Conf("cuda", True),
    ]
)
def conf(request, mojo_gpu_available: bool, cuda_available: bool):
    conf = request.param
    # to use mojo:gpu, we need to have a max supported gpu
    if conf.device == "mojo:gpu" and not mojo_gpu_available:
        pytest.skip("You do not have a GPU supported by MAX")
    if conf.device == "cuda" and not cuda_available:
        pytest.skip("Pytorch CUDA not available")

    # known issues:
    if conf.device.startswith("mojo") and conf.compile:
        pytest.xfail("Known issue: mojo device with compilation is not supported yet")

    if conf.device.startswith("mojo"):
        conf.device = conf.device.replace("gpu", "0")
        conf.device = conf.device.replace("cpu", str(len(list(get_accelerators())) - 1))
        # Make sure the device is initialized
        register_mojo_devices()

    if conf.device == "cuda":
        conf.device += ":0"

    return conf


@pytest.fixture
def cuda_available() -> bool:
    return torch.cuda.is_available()


@pytest.fixture
def mojo_gpu_available() -> bool:
    return len(list(get_accelerators())) > 1


@pytest.fixture(params=[(3,), (2, 3)])
def tensor_shapes(request):
    return request.param


@pytest.fixture(autouse=True)
def reset_compiler():
    torch.compiler.reset()
    yield


@pytest.fixture(params=["cpu", "gpu"])
def mojo_device(request, mojo_gpu_available: bool):
    if request.param == "cpu":
        yield (f"mojo:{len(get_accelerators()) - 1}")
    else:
        if not mojo_gpu_available:
            pytest.skip("You do not have a GPU supported by MAX")
        yield ("mojo:0")


@pytest.fixture
def mojo_gpu(mojo_gpu_available: bool) -> str:
    """GPU mojo device only — for ops whose fast path is GPU-gated.

    Shared by every module that needs one: four copies of this fixture used
    to disagree about whether they registered the devices first.
    """
    if not mojo_gpu_available:
        pytest.skip("You do not have a GPU supported by MAX")
    register_mojo_devices()  # idempotent; some callers have no autouse setup
    return "mojo:0"


@pytest.fixture
def fake_mojo_tensor() -> Callable[..., torch.Tensor]:
    """Build a `TorchMojoTensor` whose payload metadata is pure fiction.

    Host-only tests of the kernel wiring need a tensor that answers every
    metadata question `aten_fast` asks (`_shape`, `_dtype`, `_ptr`, ...)
    without owning device memory, so the prologue of a kernel route can be
    exercised on a machine with no GPU. The pointer is never dereferenced:
    such tests replace the native `call` entry point.
    """
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
        TorchMojoTensor,
        _row_major_strides,
        _torch_dtype_of,
    )

    def make(
        device: Device,
        *,
        dtype: DType = DType.float32,
        shape: tuple[int, ...] = (2, 3),
        strides: tuple[int, ...] | None = None,
        ptr: int = 1,
    ) -> torch.Tensor:
        shape = tuple(shape)
        strides = _row_major_strides(shape) if strides is None else tuple(strides)
        tensor = torch.Tensor._make_wrapper_subclass(
            TorchMojoTensor,
            shape,
            strides=strides,
            storage_offset=0,
            dtype=_torch_dtype_of(dtype),
            layout=torch.strided,
            device="cpu",
            requires_grad=False,
        )
        tensor._holder = object()
        tensor._ptr = ptr
        tensor._device = device
        tensor._dtype = dtype
        tensor._shape = shape
        tensor._strides = strides
        tensor._offset = 0
        tensor._itemsize = dtype.size_in_bytes
        tensor._numel = math.prod(shape)
        tensor._is_contiguous = True
        return tensor

    return make


def pytest_make_parametrize_id(val):
    """Custom ID generation for parametrized tests"""

    if isinstance(val, torch.dtype):
        return str(val).split(".")[-1]
    if isinstance(val, Conf):
        return str(val)
    # Return None to fall back to default behavior for other types
    return None


@pytest.fixture()
def call_checker():
    call_checker_instance = CallChecker()
    yield call_checker_instance
    call_checker_instance.check_was_called()


def require_cuda_autograd(device: str) -> None:
    """Skip when this process can no longer run a CUDA backward.

    `at::getAccelerator()` names exactly one accelerator device type, and it
    returns PrivateUse1 as soon as a PrivateUse1 backend is registered --
    which `register_mojo_devices()` does, process-wide and with no way to
    undo it. From then on `Node::stream()` finds no input metadata on the
    accelerator device type for a CUDA node and returns nullopt, so the
    autograd engine trips
    `TORCH_INTERNAL_ASSERT(opt_ready_stream && opt_parent_stream)`
    (engine.cpp) for *any* backward over CUDA tensors. Verified with no
    compilation involved: after `register_mojo_devices()`, a bare
    `(torch.randn(4, 4, device="cuda", requires_grad=True) * 2).sum()
    .backward()` raises that assert.

    So a test that runs or traces a CUDA backward -- which every compile of
    an `nn.Module` with trainable parameters does, via AOTAutograd's joint
    graph -- needs a process where CUDA is still torch's accelerator.
    """
    if device != "cuda":
        return
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is not None and accelerator.type != "cuda":
        pytest.skip(
            f"torch's accelerator is {accelerator.type!r}, not 'cuda': a "
            "PrivateUse1 backend (the mojo device) was registered earlier in "
            "this process, which breaks CUDA autograd inside PyTorch itself."
        )
