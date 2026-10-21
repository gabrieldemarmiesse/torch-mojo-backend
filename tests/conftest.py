# ruff: noqa: E402 -- the environment variables below must be set before the imports
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

os.environ["MODULAR_TELEMETRY_ENABLED"] = "0"
os.environ["MAX_USE_EAGER_INTERPRETER"] = "1"
os.environ["TORCH_MOJO_BACKEND_TESTING"] = "1"
# Every Mojo build under the tests (backend library, op extensions, kernel
# specializations, mojoccl) fails on a compiler warning instead of hiding it
# in captured stderr; off by default for users.
os.environ["TORCH_MOJO_BACKEND_WERROR"] = "1"
import pytest

# must be called before importing torch_mojo_backend
pytest.register_assert_rewrite("torch_mojo_backend.testing")


import torch

from torch_mojo_backend import get_accelerators, register_mojo_devices
from torch_mojo_backend.testing import CallChecker, Conf

os.environ["TORCH_MOJO_BACKEND_VERBOSE"] = "1"


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def device(request, cuda_available: bool):
    device_name = request.param
    if not cuda_available and device_name == "cuda":
        pytest.skip("CUDA not available")
    return device_name


@pytest.fixture(
    params=[
        # Enable when pytorch supports it
        # Conf("mojo:gpu", True),
        Conf("mojo:gpu", False)
        # Conf("cpu", True),
        # Conf("cuda", True),
    ]
)
def conf(request):
    conf = request.param
    # Requested lazily: a static dependency on a device fixture would mark the
    # `Conf("cpu", True)` cases `gpu` (repo-root conftest.py).
    if conf.device == "mojo:gpu" and not request.getfixturevalue("mojo_gpu_available"):
        pytest.skip("You do not have a GPU supported by MAX")
    if conf.device == "cuda" and not request.getfixturevalue("cuda_available"):
        pytest.skip("Pytorch CUDA not available")

    # known issues:
    if conf.device.startswith("mojo") and conf.compile:
        pytest.xfail("Known issue: mojo device with compilation is not supported yet")

    if conf.device.startswith("mojo"):
        conf.device = conf.device.replace("gpu", "0")
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
    return len(list(get_accelerators())) > 0


@pytest.fixture(params=[(3,), (2, 3)])
def tensor_shapes(request):
    return request.param


@pytest.fixture(autouse=True)
def reset_compiler():
    torch.compiler.reset()
    yield


@pytest.fixture(autouse=True)
def seed_rngs():
    """Seed torch's and Python's global RNGs to 0 before every test.

    Otherwise an unseeded `torch.randn` draws whatever the tests before it
    left in the global generator, so a test's data depends on which tests
    share its CI shard, and adding a test anywhere reshuffles every shard.
    A constant keeps a test's data unchanged across renames too. A test
    that seeds its own generator is unaffected.
    """
    random.seed(0)
    torch.manual_seed(0)


@pytest.fixture
def mojo_device(mojo_gpu: str) -> str:
    """There is no CPU-backed mojo device any more; this is now just an
    alias of `mojo_gpu`, kept so every existing caller stays unchanged."""
    return mojo_gpu


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


# MAX lowers an fp32 matmul to TF32 tensor cores on NVIDIA GPUs while torch
# eager defaults to full fp32, so a graph containing a matmul cannot be
# compared against eager at assert_close's fp32 defaults on GPU
# (`test_compile_matmul` in test_compile_mojo_device.py makes the same
# allowance, and so do the compiled convolution tests). Verified exactly: for `x @ w + b` on cuda the backend's output
# is bit-identical to torch's own `allow_tf32=True` result.
#
# The numbers below are the measured tf32-vs-fp32 envelope for these shapes
# over 2000 random draws: max absolute gap 4.8e-3 for one matmul and 1.5e-2
# for the chained pair in `test_get_attr_multiple_parameters`. The relative
# gap is unbounded (outputs cancel to near zero), which is why atol carries
# the tolerance. atol=2e-2 is ~1.3x the measured worst case and still ~50x
# below the O(1) error an actually wrong matmul produces on N(0, 1) data.
# CPU keeps assert_close's exact fp32 defaults.
class Tolerance(TypedDict, total=False):
    rtol: float
    atol: float


def matmul_tolerance(device: str) -> Tolerance:
    if device == "cpu":
        return {}
    return {"rtol": 1e-2, "atol": 2e-2}


def require_cuda_autograd(device: str):
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


def python_env(preload: Path | None = None) -> dict[str, str]:
    """os.environ for a child interpreter. MAX's Python interop sets
    PYTHONHOME / PYTHONEXECUTABLE to its own prefix in this process; inherited,
    they break the child's startup ("No module named 'encodings'").
    `preload` is prepended to LD_PRELOAD."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONHOME", "PYTHONEXECUTABLE")
    }
    if preload is not None:
        env["LD_PRELOAD"] = str(preload) + (
            ":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else ""
        )
    return env


def run_python(
    *args: str, timeout: float, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run `sys.executable *args` and assert it exits 0."""
    proc = subprocess.run(
        [sys.executable, *args],
        env=python_env() if env is None else env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc
