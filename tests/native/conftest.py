import contextlib

import pytest
import torch

from torch_mojo_backend import get_accelerators, native, register_mojo_devices


@pytest.fixture(autouse=True, scope="session")
def _mojo_devices_registered():
    """Every native test runs on the registered mojo device (a test that
    only uses the `mojo_device` / `mojo_gpu` fixtures of tests/conftest.py
    gets it from there as well; the call is idempotent)."""
    register_mojo_devices()


def side_stream_or_skip(device: str) -> torch.Stream:
    """A second stream on `device`, or a skip when the runtime has none.

    On Metal, torch.Stream returns the default stream, as PyTorch MPS does.
    Tests requiring independent queues cannot establish cross-stream
    ordering there. Check the backend explicitly so CUDA/ROCm construction
    failures remain test failures.
    """
    skip_if_metal(device, "Apple GPU stream objects share the default stream")
    return torch.Stream(device=device)


def skip_if_metal(device: str, reason: str):
    """Skip a case that is correct and by design on Apple's Metal backend.

    `device` is a `mojo:<index>` string; the index selects which entry of
    `get_accelerators()` to check. Several declines are real Apple-GPU-only
    limits (no float64 on the GPU)
    and one (cumsum's non-trailing-dim / bf16-f16 route) is really "only
    ever measured on NVIDIA" and happens to show up as Metal on this box --
    see the call sites for which. Never a blanket try/except: every call
    names its own reason, and CUDA/ROCm runs are untouched because the index
    they check is never Metal.
    """
    if is_metal(device):
        pytest.skip(reason)


def is_metal(device: str) -> bool:
    """Whether the `mojo:<index>` device is an Apple GPU (Metal)."""
    idx = int(device.rsplit(":", 1)[-1])
    accelerators = list(get_accelerators())
    return idx < len(accelerators) and accelerators[idx].api == "metal"


@contextlib.contextmanager
def ran(*op_names: str):
    """Assert that at least one of `op_names` ran as a native boxed kernel.

    Used for the ops with no `aten_functions` twin for `CallChecker` to key
    on.
    """
    native.op_counting(True)
    before = {name: native.op_count(name) for name in op_names}
    yield
    assert any(native.op_count(name) > before[name] for name in op_names), (
        f"none of {op_names} ran natively"
    )
