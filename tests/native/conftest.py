import contextlib
import os

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


def is_metal(device: str) -> bool:
    """Whether `device` (a `mojo:<index>` string) is an Apple GPU -- the
    inverse of `skip_if_metal`'s own check, for a test that wants to assert
    the Metal-only decline itself rather than skip past it."""
    idx = int(device.rsplit(":", 1)[-1])
    accelerators = list(get_accelerators())
    return idx < len(accelerators) and accelerators[idx].api == "metal"


def flush_subnormals_on_metal(
    t: torch.Tensor, device: str, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """`t` with its float32 / bfloat16 subnormals replaced by zeros of the
    same sign when `device` is an Apple GPU, unchanged otherwise.

    Apple GPUs flush float32 subnormals to zero, on the operands and on the
    result of float arithmetic, whatever the compile options: torch MPS's own
    kernels and a `torch.mps.compile_shader` kernel built in
    `MTLMathModeSafe` with `metal::precise::sqrt` give the same zeros as the
    mojo device (bit for bit: `x * 0.5`, `x * 1`, `x + 0` and `sqrt(x)` over
    +-1, +-2, +-0x7FFFFF and the smallest normals, measured on an M4).
    bfloat16 is computed in float32 there, so its subnormals flush too;
    float16 subnormals are normal float32 values and survive. The result is
    flushed when its exact value is tiny, before rounding: 0x3F7FFFFF times
    the smallest normal is 0 there, where IEEE rounds it up to that normal.
    So pass the exact result in float64 with `dtype` the result dtype, then
    round: `flush(flush(a).double() * b, device, torch.float32).float()`.
    Only arithmetic flushes: a kernel that selects or copies its input
    (log1p's tiny-input path) hands a float32 subnormal through unchanged.
    """
    dtype = t.dtype if dtype is None else dtype
    if not is_metal(device) or dtype not in (torch.float32, torch.bfloat16):
        return t
    subnormal = (t != 0) & (t.abs() < torch.finfo(dtype).tiny)
    return torch.where(subnormal, t * 0, t)


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


def require_stock_cuda():
    """For a test that runs or regenerates data recorded on stock CUDA
    torch: return when stock NVIDIA CUDA torch is importable here
    (`torch.version.cuda` set; a ROCm wheel's cuda.is_available() does not
    count, the data is CUDA's), else skip -- except on a CI runner with an
    NVIDIA GPU (`CI` set, a mojo accelerator whose api is "cuda"), where a
    skip would leave the recorded data unchecked, so it fails instead."""
    if torch.cuda.is_available() and torch.version.cuda is not None:
        return
    nvidia = any(a.api == "cuda" for a in get_accelerators())
    if os.environ.get("CI") and nvidia:
        pytest.fail(
            "stock CUDA torch expected on this runner: the recorded CUDA data"
            " would go unchecked"
        )
    pytest.skip("needs stock CUDA torch")


def skip_float64_on_metal(device: str, *dtypes: torch.dtype | None):
    """Skip a case involving float64 on an Apple GPU, which has no float64.
    `device` is any device string (a `Conf.device` too): only a mojo device
    on Metal skips."""
    if torch.float64 in dtypes and device.startswith("mojo:"):
        skip_if_metal(device, "Apple GPUs have no float64")


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


@contextlib.contextmanager
def ran_all(*op_names: str):
    """Assert that each of `op_names` ran as a native boxed kernel."""
    native.op_counting(True)
    before = native.op_counts()
    yield
    after = native.op_counts()
    for name in op_names:
        assert after.get(name, 0) > before.get(name, 0), (
            f"{name} did not run natively (counted: {sorted(after)})"
        )


def native_count(name: str) -> int:
    """How many times `aten::<name>` ran natively since the last reset."""
    return native.op_count(f"aten::{name}")


def reset_op_counts():
    native.op_counting(True)
    native.op_counts_reset()
