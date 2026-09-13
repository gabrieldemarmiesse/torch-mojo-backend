"""Triton kernels on the mojo device (torch_mojo_backend.triton_driver):
Triton compiles and launches through its own GPU backend; the driver only
answers the device and stream questions with the mojo device, and is
installed automatically when triton's runtime is imported after
register_mojo_devices() -- on a torch with no working CUDA build, since the
active driver is process-wide."""

import ctypes
import subprocess
import sys

import pytest
import torch

from torch_mojo_backend import get_accelerators, triton_driver
from torch_mojo_backend.native import device_module
from torch_mojo_backend.triton_driver import enable_triton

triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")
from torch._library.triton import triton_op, wrap_triton  # noqa: E402 -- after the skip
from triton.runtime import driver as active_driver  # noqa: E402 -- idem


@pytest.fixture
def mojo_triton(mojo_gpu):
    enable_triton()
    return mojo_gpu


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(
        out_ptr + offs,
        tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
        mask=mask,
    )


def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    _add_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, out, x.numel(), BLOCK=1024)
    return out


def test_triton_kernel_runs_on_mojo_tensors(mojo_triton):
    x = torch.randn(100_003, device=mojo_triton)
    y = torch.randn(100_003, device=mojo_triton)
    torch.testing.assert_close(_add(x, y).cpu(), x.cpu() + y.cpu())


def test_triton_launch_follows_the_current_mojo_stream(mojo_triton):
    s = torch.Stream(device=mojo_triton)
    with device_module.stream(s):
        a = torch.ones(1 << 22, device=mojo_triton) * 3
        b = _add(a, a)
        c = b * 2
    torch.accelerator.synchronize()
    assert float(c.sum().cpu()) == 12 * (1 << 22)


def test_triton_launch_on_a_second_device(mojo_triton):
    gpus = [d for d in get_accelerators() if getattr(d, "api", "") != "cpu"]
    if len(gpus) < 2:
        pytest.skip("needs two GPUs")
    # like torch.cuda: a Triton launch goes to the CURRENT device, so the
    # caller selects it; the tensors' device is not consulted
    x = torch.randn(4096, device="mojo:1")
    with device_module.device(1):
        y = _add(x, x)
    torch.testing.assert_close(y.cpu(), 2 * x.cpu())
    assert device_module.current_device() == 0


@triton.jit
def _double_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 2, mask=mask)


def test_the_second_device_loads_and_launches_in_its_own_context(mojo_triton):
    """Triton's `loadBinary` and its launcher both use whatever driver context
    is current and never check its device, and Inductor's DeviceGuard moves
    only mojo's TLS device. So both must run under the selected device's
    context -- and put back the one they found, which is what this checks by
    leaving device 0's current across a device-1 compile and launch."""
    if triton_driver.accelerator_api() != "cuda":
        pytest.skip("CUDA driver contexts")
    if len([d for d in get_accelerators() if getattr(d, "api", "") != "cpu"]) < 2:
        pytest.skip("needs two GPUs")
    cuda = ctypes.CDLL("libcuda.so.1")
    context_0 = triton_driver._stream_context(
        device_module.stream_native_handle(device_module.current_stream(0))
    )
    assert cuda.cuCtxSetCurrent(context_0) == 0

    x = torch.randn(4096, device="mojo:1")
    out = torch.empty_like(x)
    with device_module.device(1):
        _double_kernel[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024)
    torch.testing.assert_close(out.cpu(), 2 * x.cpu())

    after = ctypes.c_void_p()
    cuda.cuCtxGetCurrent(ctypes.byref(after))
    assert after.value == context_0.value


def test_the_automatic_hook_stands_aside_for_a_cuda_torch(mojo_gpu, monkeypatch):
    """The active Triton driver is process-wide and answers before a launch's
    arguments are looked at, so installing ours next to a working torch.cuda
    would send that process's CUDA launches to the mojo device and stream too.
    `enable_triton()` stays available, explicitly."""
    installed = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(active_driver, "set_active", installed.append)
    meta_path = list(sys.meta_path)

    triton_driver.install_triton_hook()

    assert installed == []
    assert sys.meta_path == meta_path


@triton.autotune(
    configs=[triton.Config({"BLOCK": 256}), triton.Config({"BLOCK": 1024})], key=["n"]
)
@triton.jit
def _scale_kernel(x_ptr, out_ptr, n, factor, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) * factor, mask=mask)


def test_autotune_benchmarks_through_the_mojo_device_interface(mojo_triton):
    """The autotuner times each config with do_bench: events, synchronize and
    the L2-flush buffer all come from the mojo device."""
    x = torch.randn(1 << 20, device=mojo_triton)
    out = torch.empty_like(x)
    _scale_kernel[lambda meta: (triton.cdiv(x.numel(), meta["BLOCK"]),)](
        x, out, x.numel(), 2.5
    )
    torch.testing.assert_close(out.cpu(), x.cpu() * 2.5)
    ms = triton.testing.do_bench(lambda: _add(x, x))
    assert ms > 0


def test_driver_is_installed_on_import_after_registration(mojo_gpu, tmp_path):
    """A fresh process: register the device, import triton afterwards, launch;
    no explicit enable_triton() call."""
    code = """
import torch
from torch_mojo_backend.native import device_module
from torch_mojo_backend import register_mojo_devices
register_mojo_devices()
import triton, triton.language as tl
from triton.runtime import driver
assert type(driver.active).__name__ == "MojoCudaDriver", type(driver.active)
@triton.jit
def k(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=offs < n) + 1, mask=offs < n)
x = torch.zeros(3000, device="mojo:0")
k[(3,)](x, x.numel(), BLOCK=1024)
assert x.cpu().sum().item() == 3000
print("HOOK OK")
"""
    script = tmp_path / "hook_probe.py"  # a file: triton reads the kernel's source
    script.write_text(code)
    r = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=900
    )
    assert "HOOK OK" in r.stdout, r.stdout[-500:] + r.stderr[-1500:]


@triton.jit
def _plain_scale_kernel(x_ptr, out_ptr, n, factor, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) * factor, mask=mask)


def test_triton_op_traces_under_dynamo(mojo_triton):
    """torch.library.triton_op: a Triton kernel as a custom op that dynamo
    traces through wrap_triton; eager and under the eager/aot_eager backends."""

    @triton_op("mojo_test::scale", mutates_args={})
    def scale(x: torch.Tensor, factor: float) -> torch.Tensor:
        out = torch.empty_like(x)
        wrap_triton(_plain_scale_kernel)[(triton.cdiv(x.numel(), 1024),)](
            x, out, x.numel(), factor, BLOCK=1024
        )
        return out

    x = torch.randn(5000, device=mojo_triton)
    torch.testing.assert_close(scale(x, 3.0).cpu(), x.cpu() * 3.0)
    for backend in ("eager", "aot_eager"):
        torch._dynamo.reset()
        f = torch.compile(lambda t: scale(t, 2.0) + 1, backend=backend)
        torch.testing.assert_close(f(x).cpu(), x.cpu() * 2.0 + 1)
