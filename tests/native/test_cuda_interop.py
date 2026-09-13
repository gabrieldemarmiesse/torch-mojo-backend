"""CUDA interop on the mojo device (torch_mojo_backend.cuda_interop).

A package with compiled CUDA kernels needs a CUDA build of torch, CUDA
tensors and a CUDA stream; the mojo device owns the memory and the streams.
These tests check the two halves of the bridge: the aliases really are the
same memory (both directions, views included), and an ExternalStream over the
mojo current stream really does order a torch.cuda kernel with ours. They also
pin the contract that binds the two -- an alias carries no ordering of its
own, so it is refused outside an `on_mojo_stream()` block for its own device.

Every test needs a CUDA build of torch whose driver initializes, so the whole
module skips on the CPU wheel the project normally uses.
"""

import inspect
import os
import subprocess
import sys

import pytest
import torch

from torch_mojo_backend import cuda_interop

# `torch.mojo` itself: registered at run time, so it is spelled through
# the module object here to keep the file legible to the type checker.
from torch_mojo_backend.native import device_module as mojo

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a working CUDA build of torch"
)


@pytest.fixture
def gpu(mojo_gpu):
    return mojo_gpu


def test_as_cuda_is_the_same_memory(gpu):
    m = torch.arange(12, dtype=torch.float32, device=gpu).reshape(3, 4)
    with cuda_interop.on_mojo_stream(gpu):
        c = cuda_interop.as_cuda(m)
        assert c.device.type == "cuda"
        assert c.data_ptr() == m.data_ptr()
        assert c.shape == m.shape and c.stride() == m.stride() and c.dtype == m.dtype
        torch.testing.assert_close(c.cpu(), m.cpu())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_as_cuda_round_trips_every_dtype(gpu, dtype):
    m = torch.ones(8, 3, device=gpu, dtype=dtype)
    with cuda_interop.on_mojo_stream(gpu):
        back = cuda_interop.as_mojo(cuda_interop.as_cuda(m))
        assert back.dtype == dtype and back.data_ptr() == m.data_ptr()
        torch.testing.assert_close(back.cpu(), m.cpu())


def test_as_cuda_preserves_views(gpu):
    """Shape, strides, dtype and the addressed elements, not
    `storage_offset()`: torch's DLPack export normalizes the offset into the
    pointer, so a view's alias starts at the view's first element."""
    m = torch.arange(12, dtype=torch.float32, device=gpu).reshape(3, 4)
    transposed = m.t()
    sliced = m.reshape(-1)[5:]
    with cuda_interop.on_mojo_stream(gpu):
        for v in (transposed, sliced):
            c = cuda_interop.as_cuda(v)
            assert c.data_ptr() == v.data_ptr() and c.stride() == v.stride()
            assert c.storage_offset() == 0
            torch.testing.assert_close(c.cpu(), v.cpu())
    assert sliced.storage_offset() == 5


def test_a_cuda_kernel_writes_through_the_alias(gpu):
    m = torch.zeros(64, device=gpu)
    with cuda_interop.on_mojo_stream(gpu):
        cuda_interop.as_cuda(m).add_(5.0)
    mojo.synchronize()
    torch.testing.assert_close(m.cpu(), torch.full((64,), 5.0))


def test_a_mojo_kernel_reads_cuda_allocator_memory(gpu):
    """The other direction: the memory belongs to torch's CUDA caching
    allocator, and a mojo kernel runs over it."""
    c = torch.arange(6, dtype=torch.float32, device="cuda") * 10
    with cuda_interop.on_mojo_stream(gpu):
        m = cuda_interop.as_mojo(c)
        assert m.device.type == "mojo" and m.data_ptr() == c.data_ptr()
        torch.testing.assert_close((m + 1).cpu(), c.cpu() + 1)


def test_the_alias_keeps_the_mojo_tensor_alive(gpu):
    m = torch.ones(1024, device=gpu)
    with cuda_interop.on_mojo_stream(gpu):
        alias = cuda_interop.as_cuda(m)
        del m
        mojo.synchronize()
        alias.mul_(7.0)
        mojo.synchronize()
        assert float(alias.sum().cpu()) == 7168.0


def test_the_mojo_cpu_device_has_no_cuda_alias():
    cpu_index = mojo.device_count() - 1
    m = torch.ones(4, device=f"mojo:{cpu_index}")
    with pytest.raises(ValueError, match="CPU device"):
        cuda_interop.as_cuda(m)


def test_on_mojo_stream_installs_the_mojo_stream(gpu):
    s = torch.Stream(device=gpu)
    with mojo.stream(s):
        with cuda_interop.on_mojo_stream(gpu):
            assert torch.cuda.current_stream().cuda_stream == mojo.stream_native_handle(
                s
            )
        assert torch.cuda.current_stream().cuda_stream != mojo.stream_native_handle(s)


def test_a_second_gpu_maps_to_its_cuda_ordinal(gpu):
    """The mojo device index is the CUDA ordinal. `on_mojo_stream` selects
    that device for the duration and restores the previous one."""
    if mojo.device_count() - 1 < 2:  # the last index is the MAX CPU device
        pytest.skip("needs two GPUs")
    x = torch.arange(8, dtype=torch.float32, device="mojo:1")
    before = torch.cuda.current_device()
    with cuda_interop.on_mojo_stream(1):
        assert torch.cuda.current_device() == 1
        alias = cuda_interop.as_cuda(x)
        assert alias.device == torch.device("cuda", 1)
        assert alias.data_ptr() == x.data_ptr()
        assert torch.cuda.current_stream().cuda_stream == mojo.stream_native_handle(
            mojo.current_stream(1)
        )
        alias.add_(100.0)
    assert torch.cuda.current_device() == before
    mojo.synchronize(1)
    torch.testing.assert_close(x.cpu(), torch.arange(8, dtype=torch.float32) + 100)


def test_gradients_do_not_cross_an_alias(gpu):
    """DLPack carries no autograd history, which is why `cuda_autograd`
    exists: the alias is always a leaf. (Nothing reads the memory here, which
    is what `unordered=True` is for.)"""
    x = torch.randn(4, 4, device=gpu, requires_grad=True)
    alias = cuda_interop.as_cuda(x, unordered=True)
    assert not alias.requires_grad and alias.grad_fn is None


def test_an_alias_outside_a_stream_block_is_refused(gpu):
    """An alias is memory and nothing else -- no stream handoff (the raw
    capsule skips torch's `__dlpack__(stream=)` negotiation) and no allocator
    stream tracking (a `from_blob` deleter is foreign to both allocators). The
    one thing that orders it is `on_mojo_stream`, so that is the default and
    `unordered=True` is the deliberate opt-out."""
    m = torch.ones(4, device=gpu)
    with pytest.raises(RuntimeError, match="outside on_mojo_stream"):
        cuda_interop.as_cuda(m)
    cuda_interop.as_cuda(m, unordered=True)
    with cuda_interop.on_mojo_stream(gpu):
        alias = cuda_interop.as_cuda(m)
    with pytest.raises(RuntimeError, match="outside on_mojo_stream"):
        cuda_interop.as_mojo(alias)
    cuda_interop.as_mojo(alias, unordered=True)
    with cuda_interop.on_mojo_stream(gpu):
        assert cuda_interop.as_mojo(alias).device.type == "mojo"


def test_an_alias_never_names_a_device_other_than_its_own(gpu):
    """The mojo index IS the CUDA ordinal and is not a caller's choice:
    relabelling a pointer as another GPU would copy nothing and hand out
    memory that GPU cannot address."""
    assert "index" not in inspect.signature(cuda_interop.as_cuda).parameters
    assert "index" not in inspect.signature(cuda_interop.as_mojo).parameters
    if mojo.device_count() - 1 < 2:
        pytest.skip("needs two GPUs")
    with cuda_interop.on_mojo_stream(1):
        m = torch.ones(4, device="mojo:1")
        assert cuda_interop.as_cuda(m).device == torch.device("cuda", 1)
        c = torch.ones(4, device="cuda:1")
        assert cuda_interop.as_mojo(c).device == torch.device("mojo", 1)


def test_an_alias_of_another_device_than_the_stream_is_refused(gpu):
    """One `on_mojo_stream` block is one device: another device's stream
    orders nothing here."""
    if mojo.device_count() - 1 < 2:
        pytest.skip("needs two GPUs")
    x = torch.ones(4, device="mojo:1")
    with (
        cuda_interop.on_mojo_stream(0),
        pytest.raises(RuntimeError, match="another device"),
    ):
        cuda_interop.as_cuda(x)


def test_launches_interleave_without_synchronizing(gpu):
    """mojo kernel, torch.cuda kernel, mojo kernel, one stream, no sync in
    between: the result is only right if all three were ordered."""
    n = 1 << 22
    a = torch.ones(n, device=gpu)
    s = torch.Stream(device=gpu)
    with mojo.stream(s):
        b = a * 3.0
        with cuda_interop.on_mojo_stream(gpu):
            cuda_interop.as_cuda(b).add_(1.0)
        out = b * 2.0
    torch.accelerator.synchronize()
    assert float(out.sum().cpu()) / n == 8.0


def test_call_cuda_converts_both_ways(gpu):
    def cuda_only(x, y):
        assert x.is_cuda and y.is_cuda
        return x @ y

    x = torch.randn(64, 32, device=gpu)
    y = torch.randn(32, 16, device=gpu)
    out = cuda_interop.call_cuda(cuda_only, x, y)
    assert isinstance(out, torch.Tensor) and out.device.type == "mojo"
    torch.testing.assert_close(out.cpu(), x.cpu() @ y.cpu(), rtol=1e-4, atol=1e-4)


def test_call_cuda_in_place_lands_in_the_mojo_tensor(gpu):
    def fill(t):
        assert t.is_cuda
        t.fill_(3.0)

    m = torch.zeros(16, device=gpu)
    cuda_interop.call_cuda(fill, m)
    mojo.synchronize()
    torch.testing.assert_close(m.cpu(), torch.full((16,), 3.0))


def test_an_op_that_returns_an_argument_returns_the_caller_s_tensor(gpu):
    """In-place and `out=` ops must hand back the object they were given, not
    a fresh alias of the same memory."""
    a = torch.ones(8, device=gpu)
    b = torch.full((8,), 2.0, device=gpu)

    assert cuda_interop.call_cuda(lambda x, y: x.add_(y), a, b) is a
    out = torch.empty(8, device=gpu)
    assert cuda_interop.call_cuda(torch.add, a, b, out=out) is out
    mojo.synchronize()
    torch.testing.assert_close(out.cpu(), torch.full((8,), 5.0))


def test_call_cuda_needs_a_mojo_tensor():
    with pytest.raises(ValueError, match="at least one mojo tensor"):
        cuda_interop.call_cuda(torch.add, 1, 2)


def test_fallback_runs_an_op_the_mojo_device_lacks(gpu):
    """`aten::index_select` has no mojo kernel: without the fallback it
    raises, with it the CUDA kernel runs on the mojo tensors."""
    x = torch.randn(8, 5, device=gpu)
    idx = torch.tensor([0, 3, 3, 7], device=gpu)
    with pytest.raises(NotImplementedError):
        torch.index_select(x, 0, idx)
    with cuda_interop.cuda_fallback():
        out = torch.index_select(x, 0, idx)
    assert out.device.type == "mojo"
    torch.testing.assert_close(out.cpu(), torch.index_select(x.cpu(), 0, idx.cpu()))


def test_fallback_carries_autograd(gpu):
    """`index_select` has no mojo kernel in either direction: forward and
    backward (`index_add`) both go through CUDA, on an autograd graph that
    never leaves mojo tensors."""
    torch.manual_seed(0)
    x = torch.randn(8, 5, device=gpu, requires_grad=True)
    idx = torch.tensor([0, 3, 3, 7], device=gpu)
    with cuda_interop.cuda_fallback():
        torch.index_select(x, 0, idx).sum().backward()
    xc = x.detach().cpu().requires_grad_()
    torch.index_select(xc, 0, idx.cpu()).sum().backward()
    assert x.grad is not None and xc.grad is not None
    torch.testing.assert_close(x.grad.cpu(), xc.grad)


def test_conv2d_trains_through_the_explicit_route(gpu):
    """conv2d's forward is `aten::convolution` (a mojo op); its backward is
    not, and ATen routes it to `convolution_backward_overrideable`, which the
    fallback cannot see -- `_EXPLICIT_ROUTES` registers that one by hand."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16, 16, device=gpu, requires_grad=True)
    w = torch.randn(4, 3, 3, 3, device=gpu, requires_grad=True)
    with cuda_interop.cuda_fallback():
        torch.nn.functional.conv2d(x, w, padding=1).sum().backward()
    xc = x.detach().cpu().requires_grad_()
    wc = w.detach().cpu().requires_grad_()
    torch.nn.functional.conv2d(xc, wc, padding=1).sum().backward()
    assert x.grad is not None and w.grad is not None
    assert xc.grad is not None and wc.grad is not None
    torch.testing.assert_close(x.grad.cpu(), xc.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(w.grad.cpu(), wc.grad, rtol=1e-3, atol=1e-3)


def test_a_registered_op_that_declines_does_not_reach_the_fallback(gpu):
    """The boundary of the design: the dispatcher picks a fallback only where
    no kernel is registered, so an op the backend registers and then declines
    at run time (here `aten::convolution` with transposed=True) still raises.
    """
    x = torch.randn(2, 3, 8, 8, device=gpu)
    w = torch.randn(3, 4, 3, 3, device=gpu)
    with cuda_interop.cuda_fallback(), pytest.raises(NotImplementedError):
        torch.nn.functional.conv_transpose2d(x, w, stride=2)


def test_fallback_is_only_a_fallback(gpu):
    """An op the mojo device does implement must not be diverted."""
    a = torch.ones(4, device=gpu)
    with cuda_interop.cuda_fallback():
        before = cuda_interop.fallback_counts().get("aten.add.Tensor", 0)
        torch.testing.assert_close((a + a).cpu(), torch.full((4,), 2.0))
        assert cuda_interop.fallback_counts().get("aten.add.Tensor", 0) == before


def test_the_fallback_goes_away_with_its_block(gpu):
    x = torch.randn(8, 5, device=gpu)
    idx = torch.tensor([0, 3], device=gpu)
    with cuda_interop.cuda_fallback():
        torch.index_select(x, 0, idx)
    with pytest.raises(NotImplementedError):
        torch.index_select(x, 0, idx)


def test_enable_cuda_fallback_lasts_for_the_process(gpu, tmp_path):
    """A fresh process, because the process-wide form cannot be undone. Its
    two `torch.library.Library` objects must be kept alive: each has a
    finalizer that resets its registrations."""
    script = tmp_path / "fallback_probe.py"
    script.write_text(
        "import gc, torch\n"
        "from torch_mojo_backend import register_mojo_devices\n"
        "register_mojo_devices()\n"
        "from torch_mojo_backend import cuda_interop\n"
        "cuda_interop.enable_cuda_fallback()\n"
        "cuda_interop.enable_cuda_fallback()  # idempotent\n"
        "gc.collect()\n"
        "x = torch.randn(8, 5, device='mojo:0', requires_grad=True)\n"
        "idx = torch.tensor([0, 3], device='mojo:0')\n"
        "torch.index_select(x, 0, idx).sum().backward()\n"
        "w = torch.randn(4, 3, 3, 3, device='mojo:0', requires_grad=True)\n"
        "a = torch.randn(2, 3, 16, 16, device='mojo:0', requires_grad=True)\n"
        "torch.nn.functional.conv2d(a, w, padding=1).sum().backward()\n"
        "print('FALLBACK OK')\n"
    )
    # PYTHONHOME / PYTHONEXECUTABLE: MAX's Python interop sets them in this
    # process, and inherited they point a fresh interpreter at the wrong
    # prefix ("No module named 'encodings'"). `loader.mojo` unsets them around
    # its own subprocesses for the same reason.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONHOME", "PYTHONEXECUTABLE")
    }
    r = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )
    assert "FALLBACK OK" in r.stdout, r.stdout + r.stderr


def test_causal_conv1d_forward_and_backward(gpu):
    """A real third-party CUDA package end to end: its own kernels, on mojo
    tensors, against its own reference implementation."""
    interface = pytest.importorskip("causal_conv1d.causal_conv1d_interface")
    torch.manual_seed(0)
    batch, dim, seqlen, width = 2, 64, 128, 4
    x = torch.randn(batch, dim, seqlen, device=gpu, dtype=torch.float32)
    weight = torch.randn(dim, width, device=gpu, dtype=torch.float32)
    bias = torch.randn(dim, device=gpu, dtype=torch.float32)

    fn = cuda_interop.cuda_autograd(
        lambda x, w, b: interface.causal_conv1d_fwd_function(
            x, w, b, None, None, None, True
        ),
        lambda g, x, w, b: interface.causal_conv1d_bwd_function(
            x, w, b, g, None, None, None, None, False, True
        )[:3],
    )

    x = x.requires_grad_()
    weight = weight.requires_grad_()
    bias = bias.requires_grad_()
    out = fn(x, weight, bias)
    out.sum().backward()

    xr = x.detach().cpu().requires_grad_()
    wr = weight.detach().cpu().requires_grad_()
    br = bias.detach().cpu().requires_grad_()
    ref = interface.causal_conv1d_ref(xr, wr, br, activation="silu")
    ref.sum().backward()

    assert x.grad is not None and weight.grad is not None and bias.grad is not None
    assert xr.grad is not None and wr.grad is not None and br.grad is not None
    torch.testing.assert_close(out.cpu(), ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x.grad.cpu(), xr.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(weight.grad.cpu(), wr.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(bias.grad.cpu(), br.grad, rtol=2e-3, atol=2e-3)
