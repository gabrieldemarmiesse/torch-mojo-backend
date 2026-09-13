"""`torch.compile(backend="inductor")` on the mojo device: Inductor generates
Triton kernels, `torch_mojo_backend.inductor` tells it how to reach our device
and stream, and every result is checked against the same function on CPU."""

import pytest
import torch
import torch._inductor.metrics
import torch._inductor.utils

pytest.importorskip("triton")

from torch_mojo_backend import monkeypatching  # noqa: E402
from torch_mojo_backend.inductor import enable_inductor, get_raw_stream  # noqa: E402
from torch_mojo_backend.native import (
    device_module,  # noqa: E402 -- `torch.mojo` itself, under a name ty can resolve
)


@pytest.fixture
def mojo_inductor(mojo_gpu):
    enable_inductor()
    torch._dynamo.reset()
    yield mojo_gpu
    torch._dynamo.reset()


def test_inductor_refuses_beside_a_working_cuda_torch(monkeypatch):
    """`GPU_TYPES` is one process-wide list and `get_gpu_type()` asserts at
    most one of its entries is available, so "mojo" cannot join "cuda" there
    without breaking Inductor for that process's CUDA workloads too."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch._inductor.utils, "GPU_TYPES", ["cuda", "mps", "xpu", "mtia"]
    )
    with pytest.raises(RuntimeError, match="CPU torch wheel"):
        enable_inductor()
    monkeypatching.add_mojo_to_the_inductor_gpu_types()  # the patch stands aside too
    assert "mojo" not in torch._inductor.utils.GPU_TYPES


def _compiled_matches_cpu(fn, mojo_inputs, cpu_inputs, **tolerance):
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)
    got = compiled(*mojo_inputs)
    expected = fn(*cpu_inputs)
    torch.testing.assert_close(got.cpu(), expected, **tolerance)
    return got


def test_pointwise(mojo_inductor):
    def fn(a, b):
        return (a * b + 1).relu()

    torch._inductor.metrics.reset()
    a, b = torch.randn(512, 65), torch.randn(512, 65)
    _compiled_matches_cpu(fn, (a.to(mojo_inductor), b.to(mojo_inductor)), (a, b))
    # one generated Triton kernel for the whole chain, not three eager ops
    assert torch._inductor.metrics.generated_kernel_count == 1


def test_pointwise_broadcast_and_dtype(mojo_inductor):
    def fn(a, b):
        return (a * b + 1).relu().to(torch.float16)

    a, b = torch.randn(37, 129), torch.randn(129)
    _compiled_matches_cpu(fn, (a.to(mojo_inductor), b.to(mojo_inductor)), (a, b))


def test_reduction_sum_last_dim(mojo_inductor):
    def fn(x):
        return x.sum(-1)

    x = torch.randn(129, 1027)
    _compiled_matches_cpu(fn, (x.to(mojo_inductor),), (x,), rtol=1e-5, atol=1e-4)


def test_softmax(mojo_inductor):
    def fn(x):
        return torch.softmax(x, dim=-1)

    x = torch.randn(64, 513)
    _compiled_matches_cpu(fn, (x.to(mojo_inductor),), (x,), rtol=1e-5, atol=1e-5)


def test_matmul_chain_falls_back_to_our_aten_mm(mojo_inductor):
    """Inductor lowers mm to an extern kernel: the matmul runs as our native
    aten::mm and only the relu/sum around it are generated Triton."""

    def fn(x, w):
        return (x @ w).relu().sum()

    x, w = torch.randn(128, 96), torch.randn(96, 64)
    _compiled_matches_cpu(
        fn, (x.to(mojo_inductor), w.to(mojo_inductor)), (x, w), atol=2e-3, rtol=1e-4
    )


def test_training_step_backward(mojo_inductor):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)
    )
    x = torch.randn(32, 64)
    target = torch.randn(32, 10)

    def step(model, x, target):
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        return loss

    mojo_model = torch.nn.Sequential(
        torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)
    ).to(mojo_inductor)
    mojo_model.load_state_dict(
        {k: v.to(mojo_inductor) for k, v in model.state_dict().items()}
    )

    loss = torch.compile(step, backend="inductor")(
        mojo_model, x.to(mojo_inductor), target.to(mojo_inductor)
    )
    expected = step(model, x, target)

    torch.testing.assert_close(loss.cpu(), expected, rtol=1e-4, atol=1e-4)
    for ours, theirs in zip(mojo_model.parameters(), model.parameters(), strict=True):
        assert ours.grad is not None and theirs.grad is not None
        torch.testing.assert_close(ours.grad.cpu(), theirs.grad, rtol=1e-3, atol=1e-4)


def test_launch_follows_the_current_mojo_stream(mojo_inductor):
    """A compiled kernel launched inside `torch.mojo.stream(s)` is ordered with
    our own kernels on that stream: the multiply feeding it runs on s, and
    only s is synchronized before the result is read."""

    def fn(a, b):
        return (a * b + 1).relu()

    compiled = torch.compile(fn, backend="inductor", fullgraph=True)
    s = torch.Stream(device=mojo_inductor)
    with device_module.stream(s):
        # the handle the generated wrapper passes to kernel.run(stream=...)
        assert get_raw_stream(0) == device_module.stream_native_handle(s)
        a = torch.full((1 << 22,), 3.0, device=mojo_inductor) * 2
        b = torch.ones(1 << 22, device=mojo_inductor)
        out = compiled(a, b) * 2
    assert get_raw_stream(0) != device_module.stream_native_handle(s)
    s.synchronize()
    assert float(out.sum().cpu()) == 14.0 * (1 << 22)
