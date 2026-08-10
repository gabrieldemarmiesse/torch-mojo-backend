"""Eager allreduce across GPUs on Modular's P2P comm kernels (M1 of
docs/multi_gpu_training_plan.md)."""

import pytest
import torch

from torch_mojo_backend import distributed as mojo_dist
from torch_mojo_backend import register_mojo_devices
from torch_mojo_backend.mojo_device.torch_mojo_tensor import get_ordered_accelerators

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(autouse=True)
def setup_mojo_device():
    register_mojo_devices()


def gpu_count() -> int:
    return sum(acc.label == "gpu" for acc in get_ordered_accelerators())


def world_sizes() -> list[int]:
    n = gpu_count()
    sizes = [size for size in (2, 3, n) if size <= n]
    return sorted(set(sizes))


def require_two_gpus():
    if gpu_count() < 2:
        pytest.skip("requires at least two MAX GPUs")


def make_inputs(
    world: int, numel: int, dtype: torch.dtype
) -> tuple[list[torch.Tensor], torch.Tensor]:
    cpu_tensors = [
        torch.randn(numel, dtype=torch.float32).to(dtype) for _ in range(world)
    ]
    expected_sum = sum(t.to(torch.float32) for t in cpu_tensors)
    device_tensors = [t.to(f"mojo:{i}") for i, t in enumerate(cpu_tensors)]
    return device_tensors, expected_sum


# Sizes chosen to cross the kernels' latency-bound (1-stage) and
# bandwidth-bound (2-stage) dispatch regimes, plus non-SIMD-multiple tails.
@pytest.mark.parametrize("numel", [7, 8 * 1024, 8 * 1024 + 5, 1024 * 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_all_reduce_sum(numel, dtype):
    require_two_gpus()
    for world in world_sizes():
        tensors, expected_sum = make_inputs(world, numel, dtype)
        mojo_dist.all_reduce(tensors, op="sum")
        expected = expected_sum.to(dtype)
        for i, tensor in enumerate(tensors):
            assert tensor.device == torch.device(f"mojo:{i}")
            torch.testing.assert_close(
                tensor.cpu(),
                expected,
                rtol=2e-2 if dtype != torch.float32 else 2e-5,
                atol=2e-2 if dtype != torch.float32 else 1e-5,
            )


def test_all_reduce_mean():
    require_two_gpus()
    world = world_sizes()[-1]
    tensors, expected_sum = make_inputs(world, 4096, torch.float32)
    mojo_dist.all_reduce(tensors, op="mean")
    expected = expected_sum / world
    for tensor in tensors:
        torch.testing.assert_close(tensor.cpu(), expected, rtol=2e-5, atol=1e-5)


def test_all_reduce_out_leaves_inputs_untouched():
    require_two_gpus()
    tensors, expected_sum = make_inputs(2, 1024, torch.float32)
    originals = [t.cpu() for t in tensors]
    outs = mojo_dist.all_reduce_out(tensors, op="sum")
    for tensor, original in zip(tensors, originals):
        torch.testing.assert_close(tensor.cpu(), original)
    for out in outs:
        torch.testing.assert_close(out.cpu(), expected_sum, rtol=2e-5, atol=1e-5)


def test_all_reduce_reuses_signal_buffers():
    """Back-to-back collectives share the cached Signal buffers; the barrier
    counters must survive reuse."""
    require_two_gpus()
    world = world_sizes()[-1]
    for _ in range(5):
        tensors, expected_sum = make_inputs(world, 2048, torch.float32)
        mojo_dist.all_reduce(tensors, op="sum")
        for tensor in tensors:
            torch.testing.assert_close(tensor.cpu(), expected_sum, rtol=2e-5, atol=1e-5)


def test_all_reduce_2d_and_noncontiguous():
    require_two_gpus()
    cpu_a = torch.randn(32, 16)
    cpu_b = torch.randn(32, 16)
    a = cpu_a.to("mojo:0")
    b_base = cpu_b.t().contiguous().to("mojo:1")
    b = b_base.t()  # non-contiguous view, logical shape (32, 16)
    mojo_dist.all_reduce([a, b], op="sum")
    expected = cpu_a + cpu_b
    torch.testing.assert_close(a.cpu(), expected, rtol=2e-5, atol=1e-5)
    torch.testing.assert_close(b.cpu(), expected, rtol=2e-5, atol=1e-5)


def test_all_reduce_validation_errors():
    require_two_gpus()
    x = torch.randn(8).to("mojo:0")
    y = torch.randn(8).to("mojo:1")
    with pytest.raises(ValueError, match="need one tensor per GPU"):
        mojo_dist.all_reduce([x])
    with pytest.raises(ValueError, match="must be on mojo:1"):
        mojo_dist.all_reduce([x, torch.randn(8).to("mojo:0")])
    with pytest.raises(ValueError, match="shape"):
        mojo_dist.all_reduce([x, torch.randn(4).to("mojo:1")])
    with pytest.raises(ValueError, match="dtype"):
        mojo_dist.all_reduce([x, torch.randn(8).to("mojo:1").to(torch.float16)])
    with pytest.raises(ValueError, match="unsupported reduce op"):
        mojo_dist.all_reduce([x, y], op="max")


def test_all_reduce_rejects_out_of_order_devices():
    """The comm kernels index peers by device id (ctx.id() is the rank), so
    tensors must arrive ordered mojo:0..mojo:n-1."""
    require_two_gpus()
    x = torch.randn(8).to("mojo:0")
    y = torch.randn(8).to("mojo:1")
    with pytest.raises(ValueError, match="must be on mojo:0"):
        mojo_dist.all_reduce([y, x])
    if gpu_count() >= 3:
        z = torch.randn(8).to("mojo:2")
        with pytest.raises(ValueError, match="must be on mojo:0"):
            mojo_dist.all_reduce([y, z])


def test_all_reduce_with_offset_view_inputs():
    """Contiguous views at unaligned storage offsets must be handled (the
    kernels vector-load from the base pointer)."""
    require_two_gpus()
    base_a = torch.randn(1025).to("mojo:0")
    base_b = torch.randn(1025).to("mojo:1")
    a, b = base_a[1:], base_b[1:]
    expected = (base_a.cpu()[1:] + base_b.cpu()[1:]).clone()
    head_before = base_a.cpu()[0].clone()
    mojo_dist.all_reduce([a, b], op="sum")
    torch.testing.assert_close(a.cpu(), expected, rtol=2e-5, atol=1e-5)
    torch.testing.assert_close(b.cpu(), expected, rtol=2e-5, atol=1e-5)
    # The element outside the view window must be untouched.
    torch.testing.assert_close(base_a.cpu()[0], head_before)


def test_all_reduce_int_dtype_rejected():
    require_two_gpus()
    x = torch.ones(8, dtype=torch.int32).to("mojo:0")
    y = torch.ones(8, dtype=torch.int32).to("mojo:1")
    with pytest.raises(ValueError, match="unsupported collective dtype"):
        mojo_dist.all_reduce([x, y])
