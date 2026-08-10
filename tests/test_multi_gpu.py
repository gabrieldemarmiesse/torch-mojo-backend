"""Multi-device eager-mode contracts: direct D2D copies, thread-local
current device, and per-device wrapper TensorImpl indices (M0 of
docs/multi_gpu_training_plan.md)."""

import threading

import pytest
import torch

from torch_mojo_backend import register_mojo_devices
from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
    find_equivalent_max_device,
    get_ordered_accelerators,
    peer_access_enabled,
)

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(autouse=True)
def setup_mojo_device():
    register_mojo_devices()


def gpu_count() -> int:
    return sum(acc.label == "gpu" for acc in get_ordered_accelerators())


def require_two_gpus():
    if gpu_count() < 2:
        pytest.skip("requires at least two MAX GPUs")


def test_peer_access_enabled_on_multi_gpu():
    require_two_gpus()
    assert peer_access_enabled()


def test_cross_device_to_round_trip():
    require_two_gpus()
    x = torch.randn(64, 33)
    on_zero = x.to("mojo:0")
    on_one = on_zero.to("mojo:1")
    assert on_one.device == torch.device("mojo:1")
    torch.testing.assert_close(on_one.cpu(), x)
    back = on_one.to("mojo:0")
    torch.testing.assert_close(back.cpu(), x)


def test_cross_device_copy_preserves_source_after_mutation():
    """The transfer must be ordered before any later write to the source."""
    require_two_gpus()
    x = torch.randn(256, 256).to("mojo:0")
    expected = x.cpu()
    y = x.to("mojo:1")
    x.fill_(0.0)
    torch.testing.assert_close(y.cpu(), expected)


def test_cross_device_copy_source_freed_before_sync():
    """A source dropped right after the copy must not corrupt the transfer."""
    require_two_gpus()
    expected = torch.arange(4096, dtype=torch.float32)
    y = torch.empty(4096, device="mojo:1")
    src = expected.to("mojo:0")
    y.copy_(src)
    del src
    # New work on the source device that could recycle the freed block.
    filler = torch.full((4096,), -1.0, device="mojo:0")
    torch.testing.assert_close(y.cpu(), expected)
    assert filler.cpu().min().item() == -1.0


def test_cross_device_copy_into_strided_dest():
    require_two_gpus()
    base = torch.zeros(8, 6).to("mojo:1")
    dest_view = base.t()
    src = torch.randn(6, 8).to("mojo:0")
    dest_view.copy_(src)
    torch.testing.assert_close(base.cpu(), src.cpu().t())


def test_cross_device_copy_from_non_contiguous_source():
    require_two_gpus()
    src_base = torch.randn(6, 8).to("mojo:0")
    src_view = src_base.t()
    dest = torch.empty(8, 6, device="mojo:1")
    dest.copy_(src_view)
    torch.testing.assert_close(dest.cpu(), src_base.cpu().t())


def test_cross_device_copy_with_cast():
    require_two_gpus()
    src = torch.randn(32, 8).to("mojo:0")
    dest = torch.empty(32, 8, dtype=torch.float16, device="mojo:1")
    dest.copy_(src)
    torch.testing.assert_close(dest.cpu(), src.cpu().to(torch.float16))


def test_cross_device_copy_with_broadcast():
    require_two_gpus()
    src = torch.randn(8).to("mojo:0")
    dest = torch.empty(4, 8, device="mojo:1")
    dest.copy_(src)
    torch.testing.assert_close(dest.cpu(), src.cpu().broadcast_to(4, 8))


def test_cross_device_copy_zero_numel():
    require_two_gpus()
    src = torch.empty(0, 5).to("mojo:0")
    moved = src.to("mojo:1")
    assert moved.device == torch.device("mojo:1")
    assert moved.shape == (0, 5)


def test_all_gpu_pairs_round_trip():
    require_two_gpus()
    n = gpu_count()
    x = torch.randn(17)
    expected = x.clone()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            moved = x.to(f"mojo:{i}").to(f"mojo:{j}")
            assert moved.device == torch.device(f"mojo:{j}")
            torch.testing.assert_close(moved.cpu(), expected)


def test_wrapper_tensorimpl_device_index_matches_real_device():
    """C++ consumers (e.g. DDP's Reducer) read the TensorImpl device; it must
    carry the tensor's real index, not a phantom zero."""
    if torch.mojo.device_count() < 2:
        pytest.skip("requires two Mojo devices, including the MAX CPU device")
    base_device = torch._C.TensorBase.device
    for index in range(torch.mojo.device_count()):
        t = torch.empty(2, device=f"mojo:{index}")
        assert base_device.__get__(t) == torch.device(f"mojo:{index}")
        assert t.device == torch.device(f"mojo:{index}")


def test_new_factory_honors_explicit_cross_device():
    """new_zeros/new_empty with an explicit mojo index must allocate there."""
    require_two_gpus()
    x = torch.zeros(4, device="mojo:0")
    for factory in (
        lambda: x.new_zeros((2,), device="mojo:1"),
        lambda: x.new_empty((2,), device="mojo:1"),
        lambda: x.new_ones((2,), device="mojo:1"),
        lambda: x.new_full((2,), 3.0, device="mojo:1"),
    ):
        result = factory()
        assert result.device == torch.device("mojo:1")
        assert result._device == find_equivalent_max_device(torch.device("mojo:1"))
    # Defaulted device stays on self's device.
    assert x.new_zeros((2,)).device == torch.device("mojo:0")


def test_factory_follows_tensorimpl_device():
    """empty_like and friends on mojo:i must allocate on the real device i."""
    if torch.mojo.device_count() < 2:
        pytest.skip("requires two Mojo devices, including the MAX CPU device")
    for index in range(torch.mojo.device_count()):
        t = torch.empty(3, device=f"mojo:{index}")
        like = torch.empty_like(t)
        assert like._device == find_equivalent_max_device(torch.device(f"mojo:{index}"))


def test_current_device_is_thread_local():
    if torch.mojo.device_count() < 2:
        pytest.skip("requires two Mojo devices, including the MAX CPU device")
    main_index = torch.mojo.current_device()
    observed = {}
    barrier = threading.Barrier(2)

    def worker(rank: int):
        torch.mojo.set_device(rank)
        barrier.wait()
        observed[rank] = torch.mojo.current_device()
        tensor = torch.empty(1, device="mojo")
        observed[f"device_{rank}"] = tensor._device

    threads = [threading.Thread(target=worker, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert observed[0] == 0
    assert observed[1] == 1
    assert observed["device_0"] == find_equivalent_max_device(torch.device("mojo:0"))
    assert observed["device_1"] == find_equivalent_max_device(torch.device("mojo:1"))
    # The spawned threads' set_device must not leak into this thread.
    assert torch.mojo.current_device() == main_index


def test_backward_on_second_gpu():
    """Autograd must work on a non-zero device index (calling-thread engine)."""
    require_two_gpus()
    weight_cpu = torch.randn(16, 16, requires_grad=True)
    x_cpu = torch.randn(4, 16)

    weight = weight_cpu.detach().to("mojo:1").requires_grad_(True)
    x = x_cpu.to("mojo:1")
    torch.nn.functional.gelu(x @ weight).sum().backward()

    torch.nn.functional.gelu(x_cpu @ weight_cpu).sum().backward()
    assert weight.grad is not None
    assert weight.grad.device == torch.device("mojo:1")
    torch.testing.assert_close(weight.grad.cpu(), weight_cpu.grad, rtol=1e-4, atol=1e-4)


def test_backward_with_cross_device_graph():
    """A graph whose forward moved data between GPUs must backpropagate."""
    require_two_gpus()
    x_cpu = torch.randn(8, requires_grad=True)
    x = x_cpu.detach().to("mojo:0").requires_grad_(True)
    y = x.to("mojo:1")
    (y * y).sum().backward()
    assert x.grad is not None
    assert x.grad.device == torch.device("mojo:0")
    torch.testing.assert_close(x.grad.cpu(), 2 * x_cpu.detach())


def test_independent_streams_per_device():
    """Kernels on distinct GPUs must not require host round trips between
    launches: enqueue work on every GPU, synchronize once at the end."""
    require_two_gpus()
    n = gpu_count()
    tensors = [torch.full((512, 512), float(i), device=f"mojo:{i}") for i in range(n)]
    results = [t @ t for t in tensors]
    for i, r in enumerate(results):
        expected = float(i) * float(i) * 512.0
        assert r.cpu()[0, 0].item() == pytest.approx(expected)
