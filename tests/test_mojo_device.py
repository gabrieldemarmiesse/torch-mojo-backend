"""Unit tests for basic mojo_device functionality"""

import gc
import io
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import cast

import max.driver
import pytest
import torch
from torch.optim.optimizer import _default_to_fused_or_foreach

from torch_mojo_backend import TorchMojoTensor, mojo_backend, register_mojo_devices
from torch_mojo_backend.eager_kernels import aten_fast
from torch_mojo_backend.mojo_device import (
    cuda_peer,
    dlpack,
    torch_mojo_device_module,
    torch_mojo_tensor as mojo_tensor_module,
)
from torch_mojo_backend.mojo_device.torch_mojo_device_module import (
    _reserve_philox_state,
)
from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
    _PENDING_H2D,
    find_equivalent_max_device,
    get_ordered_accelerators,
)
from torch_mojo_backend.torch_compile_backend import utils

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(autouse=True)
def setup_max_device():
    """Setup mojo_device for all tests"""
    register_mojo_devices()


def test_mojo_is_the_default_torch_accelerator():
    assert torch.accelerator.current_accelerator(check_available=True) == torch.device(
        "mojo"
    )


def test_torch_accelerator_synchronize_uses_mojo_device_module(monkeypatch):
    calls = []
    original_synchronize = torch_mojo_device_module.synchronize
    original_device = torch_mojo_device_module.current_device()

    def recording_synchronize(device=None):
        calls.append(device)
        return original_synchronize(device)

    monkeypatch.setattr(torch_mojo_device_module, "synchronize", recording_synchronize)

    try:
        torch.accelerator.synchronize()
        torch.accelerator.synchronize("mojo")
        torch.accelerator.synchronize(0)

        if torch.accelerator.device_count() > 1:
            torch_mojo_device_module.set_device(1)
            torch.accelerator.synchronize()
    finally:
        torch_mojo_device_module.set_device(original_device)

    assert calls[:3] == [original_device, original_device, 0]
    if torch.accelerator.device_count() > 1:
        assert calls[3:] == [1]
    with pytest.raises(ValueError, match="doesn't match the current accelerator mojo"):
        torch.accelerator.synchronize("cpu")


def test_tensor_to_max_device(mojo_device):
    """Test converting regular tensor to mojo_device"""
    # Create CPU tensor
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0])

    # Convert to mojo_device
    mojo_tensor = cpu_tensor.to(mojo_device)

    # Check type and properties
    assert isinstance(mojo_tensor, TorchMojoTensor)
    assert mojo_tensor.shape == (3,)
    assert mojo_tensor.dtype == torch.float32


def test_max_tensor_to_cpu(mojo_device):
    """Test converting MaxTensor back to CPU"""
    # Create tensor on mojo_device
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
    mojo_tensor = cpu_tensor.to(mojo_device)

    # Convert back to CPU
    result = mojo_tensor.to("cpu")

    # Check result
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, cpu_tensor)


def test_factory_arange(mojo_device):
    """Test torch.arange with mojo_device"""
    tensor = torch.arange(5, device=mojo_device)

    assert isinstance(tensor, TorchMojoTensor)
    assert tensor.shape == (5,)

    # Convert to CPU to check values
    cpu_result = tensor.to("cpu")
    expected = torch.arange(5)
    torch.testing.assert_close(cpu_result, expected)


@pytest.mark.xfail(reason="Fixme")
def test_factory_rand(mojo_device):
    """Test torch.rand with mojo_device"""
    tensor = torch.rand(3, 4, device=mojo_device)

    assert isinstance(tensor, TorchMojoTensor)
    assert tensor.shape == (3, 4)

    # Check that values are in [0, 1] range when converted to CPU
    cpu_result = tensor.to("cpu")
    assert torch.all(cpu_result >= 0)
    assert torch.all(cpu_result <= 1)


def test_factory_empty(mojo_device):
    """Test torch.empty with mojo_device"""
    tensor = torch.empty(2, 3, device=mojo_device)

    assert isinstance(tensor, TorchMojoTensor)
    assert tensor.shape == (2, 3)


def test_device_string_variations():
    """Test different mojo device string formats"""
    # Basic mojo device
    t1 = torch.tensor([1.0]).to("mojo")
    assert isinstance(t1, TorchMojoTensor)

    # With index (should also work)
    t2 = torch.tensor([1.0]).to("mojo:0")
    assert isinstance(t2, TorchMojoTensor)


def test_indexless_mojo_device_uses_and_restores_current_device():
    """An indexless mojo target follows torch.mojo's current device."""
    accelerators = get_ordered_accelerators()
    if len(accelerators) < 2:
        pytest.skip("requires two Mojo devices, including the MAX CPU device")

    original_index = torch_mojo_device_module.current_device()
    alternate_index = (original_index + 1) % len(accelerators)
    try:
        torch_mojo_device_module.set_device(alternate_index)

        assert (
            find_equivalent_max_device(torch.device("mojo"))
            == accelerators[alternate_index]
        )
        empty_tensor = torch.empty(1, device="mojo")
        assert isinstance(empty_tensor, TorchMojoTensor)
        assert empty_tensor._device == accelerators[alternate_index]
    finally:
        torch_mojo_device_module.set_device(original_index)

    assert torch_mojo_device_module.current_device() == original_index
    assert (
        find_equivalent_max_device(torch.device("mojo")) == accelerators[original_index]
    )


@pytest.mark.xfail(reason="TODO: add pretty repr and str")
def test_tensor_properties(mojo_device):
    """Test that MaxTensor preserves tensor properties"""
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    mojo_tensor = original.to(mojo_device)

    assert mojo_tensor.shape == (2, 2)
    assert mojo_tensor.dtype == torch.float64
    assert mojo_tensor.device == torch.device(mojo_device)

    # Test repr
    repr_str = repr(mojo_tensor)
    assert mojo_device in repr_str
    assert "size=(2, 2)" in repr_str


def test_round_trip_conversion(mojo_device):
    """Test CPU -> mojo_device -> CPU round trip"""
    original = torch.tensor([1.0, 2.0, 3.0, 4.0])

    # Round trip
    mojo_tensor = original.to(mojo_device)
    result = mojo_tensor.to("cpu")

    # Should be equal
    torch.testing.assert_close(result, original)


def test_non_blocking_cpu_to_mojo_transfers(mojo_device):
    """Both to() and copy_() honor PyTorch's non_blocking transfer API."""
    source = torch.arange(4096, dtype=torch.float32)

    via_to = source.to(mojo_device, non_blocking=True)
    via_copy = torch.empty_like(via_to)
    via_copy.copy_(source, non_blocking=True)

    torch.accelerator.synchronize(mojo_device)
    torch.testing.assert_close(via_to.cpu(), source)
    torch.testing.assert_close(via_copy.cpu(), source)


def test_non_blocking_cpu_source_lifetime(mojo_device):
    """An async upload remains valid after its temporary CPU source dies."""
    source = torch.arange(1 << 20, dtype=torch.int32)
    expected = source.clone()
    uploaded = source.to(mojo_device, non_blocking=True)
    assert isinstance(uploaded, TorchMojoTensor)
    if uploaded._device.label == "gpu":
        assert uploaded._device in _PENDING_H2D

    del source
    # Encourage the CPU allocator to reuse the released storage while the H2D
    # operation may still be queued.
    for _ in range(8):
        torch.empty_like(expected).fill_(-1)

    torch.accelerator.synchronize(mojo_device)
    assert uploaded._device not in _PENDING_H2D
    torch.testing.assert_close(uploaded.cpu(), expected)


def test_synchronized_transfers_reap_completed_cpu_sources(mojo_device):
    """Later blocking H2D/D2H operations release completed async sources."""
    first = torch.arange(4096).to(mojo_device, non_blocking=True)
    assert isinstance(first, TorchMojoTensor)
    torch.zeros(4096).to(mojo_device)
    assert first._device not in _PENDING_H2D

    second = torch.arange(4096).to(mojo_device, non_blocking=True)
    assert isinstance(second, TorchMojoTensor)
    torch.testing.assert_close(second.cpu(), torch.arange(4096))
    assert second._device not in _PENDING_H2D


def test_non_blocking_h2d_does_not_drain_prior_gpu_work(mojo_device):
    """An async upload returns without waiting for older default-stream work."""
    max_device = find_equivalent_max_device(torch.device(mojo_device))
    if max_device.label != "gpu":
        pytest.skip("requires a MAX GPU")

    a = torch.randn(4096, 4096).to(mojo_device)
    b = torch.randn(4096, 4096).to(mojo_device)
    torch_mojo_device_module.synchronize(mojo_device)

    # Establish a conservative duration for the work placed ahead of H2D.
    _ = a @ b
    torch_mojo_device_module.synchronize(mojo_device)
    started = time.perf_counter()
    _ = a @ b
    torch_mojo_device_module.synchronize(mojo_device)
    matmul_seconds = time.perf_counter() - started

    delayed = a @ b
    started = time.perf_counter()
    uploaded = torch.arange(4096).to(mojo_device, non_blocking=True)
    upload_return_seconds = time.perf_counter() - started

    assert upload_return_seconds < matmul_seconds * 0.5
    torch_mojo_device_module.synchronize(mojo_device)
    torch.testing.assert_close(uploaded.cpu(), torch.arange(4096))
    assert delayed.shape == (4096, 4096)

    # A multi-megabyte pageable source must use MAX-owned pinned staging.
    # Passing the pageable pointer straight to the stream appears asynchronous
    # for tiny inputs but drains preceding work once the transfer is large.
    elements = 1 << 20
    source = torch.arange(elements, dtype=torch.float32)
    delayed = a @ b
    started = time.perf_counter()
    large_uploaded = source.to(mojo_device, non_blocking=True)
    large_upload_return_seconds = time.perf_counter() - started

    assert large_upload_return_seconds < matmul_seconds * 0.5
    torch_mojo_device_module.synchronize(mojo_device)
    torch.testing.assert_close(large_uploaded.cpu(), source)
    assert delayed.shape == (4096, 4096)

    # Exercise the staged H2D + strided device-copy path as well. Its temporary
    # CPU source must stay alive while both operations wait behind prior work.
    destination_storage = torch.empty((elements, 2), device=mojo_device)
    destination = destination_storage[:, 1]
    assert isinstance(destination, TorchMojoTensor)
    destination.copy_(torch.zeros(elements), non_blocking=True)
    torch_mojo_device_module.synchronize(mojo_device)

    expected = source.clone()
    # The staged strided path has a few milliseconds of legitimate host-side
    # setup on this backend.  Put a longer, calibrated queue ahead of it so the
    # assertion distinguishes that setup from an accidental stream drain.
    queue_repeats = 4
    started = time.perf_counter()
    for _ in range(queue_repeats):
        _ = a @ b
    torch_mojo_device_module.synchronize(mojo_device)
    queued_matmul_seconds = time.perf_counter() - started

    delayed = [a @ b for _ in range(queue_repeats)]
    started = time.perf_counter()
    destination.copy_(source, non_blocking=True)
    strided_upload_return_seconds = time.perf_counter() - started

    assert strided_upload_return_seconds < queued_matmul_seconds * 0.5
    assert destination._device in _PENDING_H2D
    del source
    for _ in range(8):
        torch.empty_like(expected).fill_(-1)

    torch_mojo_device_module.synchronize(mojo_device)
    assert destination._device not in _PENDING_H2D
    torch.testing.assert_close(destination.cpu(), expected)
    assert all(result.shape == (4096, 4096) for result in delayed)


def test_non_blocking_mojo_to_cpu_does_not_drain_prior_gpu_work(mojo_device):
    """Async D2H returns pinned host storage without draining queued kernels."""
    max_device = find_equivalent_max_device(torch.device(mojo_device))
    if max_device.label != "gpu":
        pytest.skip("requires a MAX GPU")

    pending_d2h = getattr(mojo_tensor_module, "_PENDING_D2H", None)
    assert pending_d2h is not None, "D2H has no asynchronous lifetime tracking"

    a = torch.randn(4096, 4096).to(mojo_device)
    b = torch.randn(4096, 4096).to(mojo_device)
    expected = torch.arange(1 << 20, dtype=torch.float32)
    source = expected.to(mojo_device)

    # Warm the pinned-host allocation and DLPack adoption paths before timing.
    _ = source.to("cpu", non_blocking=True)
    torch_mojo_device_module.synchronize(mojo_device)

    queue_repeats = 8
    started = time.perf_counter()
    for _ in range(queue_repeats):
        _ = a @ b
    torch_mojo_device_module.synchronize(mojo_device)
    queued_matmul_seconds = time.perf_counter() - started

    delayed = [a @ b for _ in range(queue_repeats)]
    started = time.perf_counter()
    downloaded = source.to("cpu", non_blocking=True)
    download_return_seconds = time.perf_counter() - started

    assert download_return_seconds < queued_matmul_seconds * 0.5
    assert max_device in pending_d2h

    torch_mojo_device_module.synchronize(mojo_device)
    assert max_device not in pending_d2h
    torch.testing.assert_close(downloaded, expected)
    assert all(result.shape == (4096, 4096) for result in delayed)


def test_non_blocking_strided_d2h_survives_source_destruction(mojo_device):
    """The materialized source and pinned D2H owner outlive an async transfer."""
    max_device = find_equivalent_max_device(torch.device(mojo_device))
    if max_device.label != "gpu":
        pytest.skip("requires a MAX GPU")

    expected = torch.arange(1 << 20, dtype=torch.int32)
    storage = torch.stack((expected, -expected), dim=1).to(mojo_device)
    source = storage[:, 0]
    downloaded = source.to("cpu", non_blocking=True)
    retained = mojo_tensor_module._PENDING_D2H[max_device][-1][1]
    assert isinstance(retained, tuple) and len(retained) == 2

    del source, storage
    torch_mojo_device_module.synchronize(mojo_device)
    torch.testing.assert_close(downloaded, expected)
    assert max_device not in mojo_tensor_module._PENDING_D2H


def test_non_blocking_d2h_survives_destination_destruction(mojo_device):
    """Dropping the CPU alias early cannot release its in-flight HostBuffer."""
    max_device = find_equivalent_max_device(torch.device(mojo_device))
    if max_device.label != "gpu":
        pytest.skip("requires a MAX GPU")

    a = torch.randn(4096, 4096).to(mojo_device)
    b = torch.randn(4096, 4096).to(mojo_device)
    source = torch.arange(1 << 20, dtype=torch.float32).to(mojo_device)
    torch_mojo_device_module.synchronize(mojo_device)

    delayed = [a @ b for _ in range(8)]
    exports_before = len(dlpack._live_exports)
    downloaded = source.to("cpu", non_blocking=True)
    pending = mojo_tensor_module._PENDING_D2H[max_device]
    assert not pending[-1][0].is_ready()
    assert len(dlpack._live_exports) == exports_before + 1

    destination_ref = weakref.ref(downloaded)
    del downloaded
    gc.collect()
    assert destination_ref() is None
    assert len(dlpack._live_exports) == exports_before
    assert max_device in mojo_tensor_module._PENDING_D2H

    torch_mojo_device_module.synchronize(mojo_device)
    assert max_device not in mojo_tensor_module._PENDING_D2H
    assert all(result.shape == (4096, 4096) for result in delayed)


def test_non_blocking_d2h_adoption_failure_synchronizes(mojo_device, monkeypatch):
    """A DLPack error cannot release pinned/source owners while DMA is live."""
    max_device = find_equivalent_max_device(torch.device(mojo_device))
    if max_device.label != "gpu":
        pytest.skip("requires a MAX GPU")

    source = torch.arange(1 << 20, dtype=torch.float32).to(mojo_device)

    def fail_adoption(*_args, **_kwargs):
        raise RuntimeError("injected DLPack adoption failure")

    monkeypatch.setattr(dlpack, "make_capsule", fail_adoption)
    with pytest.raises(RuntimeError, match="injected DLPack adoption failure"):
        source.to("cpu", non_blocking=True)

    assert max_device not in mojo_tensor_module._PENDING_D2H
    assert max_device not in _PENDING_H2D


@pytest.mark.parametrize("direction", ["h2d", "d2h"])
def test_transfer_query_failure_retains_new_dma_owner(direction):
    """An older event-query error cannot drop a newly enqueued DMA owner."""

    class QueryErrorEvent:
        def is_ready(self):
            raise RuntimeError("injected event query failure")

    class CurrentEvent:
        def is_ready(self):
            return False

    class FakeStream:
        def __init__(self):
            self.current = CurrentEvent()

        def record_event(self):
            return self.current

    class FakeDevice:
        def __init__(self):
            self.default_stream = FakeStream()

    fake_device = FakeDevice()
    # _PENDING_H2D/_PENDING_D2H are keyed structurally (only default_stream is
    # read); cast the host-contract stand-in to the declared key type.
    device = cast(max.driver.Device, fake_device)
    old_owner = object()
    new_owner = object()
    if direction == "h2d":
        pending = mojo_tensor_module._PENDING_H2D
        lock = mojo_tensor_module._PENDING_H2D_LOCK

        def record():
            mojo_tensor_module._record_h2d_source(device, new_owner, non_blocking=True)

    else:
        pending = mojo_tensor_module._PENDING_D2H
        lock = mojo_tensor_module._PENDING_D2H_LOCK

        def record():
            mojo_tensor_module._record_d2h_owner(device, new_owner)

    with lock:
        pending[device] = mojo_tensor_module.deque([(QueryErrorEvent(), old_owner)])
    try:
        with pytest.raises(RuntimeError, match="injected event query failure"):
            record()
        assert list(pending[device]) == [
            (pending[device][0][0], old_owner),
            (fake_device.default_stream.current, new_owner),
        ]
    finally:
        with lock:
            pending.pop(device, None)


@pytest.mark.parametrize("direction", ["h2d", "d2h"])
def test_transfer_record_and_sync_failure_retains_owner(direction):
    """A faulted stream cannot release an owner whose DMA state is unknown."""

    class Owner:
        pass

    class FailingStream:
        def record_event(self):
            raise RuntimeError("injected event record failure")

        def synchronize(self):
            raise RuntimeError("injected recovery sync failure")

    class FakeDevice:
        def __init__(self):
            self.default_stream = FailingStream()

    device = cast(max.driver.Device, FakeDevice())
    owner = Owner()
    owner_ref = weakref.ref(owner)
    try:
        with pytest.raises(RuntimeError, match="injected recovery sync failure"):
            if direction == "h2d":
                mojo_tensor_module._record_h2d_source(device, owner, non_blocking=True)
            else:
                mojo_tensor_module._record_d2h_owner(device, owner)
        del owner
        gc.collect()
        assert owner_ref() is not None
        retained = mojo_tensor_module._FAILED_TRANSFER_OWNERS[device]
        assert len(retained) == 1 and retained[0][1] is owner_ref()
    finally:
        with mojo_tensor_module._FAILED_TRANSFER_OWNERS_LOCK:
            mojo_tensor_module._FAILED_TRANSFER_OWNERS.pop(device, None)


def test_same_device_d2d_does_not_drain_prior_gpu_work(mojo_device):
    """Contiguous and strided D2D copies stay queued on the device stream."""
    max_device = find_equivalent_max_device(torch.device(mojo_device))
    if max_device.label != "gpu":
        pytest.skip("requires a MAX GPU")

    a = torch.randn(4096, 4096).to(mojo_device)
    b = torch.randn(4096, 4096).to(mojo_device)
    elements = 1 << 20
    expected = torch.arange(elements, dtype=torch.float32)

    contiguous_source = expected.to(mojo_device)
    contiguous_destination = torch.empty_like(contiguous_source)

    strided_source_storage = torch.stack((expected, -expected), dim=1).to(mojo_device)
    strided_source = strided_source_storage[:, 0]
    strided_destination_storage = torch.empty_like(strided_source_storage)
    strided_destination = strided_destination_storage[:, 1]

    # Warm every copy path before measuring Python return latency.
    contiguous_destination.copy_(contiguous_source)
    strided_destination.copy_(strided_source)
    _ = a @ b
    torch_mojo_device_module.synchronize(mojo_device)

    started = time.perf_counter()
    _ = a @ b
    torch_mojo_device_module.synchronize(mojo_device)
    matmul_seconds = time.perf_counter() - started

    for layout, destination, source in (
        ("contiguous", contiguous_destination, contiguous_source),
        ("strided", strided_destination, strided_source),
    ):
        delayed = a @ b
        started = time.perf_counter()
        destination.copy_(source)
        copy_return_seconds = time.perf_counter() - started

        assert copy_return_seconds < matmul_seconds * 0.5, layout
        torch_mojo_device_module.synchronize(mojo_device)
        torch.testing.assert_close(destination.cpu(), expected)
        assert delayed.shape == (4096, 4096)


def test_mojo_rng_state_exact_replay_and_high_bit_seed(mojo_device):
    """Public RNG snapshots exactly replay per-device Philox reservations."""
    device = torch.device(mojo_device)
    seed = (1 << 63) + 0x12345
    torch_mojo_device_module.manual_seed_all(seed)

    initial = torch_mojo_device_module.get_rng_state(device)
    assert initial.dtype == torch.uint8
    assert initial.shape == (16,)

    assert _reserve_philox_state(device, 17) == (seed, 0)
    advanced = torch_mojo_device_module.get_rng_state(device)
    assert not torch.equal(advanced, initial)

    torch_mojo_device_module.set_rng_state(initial, device)
    assert _reserve_philox_state(device, 17) == (seed, 0)
    torch.testing.assert_close(torch_mojo_device_module.get_rng_state(device), advanced)


def test_mojo_rng_state_is_per_device():
    """Consuming one Mojo device's counter does not advance another device."""
    if torch_mojo_device_module.device_count() < 2:
        pytest.skip("requires two Mojo devices, including the MAX CPU device")

    first = torch.device("mojo:0")
    second = torch.device("mojo:1")
    torch_mojo_device_module.manual_seed_all(20260718)
    second_before = torch_mojo_device_module.get_rng_state(second)

    assert _reserve_philox_state(first, 257) == (20260718, 0)
    torch.testing.assert_close(
        torch_mojo_device_module.get_rng_state(second), second_before
    )
    assert _reserve_philox_state(second, 1) == (20260718, 0)


def test_mojo_rng_state_rejects_malformed_state(mojo_device):
    device = torch.device(mojo_device)
    with pytest.raises(ValueError, match="16-element uint8"):
        torch_mojo_device_module.set_rng_state(
            torch.zeros(16, dtype=torch.int64), device
        )
    with pytest.raises(ValueError, match="16-element uint8"):
        torch_mojo_device_module.set_rng_state(
            torch.zeros(15, dtype=torch.uint8), device
        )


def test_mojo_rng_seed_bounds_do_not_mutate_state(mojo_device):
    device = torch.device(mojo_device)
    torch_mojo_device_module.manual_seed_all(20260718)
    before = torch_mojo_device_module.get_rng_state(device)

    for invalid_seed in (1 << 64, -(1 << 63) - 1):
        with pytest.raises(ValueError, match="Overflow"):
            torch_mojo_device_module.manual_seed_all(invalid_seed)
        torch.testing.assert_close(
            torch_mojo_device_module.get_rng_state(device), before
        )

        with pytest.raises(ValueError, match="Overflow"):
            torch.manual_seed(invalid_seed)
        torch.testing.assert_close(
            torch_mojo_device_module.get_rng_state(device), before
        )
        # A CUDA-enabled PyTorch queues manual_seed_all() until its first CUDA
        # initialization.  The custom-device validation above then raises, so
        # replace that deferred invalid callback before another test initializes
        # CUDA.  This is a test-isolation concern; the valid seed preserves the
        # Mojo state asserted above.
        torch.manual_seed(20260718)


def test_mojo_rng_state_accepts_reshaped_byte_tensor(mojo_device):
    device = torch.device(mojo_device)
    torch_mojo_device_module.manual_seed_all((1 << 63) + 20260718)
    initial = torch_mojo_device_module.get_rng_state(device)
    _reserve_philox_state(device, 37)

    torch_mojo_device_module.set_rng_state(initial.reshape(4, 4), device)
    assert _reserve_philox_state(device, 37) == ((1 << 63) + 20260718, 0)


def test_mojo_rng_reservations_are_atomic_and_reject_wrap(mojo_device):
    device = torch.device(mojo_device)
    torch_mojo_device_module.manual_seed_all(20260718)

    reservations = 512
    with ThreadPoolExecutor(max_workers=16) as pool:
        bases = list(
            pool.map(lambda _: _reserve_philox_state(device, 1)[1], range(reservations))
        )
    assert sorted(bases) == list(range(reservations))

    seed = (1 << 63) + 7
    counter = (1 << 64) - 1
    encoded = seed.to_bytes(8, "little") + counter.to_bytes(8, "little")
    torch_mojo_device_module.set_rng_state(
        torch.tensor(list(encoded), dtype=torch.uint8), device
    )
    before = torch_mojo_device_module.get_rng_state(device)

    with pytest.raises(OverflowError, match="would wrap"):
        _reserve_philox_state(device, 1)
    torch.testing.assert_close(torch_mojo_device_module.get_rng_state(device), before)


def test_torch_fork_rng_restores_mojo_counter(mojo_device):
    device = torch.device(mojo_device)
    index = (
        torch_mojo_device_module.current_device()
        if device.index is None
        else device.index
    )
    torch_mojo_device_module.manual_seed_all(91)
    before = torch_mojo_device_module.get_rng_state(device)

    with torch.random.fork_rng(devices=[index], device_type="mojo"):
        assert _reserve_philox_state(device, 33) == (91, 0)
        assert not torch.equal(torch_mojo_device_module.get_rng_state(device), before)

    torch.testing.assert_close(torch_mojo_device_module.get_rng_state(device), before)


def test_dtype_preservation(mojo_device):
    """Test that dtypes are preserved during conversion"""
    for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
        original = torch.tensor([1, 2, 3], dtype=dtype)
        mojo_tensor = original.to(mojo_device)
        result = mojo_tensor.to("cpu")

        assert result.dtype == dtype
        torch.testing.assert_close(result, original)


def test_multiple_conversions():
    """Test multiple to() calls don't cause issues"""
    tensor = torch.tensor([1.0, 2.0])

    # Multiple conversions should work
    max1 = tensor.to("mojo")
    max2 = max1.to("mojo")  # Should return same tensor
    cpu1 = max2.to("cpu")
    cpu2 = cpu1.to("cpu")  # Should work normally

    # Test operations step by step for clearer errors
    diff = max1 - max2
    squared = diff**2
    summed = torch.sum(squared)
    cpu_result = summed.to("cpu")
    result_value = cpu_result.item()
    assert result_value == 0

    torch.testing.assert_close(cpu2, tensor)


def test_module_to_mojo_preserves_tied_parameters(mojo_device):
    """Module conversion must not duplicate aliased/tied parameter storage."""

    class TiedWeights(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 8)
            self.projection = torch.nn.Linear(8, 16, bias=False)
            self.projection.weight = self.embedding.weight

    module = TiedWeights()
    assert module.embedding.weight is module.projection.weight

    module.to(mojo_device)

    assert module.embedding.weight is module.projection.weight
    embedding_weight = module.embedding.weight
    projection_weight = module.projection.weight
    assert isinstance(embedding_weight, TorchMojoTensor)
    assert isinstance(projection_weight, TorchMojoTensor)
    assert embedding_weight._holder is projection_weight._holder
    assert embedding_weight._ptr == projection_weight._ptr
    assert len(list(module.parameters())) == 1


def test_mojo_parameters_enable_foreach_optimizer_selection(mojo_device):
    parameter = torch.nn.Parameter(torch.ones(8)).to(mojo_device)
    fused, foreach = _default_to_fused_or_foreach([parameter], differentiable=False)

    assert not fused
    assert foreach


@pytest.mark.parametrize("foreach", [None, True, False])
def test_mojo_adamw_step_matches_cpu(mojo_gpu_available, foreach):
    """The optimizer path used by nanoGPT must update parameters and moments."""
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    mojo_gpu = "mojo:0"
    initial = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32)
    cpu_parameter = torch.nn.Parameter(initial.clone())
    mojo_parameter = torch.nn.Parameter(initial.to(mojo_gpu))
    cpu_optimizer = torch.optim.AdamW(
        [cpu_parameter],
        lr=0.025,
        betas=(0.8, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        foreach=foreach,
    )
    mojo_optimizer = torch.optim.AdamW(
        [mojo_parameter],
        lr=0.025,
        betas=(0.8, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        foreach=foreach,
    )

    for grad in (
        torch.tensor([0.25, -0.5, 0.75, -1.0]),
        torch.tensor([-0.125, 0.25, -0.375, 0.5]),
    ):
        cpu_parameter.grad = grad.clone()
        mojo_parameter.grad = grad.to(mojo_gpu)
        cpu_optimizer.step()
        mojo_optimizer.step()

    torch_mojo_device_module.synchronize(mojo_gpu)
    torch.testing.assert_close(mojo_parameter.cpu(), cpu_parameter)
    cpu_state = cpu_optimizer.state[cpu_parameter]
    mojo_state = mojo_optimizer.state[mojo_parameter]
    for name in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(mojo_state[name].cpu(), cpu_state[name])
    assert mojo_state["step"].item() == cpu_state["step"].item() == 2


def test_mojo_checkpoint_resumes_through_portable_cpu_state(mojo_gpu_available):
    """The nanoGPT resume path loads CPU state, then moves it normally."""
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    mojo_gpu = "mojo:0"
    model = torch.nn.Linear(3, 2).to(mojo_gpu)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=None)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    checkpoint_bytes = io.BytesIO()
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        checkpoint_bytes,
    )
    checkpoint_bytes.seek(0)
    checkpoint = torch.load(checkpoint_bytes, map_location="cpu")
    assert all(tensor.device.type == "cpu" for tensor in checkpoint["model"].values())

    resumed_model = torch.nn.Linear(3, 2)
    resumed_model.load_state_dict(checkpoint["model"])
    resumed_model.to(mojo_gpu)
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(), lr=0.01, foreach=None
    )
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    for state in resumed_optimizer.state.values():
        assert state["step"].device.type == "cpu"
        assert state["exp_avg"].device == torch.device(mojo_gpu)
        assert state["exp_avg_sq"].device == torch.device(mojo_gpu)

    for parameter in resumed_model.parameters():
        parameter.grad = torch.ones_like(parameter)
    resumed_optimizer.step()
    torch_mojo_device_module.synchronize(mojo_gpu)
    assert all(
        torch.isfinite(parameter.cpu()).all()
        for parameter in resumed_model.parameters()
    )


@pytest.mark.parametrize("foreach", [None, True, False])
def test_mojo_clip_grad_norm_matches_cpu(mojo_gpu_available, foreach):
    """nanoGPT's FP32 gradient clipping uses the foreach L2-norm path."""
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    mojo_gpu = "mojo:0"
    cpu_parameters = [
        torch.nn.Parameter(torch.zeros(3)),
        torch.nn.Parameter(torch.zeros(2, 2)),
    ]
    mojo_parameters = [
        torch.nn.Parameter(parameter.detach().to(mojo_gpu))
        for parameter in cpu_parameters
    ]
    gradients = (
        torch.tensor([3.0, 4.0, -2.0]),
        torch.tensor([[1.0, -2.0], [2.0, -1.0]]),
    )
    for cpu_parameter, mojo_parameter, gradient in zip(
        cpu_parameters, mojo_parameters, gradients, strict=True
    ):
        cpu_parameter.grad = gradient.clone()
        mojo_parameter.grad = gradient.to(mojo_gpu)

    expected_norm = torch.nn.utils.clip_grad_norm_(
        cpu_parameters, 1.25, foreach=foreach
    )
    actual_norm = torch.nn.utils.clip_grad_norm_(mojo_parameters, 1.25, foreach=foreach)
    torch_mojo_device_module.synchronize(mojo_gpu)

    torch.testing.assert_close(actual_norm.cpu(), expected_norm)
    for actual, expected in zip(mojo_parameters, cpu_parameters, strict=True):
        # Both grads were assigned above.
        assert actual.grad is not None
        torch.testing.assert_close(actual.grad.cpu(), expected.grad)


def test_metal_fast_add_gate_is_decided_once(monkeypatch):
    aten_fast._has_metal_accelerator.cache_clear()
    monkeypatch.setattr(
        utils,
        "get_accelerators",
        lambda: [SimpleNamespace(api="cuda"), SimpleNamespace(api="cpu")],
    )
    assert aten_fast._has_metal_accelerator() is False

    monkeypatch.setattr(
        utils,
        "get_accelerators",
        lambda: [SimpleNamespace(api="metal"), SimpleNamespace(api="cpu")],
    )
    assert aten_fast._has_metal_accelerator() is False  # cached from the first call
    aten_fast._has_metal_accelerator.cache_clear()
    assert aten_fast._has_metal_accelerator() is True
    aten_fast._has_metal_accelerator.cache_clear()


def test_device_ordering():
    """Test that device ordering follows GPU first, CPU last convention"""
    ordered_accelerators = get_ordered_accelerators()

    # Check that we have both GPU and CPU
    gpu_devices = [acc for acc in ordered_accelerators if acc.label == "gpu"]
    cpu_devices = [acc for acc in ordered_accelerators if acc.label == "cpu"]

    # Should have at least one device
    assert len(ordered_accelerators) > 0

    # If we have both GPU and CPU, GPU should come first
    if gpu_devices and cpu_devices:
        # First device should be GPU
        assert ordered_accelerators[0].label == "gpu"
        # Last device should be CPU
        assert ordered_accelerators[-1].label == "cpu"


def test_device_mapping_consistency():
    """Test that CPU maps to highest index and GPU to lower indices"""

    ordered_accelerators = get_ordered_accelerators()

    if len(ordered_accelerators) > 1:
        # Test CPU device mapping
        cpu_device = torch.device("cpu")
        max_cpu = find_equivalent_max_device(cpu_device)

        # CPU should map to a CPU accelerator
        assert max_cpu.label == "cpu"

        # Find CPU in ordered list - should be last if we have multiple devices
        cpu_indices = [
            i for i, acc in enumerate(ordered_accelerators) if acc.label == "cpu"
        ]
        if cpu_indices:
            # If CPU exists, it should be at the highest index
            assert cpu_indices[-1] == len(ordered_accelerators) - 1


def test_gpu_first_cpu_last_convention():
    """Test the specific convention: device 0 = first GPU, highest index = CPU"""

    ordered_accelerators = get_ordered_accelerators()

    # If we have both GPU and CPU
    gpu_count = sum(1 for acc in ordered_accelerators if acc.label == "gpu")
    cpu_count = sum(1 for acc in ordered_accelerators if acc.label == "cpu")

    if gpu_count > 0 and cpu_count > 0:
        # First device should be GPU
        assert ordered_accelerators[0].label == "gpu"

        # Last device should be CPU
        assert ordered_accelerators[-1].label == "cpu"

        # Test that mojo (index 0) goes to GPU
        t_gpu = torch.tensor([1.0]).to("mojo")
        assert isinstance(t_gpu, TorchMojoTensor)

        # Test that highest index goes to CPU
        cpu_index = len(ordered_accelerators) - 1
        t_cpu = torch.tensor([1.0]).to(f"mojo:{cpu_index}")
        assert isinstance(t_cpu, TorchMojoTensor)


# Original tests from the existing file
def function_equivalent_on_both_devices(
    func, device, *args, rtol=1e-4, atol=1e-4, **kwargs
):
    # This helper checks forward values only. Keeping the first forward's
    # autograd graph alive while the same closure-owned module moves back to
    # CPU adds a legitimate TensorImpl reference, which PyTorch's required
    # swap-on-conversion path rejects. Avoid manufacturing that unrelated
    # lifetime condition in forward-equivalence tests.
    with torch.no_grad():
        out1 = func(*args, device=device, **kwargs)
        out2 = func(*args, device="cpu", **kwargs)
    if isinstance(out1, list | tuple):
        assert type(out1) is type(out2)
    else:
        assert isinstance(out1, torch.Tensor)
        assert isinstance(out2, torch.Tensor)
        out1 = [out1]
        out2 = [out2]

    # We transfer on device 1
    out1 = [o.to("cpu") for o in out1]

    for i, (o1, o2) in enumerate(zip(out1, out2)):
        assert o1.device == o2.device, f"Issue with output {i}"
        assert o1.shape == o2.shape, f"Issue with output {i}"
        assert o1.dtype == o2.dtype, f"Issue with output {i}"
        assert torch.allclose(o1, o2, rtol=rtol, atol=atol), f"Issue with output {i}"


def test_mojo_device_basic(mojo_device):
    def do_sqrt(device):
        a = torch.arange(4, device=device, dtype=torch.float32)
        return torch.sqrt(a)

    function_equivalent_on_both_devices(do_sqrt, mojo_device)


def test_mojo_device_basic_arange_sqrt(mojo_device):
    a = torch.arange(4, device=mojo_device, dtype=torch.float32)

    sqrt_result = torch.sqrt(a)

    result_cpu = sqrt_result.to("cpu")
    assert torch.allclose(
        result_cpu, torch.tensor([0.0, 1.0, 1.4142, 1.7320]), atol=1e-4
    )

    b = torch.arange(4, device=mojo_device, dtype=torch.float32)
    chained = sqrt_result + b
    chained_cpu = chained.to("cpu")
    assert torch.allclose(
        chained_cpu, torch.tensor([0.0, 2.0, 3.4142, 4.7320]), atol=1e-4
    )


def test_device_creation(mojo_device):
    torch_device = torch.device(mojo_device)
    arr = torch.arange(4, device=torch_device, dtype=torch.float32)
    arr_cpu = arr.to("cpu")

    assert torch.allclose(arr_cpu, torch.tensor([0.0, 1.0, 2.0, 3.0]), atol=1e-4)


def test_device_basic_full(mojo_device):
    def do_full(device):
        a = torch.full((2, 3), 7.0, device=device, dtype=torch.float32)
        return a

    function_equivalent_on_both_devices(do_full, mojo_device)


def test_convolution_2d(mojo_device):
    input_tensor_cpu = torch.randn(1, 3, 32, 32, device="cpu")
    weight_cpu = torch.randn(6, 3, 5, 5, device="cpu")
    bias_cpu = torch.randn(6, device="cpu")

    def do_convolution(device):
        input_tensor = input_tensor_cpu.to(device)
        weight = weight_cpu.to(device)
        bias = bias_cpu.to(device)
        return torch.nn.functional.conv2d(
            input_tensor, weight, bias=bias, stride=1, padding=2
        )

    function_equivalent_on_both_devices(do_convolution, mojo_device)


def test_simple_module(mojo_device):
    linear = torch.nn.Linear(4, 8)

    def run_module(device):
        my_linear = linear.to(device)
        return my_linear.weight

    function_equivalent_on_both_devices(run_module, mojo_device)


def test_custom_module(mojo_device):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 8)

        def forward(self, x):
            return self.linear(x)

    module = MyModule()
    input_tensor = torch.randn(2, 4)

    def run_module(device):
        in_device_module = module.to(device)
        in_device_input_tensor = input_tensor.to(device)
        return in_device_module(in_device_input_tensor)

    function_equivalent_on_both_devices(run_module, mojo_device, rtol=1e-3, atol=1e-3)


def test_custom_module_with_seqential(mojo_device):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 8)

        def forward(self, x):
            return self.linear(x)

    module = torch.nn.Sequential(MyModule())
    input_tensor = torch.randn(2, 4)

    def run_module(device):
        in_device_module = module.to(device)
        in_device_input_tensor = input_tensor.to(device)
        return in_device_module(in_device_input_tensor)

    function_equivalent_on_both_devices(run_module, mojo_device, rtol=1e-3, atol=1e-3)


def test_compile_with_max_device(mojo_device):
    @torch.compile(backend=mojo_backend)
    def do_sqrt(device):
        a = torch.arange(4, device=device, dtype=torch.float32)
        return torch.sqrt(a)

    function_equivalent_on_both_devices(do_sqrt, mojo_device)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a second backend to copy from"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_to_mojo_from_another_backend(mojo_gpu, dtype):
    """A tensor on another accelerator must transfer, not fault.

    `_to_copy` and `_copy_from` branch on "is it a TorchMojoTensor", which is
    not the same question as "is it on the host": with a GPU torch installed
    they also take a CUDA source, and both upload paths dereference
    `data_ptr()` as HOST memory.  A CUDA pointer therefore segfaulted the
    process -- no exception, no message.  Found by the OpInfo `to` samples,
    which include a cross-device target tensor.
    """
    src = torch.randn(3, 5, device="cuda", dtype=dtype)
    moved = src.to(mojo_gpu)
    assert moved.device.type == "mojo"
    assert moved.dtype == dtype
    assert torch.equal(moved.cpu(), src.cpu())

    cast = src.to(mojo_gpu, dtype=torch.float32)
    assert cast.dtype == torch.float32
    assert torch.allclose(cast.cpu(), src.float().cpu())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a second backend to copy from"
)
def test_copy_into_mojo_from_another_backend(mojo_gpu):
    """The same trap on the `copy_` path, including the broadcasting arm."""
    dest = torch.zeros(3, 5, device=mojo_gpu)
    src = torch.randn(3, 5, device="cuda")
    dest.copy_(src)
    assert torch.equal(dest.cpu(), src.cpu())

    wide = torch.zeros(4, 3, device=mojo_gpu)
    row = torch.arange(3, device="cuda", dtype=torch.float32)
    wide.copy_(row)
    assert torch.equal(wide.cpu(), row.cpu().expand(4, 3).contiguous())


def test_from_cpu_rejects_a_device_source(mojo_gpu):
    """The guard behind both call sites: a message, never a fault."""
    on_device = torch.zeros(4, device=mojo_gpu)
    assert isinstance(on_device, TorchMojoTensor)
    with pytest.raises(RuntimeError, match="requires a CPU source"):
        TorchMojoTensor._from_cpu(on_device, on_device._device)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a second backend to copy from"
)
def test_same_gpu_transfer_skips_the_host(mojo_gpu):
    """The transfer must not go through host memory when both live on one GPU.

    Asserted by timing rather than by values, because a host bounce is
    CORRECT -- just two PCIe crossings instead of one HBM copy.  537 MB
    measured 375 ms bounced against 0.64 ms on device, so an order of
    magnitude is a wide margin around that.
    """
    big = torch.randn(1 << 26, device="cuda")  # 256 MB
    big.to(mojo_gpu)
    torch_mojo_device_module.synchronize()

    start = time.perf_counter()
    moved = big.to(mojo_gpu)
    torch_mojo_device_module.synchronize()
    on_device = time.perf_counter() - start

    start = time.perf_counter()
    bounced = big.cpu().to(mojo_gpu)
    torch_mojo_device_module.synchronize()
    through_host = time.perf_counter() - start

    assert torch.equal(moved.cpu(), bounced.cpu())
    assert on_device * 10 < through_host, (
        f"expected the on-device route; {on_device * 1e3:.2f}ms on device vs "
        f"{through_host * 1e3:.2f}ms through the host"
    )


def test_pointer_ordinal_identifies_the_owning_gpu(mojo_gpu):
    """cuda_peer reads device identity off the POINTER, not off an ordinal.

    MAX's Device exposes no UUID or PCI id, so matching its `gpu:0` to torch's
    `cuda:0` by ordinal would assume both runtimes enumerate alike.  Asking the
    driver who owns each allocation is a fact rather than an assumption.
    """
    on_mojo = torch.zeros(8, device=mojo_gpu)
    assert isinstance(on_mojo, TorchMojoTensor)
    if on_mojo._device.api != "cuda":
        # tests/test_distributed.py covers hip_peer, the AMD counterpart.
        pytest.skip("cuda_peer speaks the CUDA driver API only")
    torch_mojo_device_module.synchronize()
    assert cuda_peer.device_ordinal(on_mojo._ptr) is not None
    assert cuda_peer.device_ordinal(0) is None
    assert cuda_peer.device_ordinal(torch.zeros(8).data_ptr()) is None
    assert not cuda_peer.same_physical_device(on_mojo._ptr, 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vector_norm_with_an_accumulation_dtype(mojo_gpu_available, dtype):
    """FSDP1's clip_grad_norm_ asks for the norm in float32 explicitly."""
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    cpu = torch.randn(4096, dtype=dtype)
    expected = torch.linalg.vector_norm(cpu, 2.0, dtype=torch.float32)
    got = torch.linalg.vector_norm(cpu.to("mojo:0"), 2.0, dtype=torch.float32)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-5, atol=1e-4)


def test_library_attributes_do_not_clobber_the_payload(mojo_gpu_available):
    """A mojo Parameter IS the wrapper object, so torch's own bookkeeping
    attributes land in the payload's namespace.
    ``FlatParameter._init_metadata`` writes ``_strides``/``_shapes``/
    ``_numels``; none of those may disturb the layout metadata."""
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    tensor = torch.arange(8, dtype=torch.float32, device="mojo:0")
    assert isinstance(tensor, TorchMojoTensor)
    strides = tensor._mojo_strides
    # Simulates FSDP1 writing its own bookkeeping onto the wrapper instance;
    # these names are deliberately not part of TorchMojoTensor's declared
    # payload, so setattr (not a plain attribute assignment) is the honest
    # way to write them.
    setattr(tensor, "_strides", ((1,), (1,)))
    setattr(tensor, "_shapes", (torch.Size([4]), torch.Size([4])))
    setattr(tensor, "_numels", (4, 4))
    assert tensor._mojo_strides == strides
    first, second = torch.split(tensor, [4, 4])
    assert second.cpu().tolist() == [4.0, 5.0, 6.0, 7.0]
    assert first.cpu().tolist() == [0.0, 1.0, 2.0, 3.0]


def test_data_assignment_moves_the_payload(mojo_gpu_available):
    """``x.data = y`` has to move the allocation, not just TensorImpl metadata.

    The C++ setter shallow-copies the TensorImpl and stops; for this
    storage-less wrapper that left every kernel reading the OLD buffer with
    no error at all. FSDP1 swaps a flat parameter between its sharded and
    unsharded buffers with exactly this assignment.
    """
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    destination = torch.zeros(8, device="mojo:0")
    source = torch.arange(8, dtype=torch.float32, device="mojo:0") + 100
    destination.data = source
    assert destination.cpu().tolist() == source.cpu().tolist()

    # A strided view at a non-zero offset must survive intact: FSDP hands
    # every original parameter a slice of the flat parameter this way.
    base = torch.arange(16, dtype=torch.float32, device="mojo:0")
    view = base[4:12].reshape(2, 4)
    holder = torch.zeros(1, device="mojo:0")
    holder.data = view
    assert tuple(holder.shape) == (2, 4)
    assert holder.stride() == view.stride()
    assert holder.cpu().tolist() == [[4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]


def test_as_strided_zero_copy_view(mojo_gpu_available):
    """A strided view at an offset, and the allocation-bounds refusal.

    as_strided is the one view op whose arguments are unconstrained by the
    input's own shape, so the backend must bound-check against the real
    allocation rather than trust the caller (DDP's Reducer builds gradient
    bucket views with it).
    """
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    cpu = torch.arange(24, dtype=torch.float32)
    dev = cpu.to("mojo:0")
    view = torch.as_strided(dev, (3, 4), (8, 2), 1)
    assert torch.equal(view.cpu(), torch.as_strided(cpu, (3, 4), (8, 2), 1))
    # Zero-copy: writes through the base are visible in the view.
    dev.add_(1.0)
    assert torch.equal(view.cpu(), torch.as_strided(cpu + 1, (3, 4), (8, 2), 1))
    # A layout that would reach past the allocation must be refused.
    with pytest.raises(NotImplementedError):
        torch.as_strided(dev, (5, 4), (8, 2), 1)


def test_set_source_tensor_adopts_the_allocation(mojo_gpu_available):
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    destination = torch.zeros(8, device="mojo:0")
    source = torch.arange(4, dtype=torch.float32, device="mojo:0") + 50
    returned = destination.set_(source)  # ty: ignore[invalid-argument-type] -- torch's stub lacks the Tensor overload
    assert returned is destination
    assert tuple(destination.shape) == (4,)
    assert destination.cpu().tolist() == [50.0, 51.0, 52.0, 53.0]
    # Sharing the allocation, not a copy of it.
    source.add_(1.0)
    assert destination.cpu().tolist() == [51.0, 52.0, 53.0, 54.0]
