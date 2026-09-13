"""PrivateUse1 device-module identity and RNG-state runtime contracts.

Transfer-lifetime and general functional coverage moved to
`tests/test_mojo_device.py` (this file used to duplicate a chunk of it);
`test_mojo_is_the_default_torch_accelerator` and the accelerator-synchronize
test live there too, kept once. What is unique to this file: the indexless
`"mojo"` device resolving through the current device, and the RNG-state
contract (`torch_mojo_backend/native/device_module.py`'s `get_rng_state` /
`set_rng_state` / `manual_seed` / `manual_seed_all`, backed by the C++
shim's Philox generator -- see `native/csrc/shim_runtime.cpp`).

The old `torch_mojo_device_module`/`torch_mojo_tensor` internals this file
used to import (`_reserve_philox_state`, `_PENDING_H2D`, `_record_h2d_source`
/`_record_d2h_owner`, `_FAILED_TRANSFER_OWNERS`) do not exist anymore: the
Philox counter reservation and the DMA-owner-lifetime bookkeeping for
non-blocking transfers both live in the C++ shim now, with no Python-level
state to unit test directly. What survives is the *public* RNG-state
contract (get/set round trips, per-device isolation, `fork_rng`), tested
here through `device_module` only.
"""

import pytest
import torch

from torch_mojo_backend import register_mojo_devices
from torch_mojo_backend.native import device_module

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(autouse=True)
def setup_mojo_device():
    register_mojo_devices()


def test_indexless_mojo_device_uses_current_device():
    if device_module.device_count() < 2:
        pytest.skip("requires two Mojo devices, including the MAX CPU device")
    original_index = device_module.current_device()
    alternate_index = (original_index + 1) % device_module.device_count()
    try:
        device_module.set_device(alternate_index)
        empty_tensor = torch.empty(1, device="mojo")
        assert empty_tensor.device.index == alternate_index
    finally:
        device_module.set_device(original_index)
    assert device_module.current_device() == original_index


def test_mojo_rng_state_round_trips_a_high_bit_seed(mojo_device):
    device = torch.device(mojo_device)
    seed = (1 << 63) + 0x12345
    device_module.manual_seed_all(seed)

    state = device_module.get_rng_state(device)
    assert state.dtype == torch.uint8
    assert state.shape == (16,)
    assert int.from_bytes(bytes(state.tolist()[:8]), "little") == seed

    device_module.set_rng_state(state, device)
    torch.testing.assert_close(device_module.get_rng_state(device), state)


def test_mojo_rng_state_is_per_device():
    """Seeding one Mojo device's generator does not perturb another's."""
    if device_module.device_count() < 2:
        pytest.skip("requires two Mojo devices, including the MAX CPU device")
    first = torch.device("mojo:0")
    second = torch.device("mojo", device_module.device_count() - 1)
    device_module.manual_seed_all(20260718)
    second_before = device_module.get_rng_state(second)

    with device_module.device(first):
        device_module.manual_seed(314159)
    torch.testing.assert_close(device_module.get_rng_state(second), second_before)


def test_mojo_rng_state_rejects_malformed_state(mojo_device):
    device = torch.device(mojo_device)
    with pytest.raises(ValueError, match="16-element uint8"):
        device_module.set_rng_state(torch.zeros(16, dtype=torch.int64), device)
    with pytest.raises(ValueError, match="16-element uint8"):
        device_module.set_rng_state(torch.zeros(15, dtype=torch.uint8), device)


def test_mojo_rng_seed_is_masked_to_64_bits_rather_than_rejected(mojo_device):
    """`manual_seed_all`/`manual_seed` mask the seed to 64 bits (matching the
    16-byte uint8 state's own capacity) instead of validating bounds and
    raising -- unlike the old Python RNG, which raised `ValueError` on an
    out-of-range seed. Document the new (simpler, never-user-facing-error)
    contract here rather than asserting an error that no longer happens.
    """
    device = torch.device(mojo_device)
    mask = (1 << 64) - 1
    for seed in (1 << 64, -(1 << 63) - 1):
        device_module.manual_seed_all(seed)
        state = device_module.get_rng_state(device)
        stored = int.from_bytes(bytes(state.tolist()[:8]), "little")
        assert stored == seed & mask


def test_torch_fork_rng_restores_mojo_state():
    device = torch.device("mojo:0")
    device_module.manual_seed_all(91)
    before = device_module.get_rng_state(device)

    with torch.random.fork_rng(devices=[0], device_type="mojo"):
        device_module.manual_seed_all(12345)
        assert not torch.equal(device_module.get_rng_state(device), before)

    torch.testing.assert_close(device_module.get_rng_state(device), before)
