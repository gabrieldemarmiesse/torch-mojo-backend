import threading

import torch

from ..torch_compile_backend.utils import get_accelerators

_current_device = 0
_UINT64_MASK = (1 << 64) - 1
_DEFAULT_RNG_SEED = 67_280_421_310_721
_rng_default_seed = _DEFAULT_RNG_SEED
_rng_states: dict[int, tuple[int, int]] = {}
_rng_lock = threading.Lock()


def cpu():
    return torch.device(f"mojo:{len(list(get_accelerators())) - 1}")


def _is_in_bad_fork():
    return False


def _normalize_rng_seed(seed) -> int:
    value = int(seed)
    if value < -(1 << 63) or value > _UINT64_MASK:
        raise ValueError("Overflow when unpacking long long")
    return value & _UINT64_MASK


def _rng_device_index(device=None) -> int:
    if device is None:
        index = _current_device
    elif isinstance(device, int):
        index = device
    else:
        torch_device = torch.device(device)
        if torch_device.type != "mojo":
            raise ValueError(f"expected a mojo RNG device, got {torch_device}")
        index = _current_device if torch_device.index is None else torch_device.index
    if index < 0 or index >= device_count():
        raise ValueError(f"Invalid device index {index}")
    return index


def manual_seed_all(seed):
    """Reset every Mojo device to the same Philox seed and counter zero."""
    global _rng_default_seed
    normalized = _normalize_rng_seed(seed)
    with _rng_lock:
        _rng_default_seed = normalized
        _rng_states.clear()
        for index in range(device_count()):
            _rng_states[index] = (normalized, 0)


def device_count():
    return len(list(get_accelerators()))


def get_rng_state(device=None):
    """Return the selected device's exact ``(seed, counter)`` state."""
    index = _rng_device_index(device)
    with _rng_lock:
        seed, counter = _rng_states.setdefault(index, (_rng_default_seed, 0))
    encoded = seed.to_bytes(8, "little") + counter.to_bytes(8, "little")
    return torch.tensor(list(encoded), dtype=torch.uint8)


def set_rng_state(new_state, device=None):
    """Restore an exact state produced by :func:`get_rng_state`."""
    if not isinstance(new_state, torch.Tensor):
        raise TypeError("Mojo RNG state must be a torch.Tensor")
    state = new_state.detach().cpu().contiguous()
    if state.dtype != torch.uint8 or state.numel() != 16:
        raise ValueError("Mojo RNG state must be a 16-element uint8 tensor")
    encoded = bytes(state.reshape(-1).tolist())
    seed = int.from_bytes(encoded[:8], "little")
    counter = int.from_bytes(encoded[8:], "little")
    index = _rng_device_index(device)
    with _rng_lock:
        _rng_states[index] = (seed, counter)


def _reserve_philox_state(device, counter_increment: int) -> tuple[int, int]:
    """Atomically reserve a per-device Philox counter interval.

    The caller passes the returned seed/base counter to an asynchronous device
    kernel. Reserving host state never inspects a tensor or synchronizes a
    device queue.
    """
    if type(counter_increment) is not int or counter_increment < 0:
        raise ValueError("Philox counter increment must be a nonnegative integer")
    index = _rng_device_index(device)
    with _rng_lock:
        seed, counter = _rng_states.setdefault(index, (_rng_default_seed, 0))
        if counter_increment > _UINT64_MASK - counter:
            raise OverflowError("Philox counter reservation would wrap uint64")
        _rng_states[index] = (seed, counter + counter_increment)
    return seed, counter


def is_available():
    # Always true as there is at least the CPU
    return True


def is_initialized():
    return True


def current_device():
    return _current_device


def set_device(device_idx: int):
    global _current_device
    if device_idx < 0 or device_idx >= device_count():
        raise ValueError(f"Invalid device index {device_idx}")
    _current_device = device_idx


def synchronize(device=None):
    """Wait for work and release completed asynchronous transfer owners."""
    from . import deferred_compile
    from .torch_mojo_tensor import (
        _release_synchronized_d2h_owners,
        _release_synchronized_h2d_sources,
        find_equivalent_max_device,
    )

    deferred_compile.drain()

    if device is None:
        torch_device = torch.device(f"mojo:{_current_device}")
    elif isinstance(device, int):
        torch_device = torch.device(f"mojo:{device}")
    else:
        torch_device = torch.device(device)
        if torch_device.type == "mojo" and torch_device.index is None:
            torch_device = torch.device(f"mojo:{_current_device}")

    max_device = find_equivalent_max_device(torch_device)
    max_device.default_stream.synchronize()
    _release_synchronized_h2d_sources(max_device)
    _release_synchronized_d2h_owners(max_device)


def get_amp_supported_dtype():
    return [torch.float16, torch.bfloat16]  # TODO change
