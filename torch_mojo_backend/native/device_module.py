"""`torch.mojo`: the device module torch looks up by backend name.

Devices, streams and events are torch's own generic ones (the C++ shim
implements the device guard), so this module is the thin torch.cuda-shaped
surface: device selection, RNG state, synchronize, amp dtypes, memory info.
"""

from __future__ import annotations

import ctypes

import torch

from torch_mojo_backend import native

_PRIVATEUSE1 = 20  # c10::DeviceType::PrivateUse1

DeviceLike = "int | str | torch.device | None"


def _index(device: int | str | torch.device | None) -> int:
    if device is None:
        return current_device()
    if isinstance(device, int):
        return device
    d = torch.device(device)
    if d.type != "mojo":
        raise ValueError(f"expected a mojo device, got {d}")
    return current_device() if d.index is None else d.index


def is_available() -> bool:
    return native.is_registered() and native.device_count() > 0


def is_initialized() -> bool:
    return native.is_registered()


def _lazy_init():
    """torch's `device_lazy_init(PrivateUse1)` hook: it imports `torch.mojo`
    and calls this, then marks the device initialized.

    Nothing to do here (registration already built everything), but the method
    must EXIST: without it torch never sets its per-device-type initialized
    flag, and `torch._C._accelerator_synchronizeDevice` -- which returns early
    for an uninitialized lazy-init device -- silently does nothing. That made
    `torch.accelerator.synchronize()` and `torch.mojo.synchronize()` no-ops,
    so a readback after work on a non-default stream could race it.
    """


def _is_in_bad_fork() -> bool:
    return False


def device_count() -> int:
    return native.device_count()


def cpu() -> torch.device:
    """The MAX CPU device is the last mojo index."""
    return torch.device(f"mojo:{device_count() - 1}")


def current_device() -> int:
    return int(native.shim().tmb_current_device())


def set_device(device: int | str | torch.device):
    idx = _index(device)
    if idx < 0 or idx >= device_count():
        raise ValueError(f"Invalid device index {idx}")
    native.shim().tmb_set_current_device(ctypes.c_int32(idx))


class device:
    """Context manager swapping the current device (torch.serialization needs
    it on the backend module for map_location="mojo")."""

    def __init__(self, device: int | str | torch.device | None):
        self.idx = -1 if device is None else _index(device)
        self.prev = -1

    def __enter__(self):
        self.prev = current_device()
        if self.idx >= 0:
            set_device(self.idx)

    def __exit__(self, *exc: object) -> bool:
        if self.idx >= 0:
            set_device(self.prev)
        return False


def synchronize(device: int | str | torch.device | None = None):
    torch._C._accelerator_synchronizeDevice(_index(device))


def manual_seed(seed: int):
    native.shim().tmb_rng_manual_seed(
        ctypes.c_int32(current_device()), ctypes.c_uint64(seed & ((1 << 64) - 1))
    )


def manual_seed_all(seed: int):
    native.shim().tmb_rng_manual_seed(
        ctypes.c_int32(-1), ctypes.c_uint64(seed & ((1 << 64) - 1))
    )


def seed():
    manual_seed(
        int.from_bytes(torch.randint(0, 2**62, (1,)).numpy().tobytes(), "little")
    )


def seed_all():
    manual_seed_all(
        int.from_bytes(torch.randint(0, 2**62, (1,)).numpy().tobytes(), "little")
    )


def initial_seed() -> int:
    return int.from_bytes(bytes(get_rng_state().tolist()[:8]), "little")


def get_rng_state(device: int | str | torch.device | None = None) -> torch.Tensor:
    """16 bytes: Philox seed then counter, little-endian."""
    buf = (ctypes.c_uint8 * 16)()
    native.shim().tmb_rng_get_state(ctypes.c_int32(_index(device)), buf)
    return torch.tensor(list(buf), dtype=torch.uint8)


def set_rng_state(
    new_state: torch.Tensor, device: int | str | torch.device | None = None
):
    if not isinstance(new_state, torch.Tensor):
        raise TypeError("Mojo RNG state must be a torch.Tensor")
    state = new_state.detach().cpu().contiguous()
    if state.dtype != torch.uint8 or state.numel() != 16:
        raise ValueError("Mojo RNG state must be a 16-element uint8 tensor")
    buf = (ctypes.c_uint8 * 16)(*state.reshape(-1).tolist())
    native.shim().tmb_rng_set_state(ctypes.c_int32(_index(device)), buf)


def get_rng_state_all() -> list[torch.Tensor]:
    return [get_rng_state(i) for i in range(device_count())]


def set_rng_state_all(states: list[torch.Tensor]):
    for i, s in enumerate(states):
        set_rng_state(s, i)


def get_amp_supported_dtype() -> list[torch.dtype]:
    return [torch.float16, torch.bfloat16]


def is_bf16_supported() -> bool:
    return True


def current_stream(device: int | str | torch.device | None = None) -> torch.Stream:
    return torch.accelerator.current_stream(_index(device))


def default_stream(device: int | str | torch.device | None = None) -> torch.Stream:
    """Stream id 0 is every device's default stream."""
    return torch.Stream(
        stream_id=0, device_index=_index(device), device_type=_PRIVATEUSE1
    )


def stream_native_handle(stream: torch.Stream) -> int:
    """The vendor (CUDA/HIP) handle of a mojo stream, for code that launches
    on it outside torch (Triton). `torch.Stream.native_handle` is the same
    value on torch >= 2.11; this works on any version."""
    fn = native.shim().tmb_stream_native_handle
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_int32, ctypes.c_int64]
    return fn(stream.device_index, stream.stream_id) or 0


def set_stream(stream: torch.Stream):
    torch.accelerator.set_stream(stream)


class StreamContext:
    def __init__(self, stream: torch.Stream | None):
        self.stream = stream
        self.prev: torch.Stream | None = None
        self.prev_device: int | None = None

    def __enter__(self):
        if self.stream is None:
            return
        # set_stream also makes the stream's device current (torch semantics):
        # both are restored on exit
        self.prev_device = current_device()
        self.prev = torch.accelerator.current_stream(self.stream.device.index)
        torch.accelerator.set_stream(self.stream)

    def __exit__(self, *exc: object) -> bool:
        if self.prev is not None:
            torch.accelerator.set_stream(self.prev)
        if self.prev_device is not None:
            set_device(self.prev_device)
        return False


def stream(stream: torch.Stream | None) -> StreamContext:
    return StreamContext(stream)


Stream = torch.Stream
Event = torch.Event
